import asyncio
import hashlib
import time
import uuid
from contextlib import asynccontextmanager

import logfire
from fastapi import Depends, FastAPI, HTTPException, Request, Response

from app.config import settings
from app.pipeline import run_rag_pipeline_async, shutdown_pipeline, warmup_pipeline
from app.queue import QueueLease, rag_queue
from app.schemas import ChatRequest, ChatResponse
from app.security import enforce_chat_rate_limit, ensure_session, validate_security_settings
from app.utils.logger import logger
from app.retrieval.qdrant import VectorStoreError


logfire.configure(
    service_name="financial-rag-engine",
    token=settings.LOGFIRE_TOKEN,
    send_to_logfire=bool(settings.LOGFIRE_TOKEN),
)
logfire.instrument_httpx()

REQUEST_TIMEOUT_S = settings.REQUEST_TIMEOUT_S
_QUEUE_SUPERVISORS: set[asyncio.Task] = set()


async def _supervise_pipeline(task: asyncio.Task[ChatResponse], lease: QueueLease) -> None:
    try:
        renew_interval = max(0.5, settings.RAG_CONCURRENCY_LEASE_SECONDS / 3)
        while not task.done():
            done, _ = await asyncio.wait({task}, timeout=renew_interval)
            if not done:
                await rag_queue.renew(lease)
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as error:
            logger.error("RAG task failed after request ended: %s", type(error).__name__)
    finally:
        await rag_queue.release(lease)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    validate_security_settings()
    logger.info("Initializing server models...")

    with logfire.span("Server Startup & Warmup"):
        try:
            await warmup_pipeline()
        except Exception:
            logger.exception("Warmup failed")
            raise

    app.state.ready = True
    logger.info("Server ready")
    yield
    app.state.ready = False
    shutdown_pipeline()


app = FastAPI(
    title=settings.APP_NAME,
    version="3.0.0",
    description="Production-prototype async financial RAG API",
    lifespan=lifespan,
)

logfire.instrument_fastapi(app)


@app.middleware("http")
async def request_tracking_and_session(request: Request, call_next):
    request_id = uuid.uuid4().hex
    started_at = time.perf_counter()
    status_code = 500
    response: Response | None = None

    try:
        # Establish/validate the browser session before dependencies run.
        # Health endpoints also receive a session cookie only when necessary;
        # the cookie itself carries no authorization privileges.
        response = await call_next(request)
        status_code = response.status_code

        # If a handler did not establish a session, create one here. For chat,
        # the endpoint establishes it before the rate-limit dependency executes.
        if not getattr(request.state, "session_id", None):
            ensure_session(request, response)

        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        logger.info(
            "HTTP request",
            extra={
                "extra_data": {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
                }
            },
        )


# A dependency that establishes the session before the rate-limit dependency.
async def session_dependency(request: Request, response: Response) -> str:
    return ensure_session(request, response)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/health/ready")
async def readiness_check():
    if not getattr(app.state, "ready", False):
        raise HTTPException(status_code=503, detail="Server is warming up models")
    return {"status": "ready"}


@app.post(
    "/v1/chat",
    response_model=ChatResponse,
    dependencies=[Depends(session_dependency), Depends(enforce_chat_rate_limit)],
)
async def chat_endpoint(request: Request, chat_request: ChatRequest):
    if not getattr(app.state, "ready", False):
        raise HTTPException(status_code=503, detail="Server is warming up models")

    session_id = getattr(request.state, "session_id", None)
    if not session_id:
        raise HTTPException(status_code=401, detail="Invalid chat session")

    # Inject the server-derived identity into the object consumed by the existing
    # pipeline. The browser cannot control this value because ChatRequest rejects
    # unknown client fields and the endpoint overwrites it from the signed cookie.
    pipeline_request = chat_request.model_copy(update={"session_id": session_id})
    query_hash = hashlib.sha256(chat_request.query.encode("utf-8")).hexdigest()

    deadline = asyncio.get_running_loop().time() + REQUEST_TIMEOUT_S
    try:
        lease = await asyncio.wait_for(
            rag_queue.acquire(),
            timeout=max(0, deadline - asyncio.get_running_loop().time()),
        )
    except asyncio.TimeoutError:
        logfire.error("Request timed out while waiting for a RAG slot", session_id=session_id)
        raise HTTPException(status_code=504, detail="Request timed out")

    pipeline_task = asyncio.create_task(run_rag_pipeline_async(pipeline_request))
    supervisor = asyncio.create_task(_supervise_pipeline(pipeline_task, lease))
    _QUEUE_SUPERVISORS.add(supervisor)
    supervisor.add_done_callback(_QUEUE_SUPERVISORS.discard)
    try:
        with logfire.span(
            "Chat Request Execution",
            session_id=session_id,
            query_length=len(chat_request.query),
            query_hash=query_hash,
            rag_slot=lease.member,
        ):
            return await asyncio.wait_for(
                asyncio.shield(pipeline_task),
                timeout=max(0, deadline - asyncio.get_running_loop().time()),
            )
    except asyncio.TimeoutError:
        logfire.error("Request timed out", session_id=session_id)
        raise HTTPException(status_code=504, detail="Request timed out")
    except VectorStoreError as error:
        logger.error("Vector store unavailable: %s", type(error.__cause__).__name__ if error.__cause__ else "unknown")
        raise HTTPException(status_code=503, detail="Financial document search is temporarily unavailable") from error
    except HTTPException:
        raise
    except Exception as error:
        logfire.error("Internal processing error", error_type=type(error).__name__)
        raise HTTPException(status_code=500, detail="Internal processing error")
    finally:
        if pipeline_task.done():
            await supervisor


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
