# app/main.py
import asyncio
import hashlib
import os
import time
import uuid
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, HTTPException, Request
import logfire
from app.utils.cache import init_financial_cache
from app.config import settings
from app.schemas import ChatRequest, ChatResponse
from app.security import enforce_chat_rate_limit, validate_security_settings
from app.pipeline import run_rag_pipeline_async, shutdown_pipeline, warmup_pipeline
from app.utils.logger import logger

# Configure Logfire Observability
logfire.configure(
    service_name="financial-rag-engine",
    token=settings.LOGFIRE_TOKEN,
    send_to_logfire=bool(settings.LOGFIRE_TOKEN)
)

REQUEST_TIMEOUT_S = float(os.getenv("REQUEST_TIMEOUT_S", "40"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    validate_security_settings()
    logger.info("Initializing server models...")
    with logfire.span("Server Startup & Warmup"):
        try:
            await warmup_pipeline()
            await asyncio.to_thread(init_financial_cache)
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
    version="2.3.0",
    description="High-Throughput Async RAG Engine with FlashRank & Logfire",
    lifespan=lifespan,
)

# Automatically instrument FastAPI requests for Logfire tracing
logfire.instrument_fastapi(app)
logfire.instrument_httpx()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.middleware("http")
async def request_tracking(request: Request, call_next):
    request_id = uuid.uuid4().hex
    started_at = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
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


@app.get("/health/ready")
async def readiness_check():
    if not getattr(app.state, "ready", False):
        raise HTTPException(status_code=503, detail="Server is warming up models")
    return {"status": "ready"}

@app.post("/v1/chat", response_model=ChatResponse, dependencies=[Depends(enforce_chat_rate_limit)])
async def chat_endpoint(request: ChatRequest):
    if not getattr(app.state, "ready", False):
        raise HTTPException(status_code=503, detail="Server is warming up models")
    try:
        query_hash = hashlib.sha256(request.query.encode("utf-8")).hexdigest()
        with logfire.span(
            "Chat Request Execution",
            session_id=request.session_id,
            query_length=len(request.query),
            query_hash=query_hash,
        ):
            return await asyncio.wait_for(run_rag_pipeline_async(request), timeout=REQUEST_TIMEOUT_S)
    except asyncio.TimeoutError:
        logfire.error("Request timed out", session_id=request.session_id)
        raise HTTPException(status_code=504, detail="Request timed out")
    except HTTPException:
        raise
    except Exception as e:
        logfire.exception("Internal processing error: {error}", error=str(e))
        raise HTTPException(status_code=500, detail="Internal processing error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)