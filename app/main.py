# app/main.py
import asyncio
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import logfire

from app.config import settings
from app.schemas import ChatRequest, ChatResponse
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

@app.get("/health/ready")
async def readiness_check():
    if not getattr(app.state, "ready", False):
        raise HTTPException(status_code=503, detail="Server is warming up models")
    return {"status": "ready"}

@app.post("/v1/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not getattr(app.state, "ready", False):
        raise HTTPException(status_code=503, detail="Server is warming up models")
    try:
        with logfire.span("Chat Request Execution", session_id=request.session_id, query=request.query):
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