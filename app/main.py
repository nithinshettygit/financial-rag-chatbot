from fastapi import FastAPI, HTTPException
from app.config import settings
from app.schemas import ChatRequest, ChatResponse
from app.pipeline import run_rag_pipeline
from app.utils.logger import logger

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="Stateless Production API for Docling RAG Engine"
)

@app.get("/health")
def health_check():
    return {"status": "healthy", "environment": settings.ENVIRONMENT}

@app.post("/v1/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        logger.info(f"Processing query for session: {request.session_id}")
        response = run_rag_pipeline(request)
        return response
    except Exception as e:
        logger.error(f"Error processing request for session {request.session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal processing error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)