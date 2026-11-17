import os
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Header, Depends, status
from pydantic import BaseModel, Field, validator
from starlette.concurrency import run_in_threadpool

from summarize import Summarizer

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "summarizer_ai_model.gguf")
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "my_super_secret_key")
MAX_CONTEXT_LENGTH = 512

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("summarizer-api")

# Global summarizer instance
summarizer: Optional[Summarizer] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global summarizer
    try:
        logger.info(f"Initializing model from {MODEL_PATH}...")
        summarizer = Summarizer(MODEL_PATH, n_ctx=MAX_CONTEXT_LENGTH)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # We don't raise here to allow the app to start and show health errors, 
        # but in production, you might want to crash.
    yield
    logger.info("Shutting down...")
    # Clean up resources if needed

app = FastAPI(
    title="AI Summarizer API",
    description="Local AI Summarization API using T5 and llama.cpp",
    version="1.1.0",
    lifespan=lifespan
)

# Request/Response Models
class SummaryRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to be summarized")
    instructions: Optional[str] = Field(None, description="Specific instructions for summarization")
    min_tokens: int = Field(10, ge=1, le=512, description="Minimum tokens in the output")
    max_tokens: int = Field(128, ge=1, le=512, description="Maximum tokens in the output")
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="Sampling temperature")

    @validator("max_tokens")
    def validate_tokens(cls, v, values):
        if "min_tokens" in values and v < values["min_tokens"]:
            raise ValueError("max_tokens must be greater than or equal to min_tokens")
        return v

class SummaryResponse(BaseModel):
    summary: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    processing_time_ms: int

# Security
async def verify_secret_key(x_secret_key: str = Header(..., alias="X-Secret-Key")):
    if x_secret_key != API_SECRET_KEY:
        logger.warning(f"Unauthorized access attempt with key: {x_secret_key}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing Secret Key"
        )
    return x_secret_key

@app.post("/summarize", response_model=SummaryResponse)
async def summarize_endpoint(
    request: SummaryRequest,
    secret_key: str = Depends(verify_secret_key)
):
    if summarizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded"
        )

    try:
        # Pre-validate token length
        input_tokens = summarizer.tokenize(request.text)
        if len(input_tokens) > MAX_CONTEXT_LENGTH - 50: # Leave some buffer
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Input text is too long ({len(input_tokens)} tokens). Max is {MAX_CONTEXT_LENGTH - 50}."
            )

        # Run inference in a thread pool to avoid blocking the async event loop
        result = await run_in_threadpool(
            summarizer.summarize,
            text=request.text,
            instructions=request.instructions,
            min_tokens=request.min_tokens,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        return SummaryResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Inference error occurred")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during summarization"
        )

@app.get("/health")
async def health_check():
    if summarizer is None:
        return {"status": "unhealthy", "reason": "model_not_loaded"}
    return {"status": "healthy", "model": MODEL_PATH}

if __name__ == "__main__":
    import uvicorn
    # Use environment variables for host/port or defaults
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run(app, host=host, port=port)
