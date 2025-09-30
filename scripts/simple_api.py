"""
Minimal API server for testing - bypasses complex model loading
"""
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Indian E-Consultation Analysis API (Minimal)",
    description="Simplified API for testing",
    version="1.0.0"
)

class TextRequest(BaseModel):
    text: str

class AnalysisResponse(BaseModel):
    text: str
    result: str
    status: str = "success"

@app.get("/")
async def root():
    return {"message": "Indian E-Consultation Analysis API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "API is running",
        "version": "1.0.0"
    }

@app.post("/analyze/sentiment", response_model=AnalysisResponse)
async def analyze_sentiment(request: TextRequest):
    # Simple mock sentiment analysis
    text = request.text.lower()
    if any(word in text for word in ['good', 'excellent', 'great', 'अच्छी', 'बहुत']):
        sentiment = "positive"
    elif any(word in text for word in ['bad', 'poor', 'terrible', 'खराब']):
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    return AnalysisResponse(
        text=request.text,
        result=f"Sentiment: {sentiment}",
        status="success"
    )

@app.post("/analyze/emotion", response_model=AnalysisResponse)
async def analyze_emotion(request: TextRequest):
    # Simple mock emotion detection
    text = request.text.lower()
    if any(word in text for word in ['happy', 'joy', 'खुश']):
        emotion = "joy"
    elif any(word in text for word in ['sad', 'unhappy', 'दुखी']):
        emotion = "sadness"
    elif any(word in text for word in ['angry', 'mad', 'गुस्सा']):
        emotion = "anger"
    else:
        emotion = "neutral"
    
    return AnalysisResponse(
        text=request.text,
        result=f"Emotion: {emotion}",
        status="success"
    )

@app.post("/analyze/summarize", response_model=AnalysisResponse)
async def summarize_text(request: TextRequest):
    # Simple mock summarization
    words = request.text.split()
    if len(words) > 10:
        summary = " ".join(words[:10]) + "..."
    else:
        summary = request.text
    
    return AnalysisResponse(
        text=request.text,
        result=f"Summary: {summary}",
        status="success"
    )

if __name__ == "__main__":
    logger.info("Starting minimal API server...")
    uvicorn.run(
        "simple_api:app",
        host="127.0.0.1",  # Changed to localhost only
        port=8000,
        reload=False,
        log_level="info"
    )