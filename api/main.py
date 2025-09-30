"""
FastAPI application for Indian e-consultation sentiment analysis and summarization.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn
from typing import List, Dict, Any, Optional, Union
import logging
from datetime import datetime
import os
import re
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.inference import InferencePipeline
from .schemas import (
    TextInput, TextListInput, SentimentResponse, EmotionResponse,
    SummarizationResponse, ComprehensiveResponse, HealthResponse,
    ProcessingStats, BatchProcessingRequest
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Indian E-Consultation Analysis API",
    description="API for sentiment analysis, emotion detection, and text summarization of Indian e-consultation comments",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-project.vercel.app",  # Replace with your actual Vercel URL
        "http://localhost:3000",  # Keep for local development
        "http://127.0.0.1:3000"   # Keep for local development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference pipeline instance
pipeline: Optional[InferencePipeline] = None


def get_pipeline() -> InferencePipeline:
    """Get or initialize the inference pipeline."""
    global pipeline
    if pipeline is None:
        logger.info("Initializing inference pipeline...")
        pipeline = InferencePipeline()
        logger.info("Inference pipeline initialized successfully")
    return pipeline

@app.post("/analyze/overall_summary")
async def overall_summary(payload: dict | list | TextListInput = Body(...), pipeline: InferencePipeline = Depends(get_pipeline)):
    """Generate an overall summary + distributions and key insights for a list of texts.

    Accepts either:
      {"texts": ["...", "..."]}  OR  ["...", "..."] for convenience.
    """
    try:
        # Normalize incoming payload to list of strings
        texts: list[str] = []
        if isinstance(payload, TextListInput):
            texts = payload.texts
        elif isinstance(payload, dict) and 'texts' in payload:
            raw = payload.get('texts')
            if isinstance(raw, list):
                texts = [str(t) for t in raw]
        elif isinstance(payload, list):
            texts = [str(t) for t in payload]

        if not texts:
            raise HTTPException(status_code=400, detail="No texts provided (expected 'texts' list or raw list)")

        # Limit to a reasonable sample to avoid extreme latency
        sample_texts = texts[:500]

        # Sanitize control characters
        def _clean(txt: str) -> str:
            return re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', ' ', txt).strip()
        sample_texts = [_clean(t) for t in sample_texts if _clean(t)]

        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        emotion_counts: dict[str, int] = {}

        # Basic processing (could be optimized with batching inside pipeline)
        for t in sample_texts:
            try:
                s = pipeline.analyze_sentiment(t, preprocess=True)
                label = s.get("label", "neutral")
                if label not in sentiment_counts:
                    label = "neutral"
                sentiment_counts[label] += 1
            except Exception:
                sentiment_counts["neutral"] += 1
            try:
                e = pipeline.detect_emotions(t, preprocess=True)
                primary = e.get("primary_emotion", "neutral")
                emotion_counts[primary] = emotion_counts.get(primary, 0) + 1
            except Exception:
                emotion_counts["neutral"] = emotion_counts.get("neutral", 0) + 1

        total = len(sample_texts)
        if total == 0:
            raise HTTPException(status_code=400, detail="No valid texts after preprocessing")

        positive_ratio = sentiment_counts["positive"] / total
        negative_ratio = sentiment_counts["negative"] / total

        if positive_ratio > 0.6:
            satisfaction_level = "High"
        elif positive_ratio > 0.4:
            satisfaction_level = "Moderate"
        else:
            satisfaction_level = "Low"

        # Simple primary concerns heuristic: look at most frequent words inside negative texts
        import re
        stop = set(["the","is","a","an","and","or","of","to","in","on","for","with","this","that","are","was","were","be","it","at","by","from","as","have","has","had","but","if","not","we","you","they","i","he","she","his","her","their","our"])
        concern_counts: dict[str,int] = {}
        for t in sample_texts:
            # reuse sentiment classification decision quickly
            # crude: re-run (small cost) to decide negative body words weight
            try:
                s = pipeline.analyze_sentiment(t, preprocess=True)
                if s.get("label") == "negative":
                    tokens = re.findall(r"[A-Za-z\u0900-\u097F]{3,}", t.lower())
                    for tok in tokens:
                        if tok in stop:
                            continue
                        concern_counts[tok] = concern_counts.get(tok, 0) + 1
            except Exception:
                continue

        top_concerns = ", ".join([w for w,_ in sorted(concern_counts.items(), key=lambda x: x[1], reverse=True)[:5]]) or "None prominent"

        # Build combined text for summarization (truncate to avoid model limits)
        combined_text = " \n".join(sample_texts)
        if len(combined_text) > 12000:
            combined_text = combined_text[:12000]

        summary_text = ""
        try:
            summary_res = pipeline.generate_summary(
                combined_text,
                method="extractive",
                num_sentences=5,
                max_length=256,
                min_length=60,
                preprocess=True
            )
            summary_text = summary_res.get("summary", "")
        except Exception:
            # fallback heuristic summary
            summary_text = (
                f"Across {total} comments: {int(positive_ratio*100)}% positive, "
                f"{int(negative_ratio*100)}% negative. Dominant emotion: "
                f"{max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else 'neutral'}." 
            )

        response = {
            "overall_summary": summary_text,
            "sentiment_distribution": sentiment_counts,
            "emotion_distribution": emotion_counts,
            "key_insights": {
                "satisfaction_level": satisfaction_level,
                "primary_concerns": top_concerns
            },
            "total_comments": total
        }
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating overall summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting Indian E-Consultation Analysis API")
    # Pre-initialize pipeline
    get_pipeline()
    logger.info("API startup completed")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down Indian E-Consultation Analysis API")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information."""
    return """
    <html>
        <head>
            <title>Indian E-Consultation Analysis API</title>
        </head>
        <body>
            <h1>Indian E-Consultation Analysis API</h1>
            <p>Welcome to the Indian E-Consultation Analysis API for sentiment analysis, emotion detection, and text summarization.</p>
            <h2>Features:</h2>
            <ul>
                <li>Multi-language support (Hindi, English, code-mixed)</li>
                <li>Sentiment analysis using IndicBERT</li>
                <li>Emotion detection with 6 emotion categories</li>
                <li>Text summarization (extractive and abstractive)</li>
                <li>Batch processing capabilities</li>
            </ul>
            <h2>API Documentation:</h2>
            <ul>
                <li><a href="/docs">Swagger UI</a></li>
                <li><a href="/redoc">ReDoc</a></li>
            </ul>
        </body>
    </html>
    """


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        pipeline = get_pipeline()
        stats = pipeline.get_stats()
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(),
            version="1.0.0",
            uptime_requests=stats.get('total_requests', 0)
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.get("/stats", response_model=ProcessingStats)
async def get_processing_stats(pipeline: InferencePipeline = Depends(get_pipeline)):
    """Get processing statistics."""
    stats = pipeline.get_stats()
    return ProcessingStats(**stats)


@app.post("/sentiment/analyze", response_model=Union[SentimentResponse, List[SentimentResponse]])
async def analyze_sentiment(
    request: Union[TextInput, TextListInput],
    return_probabilities: bool = False,
    preprocess: bool = True,
    pipeline: InferencePipeline = Depends(get_pipeline)
):
    """
    Analyze sentiment of text(s).
    
    - **text/texts**: Input text(s) to analyze
    - **return_probabilities**: Whether to return confidence scores for all classes
    - **preprocess**: Whether to preprocess the text
    """
    try:
        if isinstance(request, TextInput):
            # Single text
            result = pipeline.analyze_sentiment(
                request.text,
                preprocess=preprocess,
                return_probabilities=return_probabilities
            )
            return SentimentResponse(
                text=result['original_text'],
                sentiment=result['label'],
                confidence=result['confidence'],
                scores=result.get('scores', {}),
                processing_time=0  # Would need to track this
            )
        else:
            # Multiple texts
            results = pipeline.analyze_sentiment(
                request.texts,
                preprocess=preprocess,
                return_probabilities=return_probabilities
            )
            return [
                SentimentResponse(
                    text=result['original_text'],
                    sentiment=result['label'],
                    confidence=result['confidence'],
                    scores=result.get('scores', {}),
                    processing_time=0
                )
                for result in results
            ]
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/emotion/detect", response_model=Union[EmotionResponse, List[EmotionResponse]])
async def detect_emotions(
    request: Union[TextInput, TextListInput],
    top_k: int = 3,
    threshold: float = 0.1,
    preprocess: bool = True,
    pipeline: InferencePipeline = Depends(get_pipeline)
):
    """
    Detect emotions in text(s).
    
    - **text/texts**: Input text(s) to analyze
    - **top_k**: Number of top emotions to return
    - **threshold**: Minimum confidence threshold
    - **preprocess**: Whether to preprocess the text
    """
    try:
        if isinstance(request, TextInput):
            # Single text
            result = pipeline.detect_emotions(
                request.text,
                preprocess=preprocess,
                top_k=top_k,
                threshold=threshold
            )
            return EmotionResponse(
                text=result['original_text'],
                primary_emotion=result['primary_emotion'],
                confidence=result['primary_confidence'],
                all_emotions=result['all_emotions'],
                emotion_scores=result.get('emotion_scores', {}),
                processing_time=0
            )
        else:
            # Multiple texts
            results = pipeline.detect_emotions(
                request.texts,
                preprocess=preprocess,
                top_k=top_k,
                threshold=threshold
            )
            return [
                EmotionResponse(
                    text=result['original_text'],
                    primary_emotion=result['primary_emotion'],
                    confidence=result['primary_confidence'],
                    all_emotions=result['all_emotions'],
                    emotion_scores=result.get('emotion_scores', {}),
                    processing_time=0
                )
                for result in results
            ]
    except Exception as e:
        logger.error(f"Error in emotion detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarization/generate", response_model=Union[SummarizationResponse, List[SummarizationResponse]])
async def generate_summary(
    request: Union[TextInput, TextListInput],
    method: str = "extractive",
    num_sentences: int = 3,
    max_length: int = 128,
    min_length: int = 30,
    preprocess: bool = True,
    pipeline: InferencePipeline = Depends(get_pipeline)
):
    """
    Generate summaries for text(s).
    
    - **text/texts**: Input text(s) to summarize
    - **method**: Summarization method ('extractive', 'abstractive', 'hybrid')
    - **num_sentences**: Number of sentences for extractive summarization
    - **max_length**: Maximum length for abstractive summarization
    - **min_length**: Minimum length for abstractive summarization
    - **preprocess**: Whether to preprocess the text
    """
    try:
        if method not in ["extractive", "abstractive", "hybrid"]:
            raise HTTPException(
                status_code=400, 
                detail="Method must be one of: extractive, abstractive, hybrid"
            )
            
        if isinstance(request, TextInput):
            # Single text
            result = pipeline.generate_summary(
                request.text,
                method=method,
                num_sentences=num_sentences,
                max_length=max_length,
                min_length=min_length,
                preprocess=preprocess
            )
            return SummarizationResponse(
                text=result['original_text'],
                summary=result['summary'],
                method=result['method'],
                compression_ratio=result.get('compression_ratio', 0.0),
                original_length=result.get('original_length', 0),
                summary_length=result.get('summary_length', 0),
                processing_time=0
            )
        else:
            # Multiple texts
            results = pipeline.generate_summary(
                request.texts,
                method=method,
                num_sentences=num_sentences,
                max_length=max_length,
                min_length=min_length,
                preprocess=preprocess
            )
            return [
                SummarizationResponse(
                    text=result['original_text'],
                    summary=result['summary'],
                    method=result['method'],
                    compression_ratio=result.get('compression_ratio', 0.0),
                    original_length=result.get('original_length', 0),
                    summary_length=result.get('summary_length', 0),
                    processing_time=0
                )
                for result in results
            ]
    except Exception as e:
        logger.error(f"Error in summarization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/comprehensive", response_model=Union[ComprehensiveResponse, List[ComprehensiveResponse]])
async def comprehensive_analysis(
    request: Union[TextInput, TextListInput],
    include_sentiment: bool = True,
    include_emotion: bool = True,
    include_summary: bool = True,
    summary_method: str = "extractive",
    preprocess: bool = True,
    pipeline: InferencePipeline = Depends(get_pipeline)
):
    """
    Perform comprehensive analysis including sentiment, emotion, and summarization.
    
    - **text/texts**: Input text(s) to analyze
    - **include_sentiment**: Whether to include sentiment analysis
    - **include_emotion**: Whether to include emotion detection
    - **include_summary**: Whether to include summarization
    - **summary_method**: Summarization method
    - **preprocess**: Whether to preprocess the text
    """
    try:
        if isinstance(request, TextInput):
            # Single text
            result = pipeline.analyze_comprehensive(
                request.text,
                include_sentiment=include_sentiment,
                include_emotion=include_emotion,
                include_summary=include_summary,
                summary_method=summary_method,
                preprocess=preprocess
            )
            return ComprehensiveResponse(
                text=result['original_text'],
                sentiment=result.get('sentiment', {}),
                emotion=result.get('emotion', {}),
                summary=result.get('summary', {}),
                timestamp=result['timestamp'],
                processing_time=0
            )
        else:
            # Multiple texts
            results = pipeline.analyze_comprehensive(
                request.texts,
                include_sentiment=include_sentiment,
                include_emotion=include_emotion,
                include_summary=include_summary,
                summary_method=summary_method,
                preprocess=preprocess
            )
            return [
                ComprehensiveResponse(
                    text=result['original_text'],
                    sentiment=result.get('sentiment', {}),
                    emotion=result.get('emotion', {}),
                    summary=result.get('summary', {}),
                    timestamp=result['timestamp'],
                    processing_time=0
                )
                for result in results
            ]
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process/batch")
async def process_batch(
    request: BatchProcessingRequest,
    background_tasks: BackgroundTasks,
    pipeline: InferencePipeline = Depends(get_pipeline)
):
    """
    Process a batch of texts asynchronously.
    
    - **texts**: List of texts to process
    - **task_types**: List of tasks to perform ('sentiment', 'emotion', 'summary')
    - **output_format**: Output format ('json' or 'csv')
    """
    try:
        # For now, process synchronously (could be made async with task queues)
        results = []
        
        for text in request.texts:
            result = {"text": text}
            
            if "sentiment" in request.task_types:
                sentiment_result = pipeline.analyze_sentiment(text, preprocess=request.preprocess)
                result["sentiment"] = sentiment_result
                
            if "emotion" in request.task_types:
                emotion_result = pipeline.detect_emotions(text, preprocess=request.preprocess)
                result["emotion"] = emotion_result
                
            if "summary" in request.task_types:
                summary_result = pipeline.generate_summary(text, preprocess=request.preprocess)
                result["summary"] = summary_result
                
            results.append(result)
            
        return {
            "status": "completed",
            "processed_count": len(results),
            "results": results,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/stats/reset")
async def reset_stats(pipeline: InferencePipeline = Depends(get_pipeline)):
    """Reset processing statistics."""
    pipeline.reset_stats()
    return {"message": "Statistics reset successfully"}


if __name__ == "__main__":
    # Get configuration from environment variables
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    workers = int(os.getenv("API_WORKERS", "1"))
    log_level = os.getenv("LOG_LEVEL", "info")
    
    # Run the application
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        reload=False  # Set to True for development
    )