"""
Sentiment analysis API routes.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Union
import logging

from ..schemas import (
    TextInput, TextListInput, SentimentResponse,
    SentimentAnalysisRequest
)
from src.inference import InferencePipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sentiment", tags=["sentiment"])


def get_pipeline():
    """Get inference pipeline instance."""
    # This would be injected from the main app
    pass


@router.post("/analyze", response_model=Union[SentimentResponse, List[SentimentResponse]])
async def analyze_sentiment(
    request: Union[TextInput, TextListInput, SentimentAnalysisRequest],
    pipeline: InferencePipeline = Depends(get_pipeline)
):
    """
    Analyze sentiment of text(s).
    Supports single text, multiple texts, or detailed request format.
    """
    try:
        # Handle different request types
        if isinstance(request, SentimentAnalysisRequest):
            texts = [request.text]
            return_probabilities = request.return_probabilities
            preprocess = request.preprocess
        elif isinstance(request, TextInput):
            texts = [request.text]
            return_probabilities = False
            preprocess = True
        else:  # TextListInput
            texts = request.texts
            return_probabilities = False
            preprocess = True
            
        # Perform analysis
        results = pipeline.analyze_sentiment(
            texts,
            preprocess=preprocess,
            return_probabilities=return_probabilities
        )
        
        # Format response
        if len(texts) == 1:
            result = results[0] if isinstance(results, list) else results
            return SentimentResponse(
                text=result['original_text'],
                sentiment=result['label'],
                confidence=result['confidence'],
                scores=result.get('scores', {}),
                processing_time=0
            )
        else:
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


@router.post("/batch")
async def batch_sentiment_analysis(
    texts: List[str],
    return_probabilities: bool = False,
    preprocess: bool = True,
    batch_size: int = 32,
    pipeline: InferencePipeline = Depends(get_pipeline)
):
    """
    Batch sentiment analysis for multiple texts.
    """
    try:
        if len(texts) > 1000:
            raise HTTPException(
                status_code=400,
                detail="Maximum batch size is 1000 texts"
            )
            
        results = pipeline.analyze_sentiment(
            texts,
            preprocess=preprocess,
            return_probabilities=return_probabilities,
            batch_size=batch_size
        )
        
        return {
            "status": "success",
            "processed_count": len(results),
            "results": [
                {
                    "text": result['original_text'],
                    "sentiment": result['label'],
                    "confidence": result['confidence'],
                    "scores": result.get('scores', {})
                }
                for result in results
            ]
        }
        
    except Exception as e:
        logger.error(f"Error in batch sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/supported_labels")
async def get_supported_sentiment_labels():
    """Get list of supported sentiment labels."""
    return {
        "labels": ["positive", "negative", "neutral"],
        "description": {
            "positive": "Positive sentiment",
            "negative": "Negative sentiment", 
            "neutral": "Neutral sentiment"
        }
    }