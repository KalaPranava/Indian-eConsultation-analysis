"""
Summarization API routes.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Union
import logging

from ..schemas import (
    TextInput, TextListInput, SummarizationResponse,
    SummarizationRequest, SummarizationMethod
)
from src.inference import InferencePipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/summarization", tags=["summarization"])


def get_pipeline():
    """Get inference pipeline instance."""
    # This would be injected from the main app
    pass


@router.post("/generate", response_model=Union[SummarizationResponse, List[SummarizationResponse]])
async def generate_summary(
    request: Union[TextInput, TextListInput, SummarizationRequest],
    method: str = "extractive",
    pipeline: InferencePipeline = Depends(get_pipeline)
):
    """
    Generate summaries for text(s).
    Supports extractive, abstractive, and hybrid methods.
    """
    try:
        # Validate method
        if method not in ["extractive", "abstractive", "hybrid"]:
            raise HTTPException(
                status_code=400,
                detail="Method must be one of: extractive, abstractive, hybrid"
            )
            
        # Handle different request types
        if isinstance(request, SummarizationRequest):
            texts = [request.text]
            method = request.method.value
            num_sentences = request.num_sentences
            max_length = request.max_length
            min_length = request.min_length
            preprocess = request.preprocess
        elif isinstance(request, TextInput):
            texts = [request.text]
            num_sentences = 3
            max_length = 128
            min_length = 30
            preprocess = True
        else:  # TextListInput
            texts = request.texts
            num_sentences = 3
            max_length = 128
            min_length = 30
            preprocess = True
            
        # Perform summarization
        results = pipeline.generate_summary(
            texts,
            method=method,
            num_sentences=num_sentences,
            max_length=max_length,
            min_length=min_length,
            preprocess=preprocess
        )
        
        # Format response
        if len(texts) == 1:
            result = results[0] if isinstance(results, list) else results
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


@router.post("/extractive")
async def extractive_summarization(
    request: Union[TextInput, TextListInput],
    num_sentences: int = 3,
    preprocess: bool = True,
    pipeline: InferencePipeline = Depends(get_pipeline)
):
    """
    Generate extractive summaries using TextRank algorithm.
    """
    try:
        texts = [request.text] if isinstance(request, TextInput) else request.texts
        
        results = pipeline.generate_summary(
            texts,
            method="extractive",
            num_sentences=num_sentences,
            preprocess=preprocess
        )
        
        return {
            "method": "extractive",
            "summaries": [
                {
                    "text": result['original_text'],
                    "summary": result['summary'],
                    "sentences": result.get('sentences', []),
                    "compression_ratio": result.get('compression_ratio', 0.0)
                }
                for result in (results if isinstance(results, list) else [results])
            ]
        }
        
    except Exception as e:
        logger.error(f"Error in extractive summarization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/abstractive")
async def abstractive_summarization(
    request: Union[TextInput, TextListInput],
    max_length: int = 128,
    min_length: int = 30,
    preprocess: bool = True,
    pipeline: InferencePipeline = Depends(get_pipeline)
):
    """
    Generate abstractive summaries using transformer models.
    """
    try:
        texts = [request.text] if isinstance(request, TextInput) else request.texts
        
        results = pipeline.generate_summary(
            texts,
            method="abstractive",
            max_length=max_length,
            min_length=min_length,
            preprocess=preprocess
        )
        
        return {
            "method": "abstractive",
            "summaries": [
                {
                    "text": result['original_text'],
                    "summary": result['summary'],
                    "compression_ratio": result.get('compression_ratio', 0.0)
                }
                for result in (results if isinstance(results, list) else [results])
            ]
        }
        
    except Exception as e:
        logger.error(f"Error in abstractive summarization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hybrid")
async def hybrid_summarization(
    request: Union[TextInput, TextListInput],
    extractive_sentences: int = 5,
    max_length: int = 128,
    min_length: int = 30,
    preprocess: bool = True,
    pipeline: InferencePipeline = Depends(get_pipeline)
):
    """
    Generate hybrid summaries combining extractive and abstractive methods.
    """
    try:
        texts = [request.text] if isinstance(request, TextInput) else request.texts
        
        results = pipeline.generate_summary(
            texts,
            method="hybrid",
            extractive_sentences=extractive_sentences,
            max_length=max_length,
            min_length=min_length,
            preprocess=preprocess
        )
        
        return {
            "method": "hybrid",
            "summaries": [
                {
                    "text": result['original_text'],
                    "summary": result['summary'],
                    "extractive_summary": result.get('extractive_summary', ''),
                    "compression_ratio": result.get('compression_ratio', 0.0)
                }
                for result in (results if isinstance(results, list) else [results])
            ]
        }
        
    except Exception as e:
        logger.error(f"Error in hybrid summarization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/methods")
async def get_summarization_methods():
    """Get available summarization methods and their descriptions."""
    return {
        "methods": {
            "extractive": {
                "description": "Extract important sentences from original text using TextRank",
                "parameters": ["num_sentences"],
                "best_for": "Preserving original wording and key phrases"
            },
            "abstractive": {
                "description": "Generate new summary text using transformer models",
                "parameters": ["max_length", "min_length"],
                "best_for": "Creating concise, natural summaries"
            },
            "hybrid": {
                "description": "Combine extractive and abstractive approaches",
                "parameters": ["extractive_sentences", "max_length", "min_length"],
                "best_for": "Balancing accuracy and naturalness"
            }
        }
    }


@router.post("/evaluate")
async def evaluate_summary_quality(
    original_text: str,
    summary: str,
    reference_summary: str = None
):
    """
    Evaluate summary quality with various metrics.
    """
    try:
        # Basic metrics
        original_words = len(original_text.split())
        summary_words = len(summary.split())
        compression_ratio = summary_words / original_words if original_words > 0 else 0
        
        # Content overlap
        original_words_set = set(original_text.lower().split())
        summary_words_set = set(summary.lower().split())
        content_overlap = len(original_words_set & summary_words_set) / len(original_words_set) if original_words_set else 0
        
        metrics = {
            "compression_ratio": compression_ratio,
            "content_overlap": content_overlap,
            "original_length": original_words,
            "summary_length": summary_words
        }
        
        # Reference-based metrics if provided
        if reference_summary:
            reference_words_set = set(reference_summary.lower().split())
            rouge_1 = len(summary_words_set & reference_words_set) / len(reference_words_set) if reference_words_set else 0
            metrics["rouge_1_recall"] = rouge_1
            
        return {
            "metrics": metrics,
            "quality_score": (content_overlap + (1 - abs(0.3 - compression_ratio))) / 2  # Simple quality score
        }
        
    except Exception as e:
        logger.error(f"Error in summary evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))