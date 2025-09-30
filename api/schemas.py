"""
Pydantic schemas for API request and response models.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum


class TextInput(BaseModel):
    """Single text input model."""
    text: str = Field(..., min_length=1, max_length=5000, description="Input text to analyze")
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty')
        return v


class TextListInput(BaseModel):
    """Multiple texts input model."""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to analyze")
    
    @validator('texts')
    def texts_not_empty(cls, v):
        if not v:
            raise ValueError('Texts list cannot be empty')
        for text in v:
            if not text.strip():
                raise ValueError('Text cannot be empty')
        return v


class SentimentResponse(BaseModel):
    """Sentiment analysis response model."""
    text: str = Field(..., description="Original input text")
    sentiment: str = Field(..., description="Predicted sentiment (positive, negative, neutral)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    scores: Optional[Dict[str, float]] = Field(None, description="Scores for all sentiment classes")
    processing_time: float = Field(..., ge=0.0, description="Processing time in seconds")


class EmotionInfo(BaseModel):
    """Individual emotion information."""
    emotion: str = Field(..., description="Emotion name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class EmotionResponse(BaseModel):
    """Emotion detection response model."""
    text: str = Field(..., description="Original input text")
    primary_emotion: str = Field(..., description="Primary detected emotion")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence for primary emotion")
    all_emotions: List[EmotionInfo] = Field(..., description="All detected emotions above threshold")
    emotion_scores: Optional[Dict[str, float]] = Field(None, description="Scores for all emotion classes")
    processing_time: float = Field(..., ge=0.0, description="Processing time in seconds")


class SummarizationMethod(str, Enum):
    """Summarization methods."""
    extractive = "extractive"
    abstractive = "abstractive"
    hybrid = "hybrid"


class SummarizationResponse(BaseModel):
    """Text summarization response model."""
    text: str = Field(..., description="Original input text")
    summary: str = Field(..., description="Generated summary")
    method: str = Field(..., description="Summarization method used")
    compression_ratio: float = Field(..., ge=0.0, le=1.0, description="Summary compression ratio")
    original_length: int = Field(..., ge=0, description="Original text length in characters")
    summary_length: int = Field(..., ge=0, description="Summary length in characters")
    processing_time: float = Field(..., ge=0.0, description="Processing time in seconds")


class ComprehensiveResponse(BaseModel):
    """Comprehensive analysis response model."""
    text: str = Field(..., description="Original input text")
    sentiment: Optional[Dict[str, Any]] = Field(None, description="Sentiment analysis results")
    emotion: Optional[Dict[str, Any]] = Field(None, description="Emotion detection results")
    summary: Optional[Dict[str, Any]] = Field(None, description="Summarization results")
    timestamp: str = Field(..., description="Analysis timestamp")
    processing_time: float = Field(..., ge=0.0, description="Total processing time in seconds")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    uptime_requests: int = Field(..., ge=0, description="Total requests processed since startup")


class ProcessingStats(BaseModel):
    """Processing statistics response model."""
    total_requests: int = Field(..., ge=0, description="Total number of requests processed")
    sentiment_requests: int = Field(..., ge=0, description="Number of sentiment analysis requests")
    emotion_requests: int = Field(..., ge=0, description="Number of emotion detection requests")
    summarization_requests: int = Field(..., ge=0, description="Number of summarization requests")
    avg_processing_time: float = Field(..., ge=0.0, description="Average processing time in seconds")
    last_request_time: Optional[str] = Field(None, description="Timestamp of last request")


class TaskType(str, Enum):
    """Available task types for batch processing."""
    sentiment = "sentiment"
    emotion = "emotion"
    summary = "summary"


class OutputFormat(str, Enum):
    """Output formats for batch processing."""
    json = "json"
    csv = "csv"


class BatchProcessingRequest(BaseModel):
    """Batch processing request model."""
    texts: List[str] = Field(..., min_items=1, max_items=1000, description="List of texts to process")
    task_types: List[TaskType] = Field(..., min_items=1, description="Types of analysis to perform")
    output_format: OutputFormat = Field(OutputFormat.json, description="Output format")
    preprocess: bool = Field(True, description="Whether to preprocess texts")
    
    @validator('texts')
    def texts_not_empty(cls, v):
        for text in v:
            if not text.strip():
                raise ValueError('Text cannot be empty')
        return v


class BatchProcessingResponse(BaseModel):
    """Batch processing response model."""
    status: str = Field(..., description="Processing status")
    processed_count: int = Field(..., ge=0, description="Number of texts processed")
    results: List[Dict[str, Any]] = Field(..., description="Processing results")
    timestamp: datetime = Field(..., description="Processing completion timestamp")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")


class ModelInfo(BaseModel):
    """Model information."""
    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model type (sentiment, emotion, summarization)")
    version: Optional[str] = Field(None, description="Model version")
    language_support: List[str] = Field(..., description="Supported languages")
    last_updated: Optional[datetime] = Field(None, description="Last update timestamp")


class SystemInfo(BaseModel):
    """System information response."""
    api_version: str = Field(..., description="API version")
    models: List[ModelInfo] = Field(..., description="Available models")
    supported_languages: List[str] = Field(..., description="Supported languages")
    max_text_length: int = Field(..., description="Maximum text length for processing")
    max_batch_size: int = Field(..., description="Maximum batch size")
    features: List[str] = Field(..., description="Available features")


# Request models for specific endpoints

class SentimentAnalysisRequest(BaseModel):
    """Sentiment analysis specific request."""
    text: str = Field(..., min_length=1, max_length=5000)
    return_probabilities: bool = Field(False, description="Return probabilities for all classes")
    preprocess: bool = Field(True, description="Apply text preprocessing")


class EmotionDetectionRequest(BaseModel):
    """Emotion detection specific request."""
    text: str = Field(..., min_length=1, max_length=5000)
    top_k: int = Field(3, ge=1, le=10, description="Number of top emotions to return")
    threshold: float = Field(0.1, ge=0.0, le=1.0, description="Minimum confidence threshold")
    preprocess: bool = Field(True, description="Apply text preprocessing")


class SummarizationRequest(BaseModel):
    """Summarization specific request."""
    text: str = Field(..., min_length=50, max_length=5000)
    method: SummarizationMethod = Field(SummarizationMethod.extractive, description="Summarization method")
    num_sentences: int = Field(3, ge=1, le=10, description="Number of sentences for extractive summarization")
    max_length: int = Field(128, ge=30, le=500, description="Maximum length for abstractive summarization")
    min_length: int = Field(30, ge=10, le=200, description="Minimum length for abstractive summarization")
    preprocess: bool = Field(True, description="Apply text preprocessing")


class ComprehensiveAnalysisRequest(BaseModel):
    """Comprehensive analysis specific request."""
    text: str = Field(..., min_length=1, max_length=5000)
    include_sentiment: bool = Field(True, description="Include sentiment analysis")
    include_emotion: bool = Field(True, description="Include emotion detection")
    include_summary: bool = Field(True, description="Include text summarization")
    summary_method: SummarizationMethod = Field(SummarizationMethod.extractive, description="Summarization method")
    preprocess: bool = Field(True, description="Apply text preprocessing")


# Configuration models

class ModelConfig(BaseModel):
    """Model configuration."""
    name: str
    cache_dir: str
    max_length: Optional[int] = None
    batch_size: Optional[int] = None
    device: Optional[str] = None


class APIConfig(BaseModel):
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    log_level: str = "info"
    cors_origins: List[str] = ["*"]
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    rate_limit: Optional[int] = None


class PreprocessingConfig(BaseModel):
    """Text preprocessing configuration."""
    clean_html: bool = True
    normalize_unicode: bool = True
    remove_urls: bool = True
    remove_mentions: bool = True
    remove_special_chars: bool = True
    remove_stopwords: bool = False
    min_length: int = 10
    max_length: int = 1000