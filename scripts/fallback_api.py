"""
Fallback API server - Heuristic-based analysis without heavy dependencies
This version runs without transformers/torch dependencies for reliable startup
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import logging
from typing import Dict, Any, Optional, List
import time
from collections import Counter
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Indian E-Consultation Analysis API (Fallback Mode)",
    description="Heuristic-based sentiment analysis, emotion detection, and text summarization",
    version="2.0.0-fallback"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    method: str
    scores: Optional[Dict[str, float]] = None
    language_detected: Optional[str] = None
    reasoning: Optional[str] = None
    processing_time: Optional[float] = None

class EmotionResponse(BaseModel):
    text: str
    primary_emotion: str
    emotion_scores: Dict[str, float]
    confidence: float
    intensity: Optional[str] = None
    urgency_level: Optional[str] = None
    medical_context: Optional[bool] = None
    method: str
    reasoning: Optional[str] = None
    processing_time: Optional[float] = None

class SummaryRequest(BaseModel):
    text: str
    method: Optional[str] = "extractive"
    max_length: Optional[int] = 120

class SummaryResponse(BaseModel):
    original_text: str
    summary: str
    summary_length: int
    original_length: int
    compression_ratio: float
    method: str
    key_sentences: Optional[List[str]] = None
    important_entities: Optional[List[str]] = None
    quality_score: Optional[float] = None
    reasoning: Optional[str] = None
    processing_time: Optional[float] = None

class BatchRequest(BaseModel):
    texts: List[str]
    analyses: List[str] = Field(default_factory=lambda: ["sentiment", "emotion", "summarize"])

class OverallSummaryRequest(BaseModel):
    texts: List[str]
    max_length: Optional[int] = 200

def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Enhanced heuristic sentiment analysis"""
    start_time = time.time()
    text_lower = text.lower()
    
    # Enhanced word lists with medical context
    positive_words = [
        'अच्छी', 'बहुत', 'खुश', 'संतुष्ट', 'बेहतर', 'उत्कृष्ट', 'प्रभावी', 
        'ठीक', 'स्वस्थ', 'सुधार', 'लाभ', 'राहत', 'कामयाब',
        'good', 'excellent', 'great', 'happy', 'satisfied', 'wonderful', 
        'amazing', 'effective', 'helpful', 'better', 'improved', 'relief'
    ]
    
    negative_words = [
        'खराब', 'बुरा', 'गलत', 'निराश', 'असंतुष्ट', 'समस्या', 'दर्द', 'बीमार',
        'bad', 'terrible', 'awful', 'disappointed', 'poor', 'problem', 
        'issue', 'pain', 'worse', 'failed', 'difficult'
    ]
    
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    neutral_count = max(len(text_lower.split()) - (positive_count + negative_count), 0)
    total_count = max(positive_count + negative_count + neutral_count, 1)
    
    scores = {
        'positive': round(positive_count / total_count, 3),
        'negative': round(negative_count / total_count, 3),
        'neutral': round(neutral_count / total_count, 3)
    }
    
    if positive_count > negative_count:
        sentiment = "positive"
        confidence = min(0.7 + (positive_count * 0.1), 0.95)
    elif negative_count > positive_count:
        sentiment = "negative"
        confidence = min(0.7 + (negative_count * 0.1), 0.95)
    else:
        sentiment = "neutral"
        confidence = 0.6
        
    processing_time = time.time() - start_time
    
    return {
        'sentiment': sentiment,
        'confidence': confidence,
        'scores': scores,
        'method': 'enhanced_heuristic',
        'language_detected': 'mixed' if any(ord(c) > 127 for c in text) else 'english',
        'reasoning': f'Determined using enhanced keyword analysis (P:{positive_count}, N:{negative_count})',
        'processing_time': processing_time
    }

def analyze_emotion(text: str) -> Dict[str, Any]:
    """Enhanced heuristic emotion analysis"""
    start_time = time.time()
    text_lower = text.lower()
    
    emotion_keywords = {
        'joy': ['खुश', 'प्रसन्न', 'अच्छा', 'happy', 'joy', 'excited', 'great', 'wonderful'],
        'sadness': ['दुखी', 'उदास', 'परेशान', 'sad', 'disappointed', 'unhappy', 'depressed'],
        'anger': ['गुस्सा', 'क्रोध', 'नाराज', 'angry', 'mad', 'furious', 'annoyed', 'frustrated'],
        'fear': ['डर', 'चिंता', 'घबराहट', 'scared', 'worried', 'anxious', 'nervous'],
        'surprise': ['आश्चर्य', 'हैरान', 'अचंभा', 'surprised', 'shocked', 'amazed', 'unexpected'],
        'neutral': ['सामान्य', 'ठीक', 'normal', 'okay', 'fine']
    }
    
    emotion_scores = {}
    for emotion, keywords in emotion_keywords.items():
        score = sum(1 for word in keywords if word in text_lower)
        emotion_scores[emotion] = float(score)
    
    primary_emotion = max(emotion_scores, key=emotion_scores.get)
    if emotion_scores[primary_emotion] == 0:
        primary_emotion = 'neutral'
    
    dominant_score = emotion_scores.get(primary_emotion, 0)
    total_score = max(sum(emotion_scores.values()), 1)
    confidence = round(0.55 + (dominant_score / total_score) * 0.4, 3)
    
    intensity = 'high' if dominant_score >= 3 else 'medium' if dominant_score >= 2 else 'low'
    urgency_level = 'high' if primary_emotion in {'anger', 'fear'} and dominant_score >= 2 else 'moderate' if dominant_score >= 1 else 'low'
    
    processing_time = time.time() - start_time
    
    return {
        'primary_emotion': primary_emotion,
        'emotion_scores': emotion_scores,
        'confidence': confidence,
        'intensity': intensity,
        'urgency_level': urgency_level,
        'medical_context': any(word in text_lower for word in ['doctor', 'hospital', 'medicine', 'डॉक्टर', 'अस्पताल']),
        'method': 'enhanced_keyword_heuristic',
        'reasoning': f'Detected via keyword analysis, dominant: {primary_emotion} (score: {dominant_score})',
        'processing_time': processing_time
    }

def summarize_text(text: str, max_sentences: int = 2) -> Dict[str, Any]:
    """Enhanced extractive summarization"""
    start_time = time.time()
    
    sentences = [s.strip() for s in text.replace('।', '.').split('.') if s.strip()]
    original_length = len(text)
    
    if len(sentences) <= max_sentences:
        summary = text
        key_sentences = sentences
    else:
        # Simple ranking by sentence length and position
        ranked_sentences = []
        for i, sentence in enumerate(sentences):
            score = len(sentence.split()) + (1 / (i + 1))  # Length + position weight
            ranked_sentences.append((score, sentence))
        
        ranked_sentences.sort(reverse=True)
        key_sentences = [s[1] for s in ranked_sentences[:max_sentences]]
        summary = '. '.join(key_sentences)
        if not summary.endswith('.'):
            summary += '.'
    
    summary_length = len(summary)
    compression_ratio = summary_length / max(original_length, 1)
    quality_score = round(min(0.5 + (summary_length / (original_length + 1)) * 0.4, 0.85), 3)
    
    # Extract simple entities
    words = re.findall(r'\b[A-Za-z\u0900-\u097F]{3,}\b', text)
    important_entities = list(set([w for w in words if len(w) > 4]))[:5]
    
    processing_time = time.time() - start_time
    
    return {
        'summary': summary,
        'original_length': original_length,
        'summary_length': summary_length,
        'compression_ratio': compression_ratio,
        'method': 'enhanced_extractive',
        'key_sentences': key_sentences,
        'important_entities': important_entities,
        'quality_score': quality_score,
        'reasoning': f'Extracted {len(key_sentences)} key sentences using position and length ranking',
        'processing_time': processing_time
    }

@app.get("/")
async def root():
    return {
        "message": "Indian E-Consultation Analysis API (Fallback Mode) is running", 
        "status": "healthy",
        "version": "2.0.0-fallback",
        "features": ["sentiment_analysis", "emotion_detection", "text_summarization"],
        "mode": "heuristic_fallback"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "Fallback API is running",
        "version": "2.0.0-fallback",
        "mode": "heuristic"
    }

@app.post("/analyze/sentiment", response_model=SentimentResponse)
async def analyze_sentiment_endpoint(request: TextRequest):
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
            
        result = analyze_sentiment(request.text)
        
        return SentimentResponse(
            text=request.text,
            sentiment=result['sentiment'],
            confidence=result['confidence'],
            method=result['method'],
            scores=result['scores'],
            language_detected=result['language_detected'],
            reasoning=result['reasoning'],
            processing_time=result['processing_time']
        )
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/emotion", response_model=EmotionResponse)
async def analyze_emotion_endpoint(request: TextRequest):
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
            
        result = analyze_emotion(request.text)
        
        return EmotionResponse(
            text=request.text,
            primary_emotion=result['primary_emotion'],
            emotion_scores=result['emotion_scores'],
            confidence=result['confidence'],
            intensity=result['intensity'],
            urgency_level=result['urgency_level'],
            medical_context=result['medical_context'],
            method=result['method'],
            reasoning=result['reasoning'],
            processing_time=result['processing_time']
        )
    except Exception as e:
        logger.error(f"Error in emotion analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/summarize", response_model=SummaryResponse)
async def summarize_text_endpoint(request: SummaryRequest):
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
            
        max_sentences = max(1, min((request.max_length or 120) // 60, 5))
        result = summarize_text(request.text, max_sentences)
        
        return SummaryResponse(
            original_text=request.text,
            summary=result['summary'],
            summary_length=result['summary_length'],
            original_length=result['original_length'],
            compression_ratio=result['compression_ratio'],
            method=result['method'],
            key_sentences=result['key_sentences'],
            important_entities=result['important_entities'],
            quality_score=result['quality_score'],
            reasoning=result['reasoning'],
            processing_time=result['processing_time']
        )
    except Exception as e:
        logger.error(f"Error in summarization: {e}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

@app.post("/analyze/batch")
async def analyze_batch(request: BatchRequest):
    """Batch analysis of multiple texts"""
    try:
        results = []
        
        for i, text in enumerate(request.texts):
            if not text.strip():
                continue
                
            result = {"id": i + 1, "text": text}
            
            if "sentiment" in request.analyses:
                result["sentiment"] = analyze_sentiment(text)
                
            if "emotion" in request.analyses:
                result["emotions"] = analyze_emotion(text)
                
            if "summarize" in request.analyses:
                result["summary"] = summarize_text(text, max_sentences=2)
                
            results.append(result)
            
        return {"results": results, "total_processed": len(results)}
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting Indian E-Consultation Analysis API (Fallback Mode)")
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info"
    )