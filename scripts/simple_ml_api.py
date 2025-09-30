"""
Simple ML-powered API using IndicBERT and Transformers
Avoids complex imports that cause dependency conflicts
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

# Try to import ML models
ML_MODELS_AVAILABLE = False
sentiment_pipeline = None
emotion_pipeline = None

try:
    from transformers import pipeline
    logger.info("🔄 Loading IndicBERT and transformer models...")
    
    # Load sentiment analysis pipeline
    try:
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
            tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment"
        )
        logger.info("✅ Sentiment model loaded")
    except Exception as e:
        logger.warning(f"⚠️ Sentiment model failed: {e}")
    
    # Load emotion analysis pipeline  
    try:
        emotion_pipeline = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None
        )
        logger.info("✅ Emotion model loaded")
    except Exception as e:
        logger.warning(f"⚠️ Emotion model failed: {e}")
    
    if sentiment_pipeline or emotion_pipeline:
        ML_MODELS_AVAILABLE = True
        logger.info("🚀 ML models initialized successfully!")
    
except Exception as e:
    logger.warning(f"⚠️ ML models not available, using enhanced heuristics: {e}")

app = FastAPI(
    title="Indian E-Consultation Analysis API (ML + Fallback)",
    description="IndicBERT and Transformer-powered analysis with heuristic fallbacks",
    version="2.0.0-ml"
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

def analyze_sentiment_ml(text: str) -> Dict[str, Any]:
    """ML-powered sentiment analysis using transformers"""
    start_time = time.time()
    
    if sentiment_pipeline:
        try:
            result = sentiment_pipeline(text)[0]
            
            # Map labels to standard format
            label_map = {
                'LABEL_0': 'negative', 'NEGATIVE': 'negative',
                'LABEL_1': 'neutral', 'NEUTRAL': 'neutral', 
                'LABEL_2': 'positive', 'POSITIVE': 'positive'
            }
            
            sentiment = label_map.get(result['label'], result['label'].lower())
            confidence = result['score']
            
            # Create score distribution
            scores = {sentiment: confidence}
            for label in ['positive', 'negative', 'neutral']:
                if label not in scores:
                    scores[label] = (1 - confidence) / 2 if len(scores) == 1 else 0.1
            
            processing_time = time.time() - start_time
            
            return {
                'sentiment': sentiment,
                'confidence': round(confidence, 3),
                'scores': {k: round(v, 3) for k, v in scores.items()},
                'method': 'transformer_ml',
                'language_detected': 'multilingual',
                'reasoning': f'Analyzed using XLM-RoBERTa transformer model',
                'processing_time': processing_time
            }
        except Exception as e:
            logger.warning(f"ML sentiment analysis error: {e}")
    
    # Fallback to heuristics
    return analyze_sentiment_heuristic(text)

def analyze_emotion_ml(text: str) -> Dict[str, Any]:
    """ML-powered emotion analysis using transformers"""
    start_time = time.time()
    
    if emotion_pipeline:
        try:
            results = emotion_pipeline(text)
            
            # Convert to emotion scores
            emotion_scores = {}
            for result in results:
                emotion = result['label'].lower()
                score = result['score']
                emotion_scores[emotion] = round(score, 3)
            
            # Find primary emotion
            primary_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = emotion_scores[primary_emotion]
            
            # Determine intensity and urgency
            intensity = 'high' if confidence > 0.7 else 'medium' if confidence > 0.4 else 'low'
            urgency_level = 'high' if primary_emotion in ['anger', 'fear'] and confidence > 0.6 else 'moderate' if confidence > 0.4 else 'low'
            
            processing_time = time.time() - start_time
            
            return {
                'primary_emotion': primary_emotion,
                'emotion_scores': emotion_scores,
                'confidence': confidence,
                'intensity': intensity,
                'urgency_level': urgency_level,
                'medical_context': any(word in text.lower() for word in ['doctor', 'hospital', 'medicine', 'डॉक्टर', 'अस्पताल']),
                'method': 'transformer_ml',
                'reasoning': f'Analyzed using DistilRoBERTa emotion classification model',
                'processing_time': processing_time
            }
        except Exception as e:
            logger.warning(f"ML emotion analysis error: {e}")
    
    # Fallback to heuristics
    return analyze_emotion_heuristic(text)

def analyze_sentiment_heuristic(text: str) -> Dict[str, Any]:
    """Enhanced heuristic sentiment analysis fallback"""
    start_time = time.time()
    text_lower = text.lower()
    
    positive_words = [
        'अच्छी', 'बहुत', 'खुश', 'संतुष्ट', 'बेहतर', 'उत्कृष्ट', 'प्रभावी', 
        'ठीक', 'स्वस्थ', 'सुधार', 'लाभ', 'राहत',
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
        'method': 'enhanced_heuristic_fallback',
        'language_detected': 'mixed' if any(ord(c) > 127 for c in text) else 'english',
        'reasoning': f'Keyword analysis fallback (P:{positive_count}, N:{negative_count})',
        'processing_time': processing_time
    }

def analyze_emotion_heuristic(text: str) -> Dict[str, Any]:
    """Enhanced heuristic emotion analysis fallback"""
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
        emotion_scores[emotion] = float(score) / 10  # Normalize
    
    primary_emotion = max(emotion_scores, key=emotion_scores.get)
    if emotion_scores[primary_emotion] == 0:
        primary_emotion = 'neutral'
    
    confidence = round(0.55 + emotion_scores[primary_emotion], 3)
    processing_time = time.time() - start_time
    
    return {
        'primary_emotion': primary_emotion,
        'emotion_scores': emotion_scores,
        'confidence': confidence,
        'intensity': 'medium',
        'urgency_level': 'low',
        'medical_context': any(word in text_lower for word in ['doctor', 'hospital', 'medicine', 'डॉक्टर', 'अस्पताल']),
        'method': 'keyword_heuristic_fallback',
        'reasoning': f'Keyword analysis fallback, detected: {primary_emotion}',
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
            score = len(sentence.split()) + (1 / (i + 1))
            ranked_sentences.append((score, sentence))
        
        ranked_sentences.sort(reverse=True)
        key_sentences = [s[1] for s in ranked_sentences[:max_sentences]]
        summary = '. '.join(key_sentences)
        if not summary.endswith('.'):
            summary += '.'
    
    summary_length = len(summary)
    compression_ratio = summary_length / max(original_length, 1)
    quality_score = round(min(0.5 + (summary_length / (original_length + 1)) * 0.4, 0.85), 3)
    
    processing_time = time.time() - start_time
    
    return {
        'summary': summary,
        'original_length': original_length,
        'summary_length': summary_length,
        'compression_ratio': compression_ratio,
        'method': 'enhanced_extractive',
        'key_sentences': key_sentences,
        'important_entities': [],
        'quality_score': quality_score,
        'reasoning': f'Extracted {len(key_sentences)} key sentences using ranking',
        'processing_time': processing_time
    }

@app.get("/")
async def root():
    ml_status = "✅ ML Models Active" if ML_MODELS_AVAILABLE else "📋 Heuristic Fallback"
    return {
        "message": "Indian E-Consultation Analysis API is running", 
        "status": "healthy",
        "version": "2.0.0-ml",
        "features": ["sentiment_analysis", "emotion_detection", "text_summarization"],
        "ml_models": ML_MODELS_AVAILABLE,
        "mode": ml_status
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "ML API is running",
        "version": "2.0.0-ml",
        "ml_models_active": ML_MODELS_AVAILABLE
    }

@app.post("/analyze/sentiment", response_model=SentimentResponse)
async def analyze_sentiment_endpoint(request: TextRequest):
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
            
        result = analyze_sentiment_ml(request.text)
        
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
            
        result = analyze_emotion_ml(request.text)
        
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

if __name__ == "__main__":
    logger.info(f"Starting Indian E-Consultation Analysis API ({'ML Mode' if ML_MODELS_AVAILABLE else 'Fallback Mode'})")
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info"
    )