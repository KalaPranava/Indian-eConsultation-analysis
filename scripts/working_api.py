"""
Advanced API server with NLP models - IndicBERT, NLTK, Transformers
Upgraded from heuristic analysis to actual machine learning models
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import logging
from typing import Dict, Any, Optional, List
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import advanced models
try:
    from models.advanced_sentiment import create_sentiment_analyzer, EnsembleSentimentAnalyzer
    from models.advanced_emotion import create_emotion_detector, AdvancedEmotionDetector
    from models.advanced_summarization import create_text_summarizer, AdvancedTextSummarizer
    ADVANCED_MODELS_AVAILABLE = True
    logger.info("✅ Advanced NLP models imported successfully")
except ImportError as e:
    logger.warning(f"⚠️  Advanced models not available, will use fallbacks: {e}")
    ADVANCED_MODELS_AVAILABLE = False

app = FastAPI(
    title="Advanced Indian E-Consultation Analysis API",
    description="AI-powered sentiment analysis, emotion detection, and text summarization using IndicBERT, NLTK, and Transformer models",
    version="2.0.0"
)

# Add CORS middleware to handle cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize advanced models
advanced_sentiment_analyzer = None
advanced_emotion_detector = None
advanced_text_summarizer = None

@app.on_event("startup")
async def initialize_models():
    """Initialize advanced NLP models on startup"""
    global advanced_sentiment_analyzer, advanced_emotion_detector, advanced_text_summarizer
    
    if ADVANCED_MODELS_AVAILABLE:
        try:
            logger.info("🚀 Initializing advanced NLP models...")
            
            # Initialize sentiment analyzer (ensemble)
            logger.info("Loading ensemble sentiment analyzer...")
            advanced_sentiment_analyzer = create_sentiment_analyzer(use_ensemble=True)
            
            # Initialize emotion detector
            logger.info("Loading advanced emotion detector...")
            advanced_emotion_detector = create_emotion_detector()
            
            # Initialize text summarizer
            logger.info("Loading advanced text summarizer...")
            advanced_text_summarizer = create_text_summarizer()
            
            if advanced_sentiment_analyzer and advanced_emotion_detector and advanced_text_summarizer:
                logger.info("✅ All advanced NLP models initialized successfully!")
            else:
                logger.warning("⚠️  Some advanced models failed to initialize, fallbacks will be used")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize advanced models: {str(e)}")
            logger.info("📋 Will use heuristic fallbacks for analysis")
    else:
        logger.info("📋 Advanced models not available, using heuristic analysis")

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
    method: Optional[str] = "auto"
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

def analyze_sentiment_fallback(text: str) -> Dict[str, Any]:
    """Fallback sentiment analysis using simple heuristics."""
    text_lower = text.lower()
    
    # Hindi positive words
    positive_words = ['अच्छी', 'बहुत', 'खुश', 'संतुष्ट', 'बेहतर', 'उत्कृष्ट', 'प्रभावी', 
                     'good', 'excellent', 'great', 'happy', 'satisfied', 'wonderful', 'amazing']
    
    # Hindi negative words  
    negative_words = ['खराब', 'बुरा', 'गलत', 'निराश', 'असंतुष्ट', 'समस्या', 
                     'bad', 'terrible', 'awful', 'disappointed', 'poor', 'problem', 'issue']
    
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    neutral_count = max(len(text_lower.split()) - (positive_count + negative_count), 0)
    total_count = positive_count + negative_count + neutral_count
    total_count = total_count if total_count > 0 else 1
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
        
    return {
        'sentiment': sentiment,
        'confidence': confidence,
        'scores': scores,
        'method': 'heuristic_rules',
        'language_detected': None,
        'reasoning': 'Determined using keyword counting heuristics'
    }

def analyze_emotion_fallback(text: str) -> Dict[str, Any]:
    """Fallback emotion analysis using simple heuristics."""
    text_lower = text.lower()
    
    emotion_words = {
        'joy': ['खुश', 'प्रसन्न', 'happy', 'joy', 'excited', 'great', 'अच्छी'],
        'sadness': ['दुखी', 'उदास', 'sad', 'disappointed', 'unhappy'],
        'anger': ['गुस्सा', 'क्रोध', 'angry', 'mad', 'furious', 'annoyed'],
        'fear': ['डर', 'चिंता', 'scared', 'worried', 'anxious'],
        'surprise': ['आश्चर्य', 'हैरान', 'surprised', 'shocked', 'amazed'],
        'neutral': []
    }
    
    emotion_scores = {}
    for emotion, words in emotion_words.items():
        score = sum(1 for word in words if word in text_lower)
        emotion_scores[emotion] = score
        
    primary_emotion = max(emotion_scores, key=emotion_scores.get)
    if emotion_scores[primary_emotion] == 0:
        primary_emotion = 'neutral'
    dominant_score = emotion_scores.get(primary_emotion, 0)
    total_score = sum(emotion_scores.values()) or 1
    confidence = round(0.55 + (dominant_score / total_score) * 0.4, 3)
    intensity = 'high' if dominant_score >= 3 else 'medium' if dominant_score == 2 else 'low'
    urgency_level = 'high' if primary_emotion in {'anger', 'fear'} and dominant_score >= 2 else 'moderate' if dominant_score >= 1 else 'low'
        
    return {
        'primary_emotion': primary_emotion,
        'emotion_scores': {k: float(v) for k, v in emotion_scores.items()},
        'all_emotions': {k: float(v) for k, v in emotion_scores.items()},
        'confidence': confidence,
        'intensity': intensity,
        'urgency_level': urgency_level,
        'medical_context': False,
        'method': 'keyword_heuristic',
        'reasoning': 'Determined using keyword emotion heuristics'
    }

def summarize_text_fallback(text: str, max_sentences: int = 2) -> Dict[str, Any]:
    """Fallback summarization using simple sentence extraction."""
    sentences = text.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= max_sentences:
        summary = text
        key_sentences = sentences
    else:
        # Take first and most important sentences
        key_sentences = sentences[:max_sentences]
        summary = '. '.join(key_sentences)
        if not summary.endswith('.'):
            summary += '.'
    original_length = len(text)
    summary_length = len(summary)
    compression_ratio = summary_length / original_length if original_length > 0 else 0
    quality_score = round(min(0.5 + (summary_length / (original_length + 1)) * 0.4, 0.75), 3)
        
    return {
        'summary': summary,
        'original_length': original_length,
        'summary_length': summary_length,
        'compression_ratio': compression_ratio,
        'method': 'extractive_heuristic',
        'key_sentences': key_sentences,
        'important_entities': [],
        'quality_score': quality_score,
        'reasoning': 'Generated using lead sentence extraction'
    }

@app.get("/")
async def root():
    return {
        "message": "Indian E-Consultation Analysis API is running", 
        "status": "healthy",
        "version": "2.0.0",
        "features": ["sentiment_analysis", "emotion_detection", "text_summarization"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "API is running",
        "version": "2.0.0"
    }

@app.post("/analyze/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: TextRequest):
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
            
        # Use advanced analysis instead of fallback
        result = analyze_sentiment_advanced(request.text)
        
        return SentimentResponse(
            text=request.text,
            sentiment=result['sentiment'],
            confidence=result['confidence'],
            method=result.get('method', 'advanced_nlp'),
            scores=result.get('scores'),
            language_detected=result.get('language_detected'),
            reasoning=result.get('reasoning'),
            processing_time=result.get('processing_time')
        )
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/emotion", response_model=EmotionResponse)
async def analyze_emotion(request: TextRequest):
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
            
        # Use advanced analysis instead of fallback
        result = analyze_emotion_advanced(request.text)
        
        return EmotionResponse(
            text=request.text,
            primary_emotion=result['primary_emotion'],
            emotion_scores=result.get('emotion_scores', {}),
            confidence=result.get('confidence', 0.0),
            intensity=result.get('intensity'),
            urgency_level=result.get('urgency_level'),
            medical_context=result.get('medical_context'),
            method=result.get('method', 'advanced_nlp'),
            reasoning=result.get('reasoning'),
            processing_time=result.get('processing_time')
        )
    except Exception as e:
        logger.error(f"Error in emotion analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/summarize", response_model=SummaryResponse)
async def summarize_text(request: SummaryRequest):
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
            
        result = summarize_text_advanced(
            request.text,
            method=request.method or "auto",
            max_length=request.max_length or 120
        )
        
        return SummaryResponse(
            original_text=request.text,
            summary=result['summary'],
            summary_length=result.get('summary_length', len(result['summary'])),
            original_length=result.get('original_length', len(request.text)),
            compression_ratio=result['compression_ratio'],
            method=result.get('method', request.method or "auto"),
            key_sentences=result.get('key_sentences'),
            important_entities=result.get('important_entities'),
            quality_score=result.get('quality_score'),
            reasoning=result.get('reasoning'),
            processing_time=result.get('processing_time')
        )
    except Exception as e:
        logger.error(f"Error in summarization: {e}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

class BatchRequest(BaseModel):
    texts: List[str]
    analyses: List[str] = Field(default_factory=lambda: ["sentiment", "emotion", "summarize"])

@app.post("/analyze/batch")
async def analyze_batch(request: BatchRequest):
    """Batch analysis of multiple texts"""
    try:
        results = []
        
        for i, text in enumerate(request.texts):
            if not text.strip():
                continue
                
            result = {"id": i + 1, "text": text}
            
            # Sentiment analysis - use advanced model
            if "sentiment" in request.analyses:
                sentiment = analyze_sentiment_advanced(text)
                result["sentiment"] = sentiment
                
            # Emotion analysis - use advanced model
            if "emotion" in request.analyses:
                emotion = analyze_emotion_advanced(text)
                result["emotions"] = emotion
                
            # Summarization - use advanced model
            if "summarize" in request.analyses:
                summary = summarize_text_advanced(text, method="auto", max_length=80)
                result["summary"] = summary
                
            results.append(result)
            
        return {"results": results, "total_processed": len(results)}
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

class OverallSummaryRequest(BaseModel):
    texts: List[str]
    max_length: Optional[int] = 200

@app.post("/analyze/overall_summary")
async def generate_overall_summary(request: OverallSummaryRequest):
    """Generate overall summary of all comments"""
    try:
        if not request.texts:
            raise HTTPException(status_code=400, detail="At least one text is required")

        from collections import Counter

        sentiment_counts: Counter[str] = Counter()
        emotion_counts: Counter[str] = Counter()
        urgency_counts: Counter[str] = Counter()
        important_entities: Counter[str] = Counter()
        key_sentence_counter: Counter[str] = Counter()

        sentiment_confidences: List[float] = []
        summary_quality_scores: List[float] = []

        valid_texts: List[str] = []

        for text in request.texts:
            if not text or not text.strip():
                continue
            valid_texts.append(text)

            sentiment = analyze_sentiment_advanced(text)
            sentiment_label = sentiment.get("sentiment", "neutral")
            sentiment_counts[sentiment_label] += 1
            if sentiment.get("confidence") is not None:
                sentiment_confidences.append(sentiment["confidence"])

            emotion = analyze_emotion_advanced(text)
            primary_emotion = emotion.get("primary_emotion", "neutral")
            emotion_counts[primary_emotion] += 1
            urgency_level = emotion.get("urgency_level")
            if urgency_level:
                urgency_counts[urgency_level] += 1

            summary = summarize_text_advanced(
                text,
                method="auto",
                max_length=min(request.max_length or 200, 200)
            )
            if summary.get("quality_score") is not None:
                summary_quality_scores.append(summary["quality_score"])
            for sentence in summary.get("key_sentences") or []:
                if sentence:
                    key_sentence_counter[sentence] += 1
            for entity in summary.get("important_entities") or []:
                if entity:
                    important_entities[entity] += 1

        total_comments = len(valid_texts)
        if total_comments == 0:
            raise HTTPException(status_code=400, detail="No valid texts provided")

        pos_pct = round((sentiment_counts.get("positive", 0) / total_comments) * 100, 2)
        neg_pct = round((sentiment_counts.get("negative", 0) / total_comments) * 100, 2)
        neu_pct = round((sentiment_counts.get("neutral", 0) / total_comments) * 100, 2)

        dominant_emotion, dominant_emotion_count = ("neutral", 0)
        if emotion_counts:
            dominant_emotion, dominant_emotion_count = emotion_counts.most_common(1)[0]

        overall_summary = summarize_text_advanced(
            " ".join(valid_texts),
            method="hybrid",
            max_length=request.max_length or 220
        )

        avg_sentiment_confidence = round(sum(sentiment_confidences) / len(sentiment_confidences), 3) if sentiment_confidences else 0.0
        avg_summary_quality = round(sum(summary_quality_scores) / len(summary_quality_scores), 3) if summary_quality_scores else 0.0

        key_themes = [sentence for sentence, _ in key_sentence_counter.most_common(5)]
        top_entities = [entity for entity, _ in important_entities.most_common(5)]

        for label in ("positive", "neutral", "negative"):
            if label not in sentiment_counts:
                sentiment_counts[label] = 0
        for label in ("joy", "sadness", "anger", "fear", "surprise", "neutral"):
            if label not in emotion_counts:
                emotion_counts[label] = 0
        for label in ("high", "moderate", "low"):
            if label not in urgency_counts:
                urgency_counts[label] = 0

        return {
            "overall_summary": overall_summary["summary"],
            "overall_summary_details": overall_summary,
            "sentiment_distribution": dict(sentiment_counts),
            "emotion_distribution": dict(emotion_counts),
            "urgency_distribution": dict(urgency_counts),
            "total_comments": total_comments,
            "model_confidence": avg_sentiment_confidence,
            "average_summary_quality": avg_summary_quality,
            "key_themes": key_themes,
            "key_entities": top_entities,
            "meta": {
                "positive_pct": pos_pct,
                "negative_pct": neg_pct,
                "neutral_pct": neu_pct,
                "dominant_emotion": dominant_emotion,
                "dominant_emotion_support": dominant_emotion_count
            },
            "key_insights": {
                "satisfaction_level": "high" if pos_pct > 70 else "moderate" if pos_pct > 50 else "low",
                "priority_attention_required": urgency_counts.get("high", 0),
                "top_concerns": key_themes[:3],
                "emotional_state": dominant_emotion
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating overall summary: {e}")
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")

@app.post("/analyze/wordcloud")
async def generate_wordcloud_data(request: BatchRequest):
    """Generate word frequency data for word cloud"""
    try:
        from collections import Counter
        import re
        
        # Combine all texts
        all_text = " ".join(request.texts).lower()
        
        # Remove Hindi and English stopwords
        hindi_stopwords = {'है', 'हैं', 'का', 'की', 'के', 'में', 'से', 'को', 'और', 'यह', 'वह', 'एक', 'तो', 'ही', 'भी'}
        english_stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'a', 'an'}
        
        # Extract words (handle both Hindi and English)
        words = re.findall(r'[\w\u0900-\u097F]+', all_text)
        
        # Filter words
        filtered_words = []
        for word in words:
            if (len(word) > 2 and 
                word not in hindi_stopwords and 
                word not in english_stopwords and
                not word.isdigit()):
                filtered_words.append(word)
        
        # Count frequencies
        word_counts = Counter(filtered_words)
        
        # Create word cloud data with comments mapping
        wordcloud_data = []
        word_to_comments = {}
        
        # Map words to comments containing them
        for i, text in enumerate(request.texts):
            text_lower = text.lower()
            for word in word_counts.keys():
                if word in text_lower:
                    if word not in word_to_comments:
                        word_to_comments[word] = []
                    word_to_comments[word].append({
                        "id": i + 1,
                        "text": text[:200] + "..." if len(text) > 200 else text
                    })
        
        # Prepare word cloud data
        for word, count in word_counts.most_common(50):  # Top 50 words
            wordcloud_data.append({
                "text": word,
                "size": count,
                "comments": word_to_comments.get(word, [])
            })
        
        return {
            "wordcloud_data": wordcloud_data,
            "total_words": len(filtered_words),
            "unique_words": len(word_counts)
        }
        
    except Exception as e:
        logger.error(f"Error generating word cloud data: {e}")
        raise HTTPException(status_code=500, detail=f"Word cloud generation failed: {str(e)}")

def analyze_sentiment_advanced(text: str) -> Dict[str, Any]:
    """Advanced sentiment analysis using NLP models"""
    if advanced_sentiment_analyzer:
        try:
            logger.info(f"🧠 Using advanced sentiment analysis for: {text[:50]}...")
            result = advanced_sentiment_analyzer.analyze(text)
            return {
                "sentiment": result.label,
                "confidence": result.confidence,
                "scores": result.scores,
                "method": result.method,
                "language_detected": result.language_detected,
                "reasoning": result.reasoning,
                "processing_time": result.processing_time
            }
        except Exception as e:
            logger.warning(f"⚠️  Advanced sentiment analysis failed: {e}, using fallback")
            return analyze_sentiment_fallback(text)
    else:
        logger.info("📋 Using fallback sentiment analysis")
        return analyze_sentiment_fallback(text)

def analyze_emotion_advanced(text: str) -> Dict[str, Any]:
    """Advanced emotion analysis using NLP models"""
    if advanced_emotion_detector:
        try:
            logger.info(f"🎭 Using advanced emotion analysis for: {text[:50]}...")
            result = advanced_emotion_detector.analyze(text, context="medical")
            return {
                "primary_emotion": result.primary_emotion,
                "confidence": result.confidence,
                "all_emotions": result.all_emotions,
                "emotion_scores": result.all_emotions,
                "intensity": result.intensity,
                "method": result.method,
                "urgency_level": result.urgency_level,
                "medical_context": result.medical_context_detected,
                "reasoning": result.reasoning,
                "processing_time": result.processing_time
            }
        except Exception as e:
            logger.warning(f"⚠️  Advanced emotion analysis failed: {e}, using fallback")
            return analyze_emotion_fallback(text)
    else:
        logger.info("📋 Using fallback emotion analysis")
        return analyze_emotion_fallback(text)

def summarize_text_advanced(text: str, method: str = "hybrid", max_length: int = 100) -> Dict[str, Any]:
    """Advanced text summarization using NLP models"""
    fallback_sentences = max(1, min((max_length // 60) if max_length else 2, 5))
    if advanced_text_summarizer:
        try:
            logger.info(f"📝 Using advanced summarization for: {text[:50]}...")
            result = advanced_text_summarizer.summarize(
                text=text, 
                method=method, 
                max_length=max_length,
                target_sentences=2
            )
            return {
                "summary": result.summary,
                "original_length": result.original_length,
                "summary_length": result.summary_length,
                "compression_ratio": result.compression_ratio,
                "method": result.method_used,
                "key_sentences": result.key_sentences,
                "important_entities": result.important_entities,
                "quality_score": result.summary_quality_score,
                "reasoning": result.reasoning,
                "processing_time": result.processing_time
            }
        except Exception as e:
            logger.warning(f"⚠️  Advanced summarization failed: {e}, using fallback")
            return summarize_text_fallback(text, max_sentences=fallback_sentences)
    else:
        logger.info("📋 Using fallback summarization")
        return summarize_text_fallback(text, max_sentences=fallback_sentences)

if __name__ == "__main__":
    logger.info("Starting Indian E-Consultation Analysis API (Simplified)")
    uvicorn.run(
        "working_api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info"
    )