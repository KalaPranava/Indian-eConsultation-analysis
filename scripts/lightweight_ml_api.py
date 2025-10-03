"""
Lightweight ML API optimized for 512MB memory constraint
Uses efficient models: VADER, TextBlob, custom lexicons, and sklearn
Perfect for cost-effective deployment while maintaining good accuracy
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import logging
from typing import Dict, Any, Optional, List
import time
import numpy as np
from collections import Counter
import re
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize lightweight ML models
ML_MODELS = {}
MODEL_STATUS = {"loaded": False, "error": None}

def load_lightweight_models():
    """Load lightweight ML models optimized for memory efficiency"""
    global ML_MODELS, MODEL_STATUS
    
    try:
        logger.info("🔄 Loading lightweight ML models...")
        
        # Load NLTK VADER (lexicon-based, ~5MB)
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
            from nltk.sentiment import SentimentIntensityAnalyzer
            ML_MODELS['vader'] = SentimentIntensityAnalyzer()
            logger.info("✅ VADER sentiment loaded (~5MB)")
        except Exception as e:
            logger.warning(f"VADER failed: {e}")
        
        # Load TextBlob (simple ML, ~10MB)
        try:
            from textblob import TextBlob
            # Test TextBlob
            test_blob = TextBlob("This is good")
            test_sentiment = test_blob.sentiment
            ML_MODELS['textblob'] = TextBlob
            logger.info("✅ TextBlob loaded (~10MB)")
        except Exception as e:
            logger.warning(f"TextBlob failed: {e}")
        
        # Load scikit-learn for lightweight ML (~20MB)
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.naive_bayes import MultinomialNB
            ML_MODELS['tfidf'] = TfidfVectorizer(max_features=2000, stop_words='english')
            ML_MODELS['logistic'] = LogisticRegression()
            ML_MODELS['naive_bayes'] = MultinomialNB()
            logger.info("✅ Scikit-learn models loaded (~20MB)")
        except Exception as e:
            logger.warning(f"Scikit-learn failed: {e}")
        
        # Initialize custom multilingual lexicons (Hindi + English)
        ML_MODELS['hindi_sentiment'] = load_hindi_sentiment_lexicon()
        ML_MODELS['emotion_lexicon'] = load_emotion_lexicon()
        logger.info("✅ Custom lexicons loaded (~5MB)")
        
        MODEL_STATUS["loaded"] = len(ML_MODELS) > 0
        total_memory = estimate_memory_usage()
        logger.info(f"🚀 Lightweight models loaded: {list(ML_MODELS.keys())}")
        logger.info(f"📊 Estimated memory usage: ~{total_memory}MB (fits in 512MB)")
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {str(e)}")
        MODEL_STATUS["error"] = str(e)

def estimate_memory_usage():
    """Estimate total memory usage of loaded models"""
    memory_estimates = {
        'vader': 5,
        'textblob': 10,
        'tfidf': 15,
        'logistic': 5,
        'naive_bayes': 3,
        'hindi_sentiment': 3,
        'emotion_lexicon': 2
    }
    total = sum(memory_estimates.get(model, 0) for model in ML_MODELS.keys())
    return total

def load_hindi_sentiment_lexicon():
    """Load Hindi sentiment words for multilingual support"""
    return {
        'positive': {
            # Hindi positive words
            'अच्छा': 0.8, 'अच्छी': 0.8, 'अच्छे': 0.8, 'बहुत': 0.6, 'खुश': 0.9,
            'संतुष्ट': 0.8, 'बेहतर': 0.7, 'उत्कृष्ट': 0.9, 'प्रभावी': 0.7,
            'ठीक': 0.6, 'स्वस्थ': 0.8, 'सुधार': 0.7, 'लाभ': 0.7, 'राहत': 0.8,
            'कामयाब': 0.8, 'सफल': 0.8, 'धन्यवाद': 0.9, 'शुक्रिया': 0.9,
            # English positive words
            'good': 0.7, 'great': 0.8, 'excellent': 0.9, 'amazing': 0.9,
            'wonderful': 0.9, 'fantastic': 0.9, 'perfect': 0.9, 'love': 0.8,
            'helpful': 0.7, 'effective': 0.7, 'satisfied': 0.8, 'happy': 0.8,
            'relief': 0.7, 'better': 0.6, 'improved': 0.7, 'recommend': 0.8,
            'professional': 0.6, 'caring': 0.7, 'kind': 0.7, 'thank': 0.7
        },
        'negative': {
            # Hindi negative words
            'खराब': -0.8, 'बुरा': -0.8, 'गलत': -0.7, 'निराश': -0.8, 'असंतुष्ट': -0.8,
            'समस्या': -0.6, 'दर्द': -0.7, 'बीमार': -0.6, 'परेशान': -0.7,
            'चिंता': -0.6, 'डर': -0.7, 'गुस्सा': -0.8, 'नाराज': -0.7,
            # English negative words
            'bad': -0.7, 'terrible': -0.9, 'awful': -0.9, 'horrible': -0.9,
            'disappointed': -0.7, 'poor': -0.6, 'problem': -0.6, 'issue': -0.5,
            'pain': -0.6, 'worse': -0.7, 'failed': -0.8, 'difficult': -0.5,
            'rude': -0.8, 'unprofessional': -0.8, 'slow': -0.4, 'expensive': -0.4,
            'waste': -0.7, 'complaint': -0.6, 'angry': -0.8, 'frustrated': -0.7
        }
    }

def load_emotion_lexicon():
    """Load emotion detection lexicon"""
    return {
        'joy': ['happy', 'excited', 'cheerful', 'delighted', 'pleased', 'खुश', 'प्रसन्न'],
        'sadness': ['sad', 'depressed', 'unhappy', 'melancholy', 'grief', 'दुखी', 'उदास'],
        'anger': ['angry', 'furious', 'irritated', 'annoyed', 'mad', 'गुस्सा', 'नाराज'],
        'fear': ['afraid', 'scared', 'worried', 'anxious', 'nervous', 'डर', 'चिंता'],
        'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'हैरान'],
        'disgust': ['disgusted', 'revolted', 'repulsed', 'घृणा'],
        'anticipation': ['excited', 'eager', 'hopeful', 'expecting', 'उत्साह']
    }

# Load models on startup
load_lightweight_models()

app = FastAPI(
    title="Lightweight Indian E-Consultation Analysis API",
    description="Memory-optimized ML API using VADER, TextBlob, and custom lexicons (< 512MB)",
    version="3.0.0-lightweight"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001", 
        "https://insightgov-platform.vercel.app",
        "https://*.vercel.app"
    ],
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

class OverallSummaryRequest(BaseModel):
    texts: List[str]
    max_length: Optional[int] = 250

class OverallSummaryResponse(BaseModel):
    overall_summary: str
    sentiment_distribution: Dict[str, int]
    emotion_distribution: Dict[str, int]
    key_insights: Dict[str, Any]
    methods_used: List[str]
    processing_time: float


def analyze_sentiment_lightweight(text: str) -> Dict[str, Any]:
    """Lightweight sentiment analysis using multiple efficient methods"""
    start_time = time.time()
    
    # Method 1: VADER (lexicon-based, very fast)
    if 'vader' in ML_MODELS:
        try:
            vader = ML_MODELS['vader']
            vader_scores = vader.polarity_scores(text)
            
            compound = vader_scores['compound']
            if compound >= 0.05:
                sentiment = 'positive'
            elif compound <= -0.05:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            confidence = abs(compound) if abs(compound) > 0.1 else 0.5
            
            scores = {
                'positive': round(vader_scores['pos'], 3),
                'negative': round(vader_scores['neg'], 3),
                'neutral': round(vader_scores['neu'], 3)
            }
            
            processing_time = time.time() - start_time
            
            return {
                'sentiment': sentiment,
                'confidence': round(confidence, 3),
                'scores': scores,
                'method': 'vader_lightweight',
                'language_detected': detect_language(text),
                'reasoning': f'VADER lexicon analysis (compound: {compound:.3f})',
                'processing_time': processing_time
            }
        except Exception as e:
            logger.warning(f"VADER failed: {e}")
    
    # Method 2: TextBlob (simple ML)
    if 'textblob' in ML_MODELS:
        try:
            TextBlob = ML_MODELS['textblob']
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            confidence = abs(polarity) if abs(polarity) > 0.1 else 0.5
            
            # Convert to score distribution
            pos_score = max(0, polarity)
            neg_score = max(0, -polarity)
            neu_score = 1 - abs(polarity)
            
            total = pos_score + neg_score + neu_score
            scores = {
                'positive': round(pos_score / total, 3),
                'negative': round(neg_score / total, 3),
                'neutral': round(neu_score / total, 3)
            }
            
            processing_time = time.time() - start_time
            
            return {
                'sentiment': sentiment,
                'confidence': round(confidence, 3),
                'scores': scores,
                'method': 'textblob_lightweight',
                'language_detected': detect_language(text),
                'reasoning': f'TextBlob polarity analysis (score: {polarity:.3f})',
                'processing_time': processing_time
            }
        except Exception as e:
            logger.warning(f"TextBlob failed: {e}")
    
    # Method 3: Custom multilingual lexicon
    return analyze_sentiment_custom_lexicon(text, start_time)

def analyze_sentiment_custom_lexicon(text: str, start_time: float) -> Dict[str, Any]:
    """Custom lexicon-based sentiment analysis for Hindi/English"""
    text_lower = text.lower()
    
    if 'hindi_sentiment' not in ML_MODELS:
        return fallback_sentiment_analysis(text, start_time)
    
    lexicon = ML_MODELS['hindi_sentiment']
    
    positive_score = 0
    negative_score = 0
    word_count = 0
    
    # Calculate weighted sentiment scores
    words = text_lower.split()
    for word in words:
        if word in lexicon['positive']:
            positive_score += lexicon['positive'][word]
            word_count += 1
        elif word in lexicon['negative']:
            negative_score += abs(lexicon['negative'][word])
            word_count += 1
    
    # Normalize scores
    if word_count > 0:
        positive_score /= len(words)
        negative_score /= len(words)
    
    # Determine sentiment
    if positive_score > negative_score:
        sentiment = 'positive'
        confidence = min(positive_score * 2, 0.9)
    elif negative_score > positive_score:
        sentiment = 'negative'
        confidence = min(negative_score * 2, 0.9)
    else:
        sentiment = 'neutral'
        confidence = 0.5
    
    # Create score distribution
    total = positive_score + negative_score + 0.3  # neutral baseline
    scores = {
        'positive': round(positive_score / total, 3),
        'negative': round(negative_score / total, 3),
        'neutral': round(0.3 / total, 3)
    }
    
    processing_time = time.time() - start_time
    
    return {
        'sentiment': sentiment,
        'confidence': round(confidence, 3),
        'scores': scores,
        'method': 'custom_multilingual_lexicon',
        'language_detected': detect_language(text),
        'reasoning': f'Custom lexicon analysis (P:{positive_score:.2f}, N:{negative_score:.2f})',
        'processing_time': processing_time
    }

def analyze_emotion_lightweight(text: str) -> Dict[str, Any]:
    """Lightweight emotion analysis using lexicon-based approach"""
    start_time = time.time()
    
    if 'emotion_lexicon' not in ML_MODELS:
        return fallback_emotion_analysis(text, start_time)
    
    emotion_lexicon = ML_MODELS['emotion_lexicon']
    text_lower = text.lower()
    emotion_scores = {}
    
    # Calculate emotion scores based on word presence
    for emotion, words in emotion_lexicon.items():
        score = 0
        for word in words:
            if word in text_lower:
                # Boost score based on word frequency
                score += text_lower.count(word) * 0.2
        emotion_scores[emotion] = round(score, 3)
    
    # If no emotions detected, use sentiment-based emotion mapping
    if sum(emotion_scores.values()) == 0:
        sentiment_result = analyze_sentiment_lightweight(text)
        sentiment = sentiment_result['sentiment']
        confidence = sentiment_result['confidence']
        
        if sentiment == 'positive':
            emotion_scores = {'joy': confidence, 'anticipation': confidence * 0.7}
        elif sentiment == 'negative':
            emotion_scores = {'sadness': confidence, 'anger': confidence * 0.5}
        else:
            emotion_scores = {'neutral': 0.8}
    
    # Normalize scores
    total_score = sum(emotion_scores.values())
    if total_score > 0:
        emotion_scores = {k: round(v / total_score, 3) for k, v in emotion_scores.items()}
    
    primary_emotion = max(emotion_scores, key=emotion_scores.get) if emotion_scores else 'neutral'
    confidence = emotion_scores.get(primary_emotion, 0.5)
    
    # Determine intensity and urgency
    intensity = 'high' if confidence > 0.7 else 'medium' if confidence > 0.4 else 'low'
    urgency_level = 'high' if primary_emotion in ['anger', 'fear'] and confidence > 0.6 else 'moderate' if confidence > 0.4 else 'low'
    
    processing_time = time.time() - start_time
    
    return {
        'text': text,
        'primary_emotion': primary_emotion,
        'emotion_scores': emotion_scores,
        'confidence': round(confidence, 3),
        'intensity': intensity,
        'urgency_level': urgency_level,
        'medical_context': detect_medical_context(text),
        'method': 'lexicon_based_lightweight',
        'reasoning': f'Lexicon-based emotion detection ({primary_emotion}: {confidence:.3f})',
        'processing_time': processing_time
    }

def detect_language(text: str) -> str:
    """Lightweight language detection"""
    # Simple heuristic based on character sets
    hindi_chars = sum(1 for c in text if ord(c) >= 0x0900 and ord(c) <= 0x097F)
    total_chars = len([c for c in text if c.isalpha()])
    
    if total_chars == 0:
        return 'unknown'
    
    hindi_ratio = hindi_chars / total_chars
    
    if hindi_ratio > 0.3:
        return 'hindi' if hindi_ratio > 0.7 else 'mixed'
    else:
        return 'english'

def detect_medical_context(text: str) -> bool:
    """Detect if text is related to medical consultation"""
    medical_keywords = [
        'doctor', 'डॉक्टर', 'hospital', 'अस्पताल', 'medicine', 'दवा', 'treatment', 'इलाज',
        'patient', 'मरीज', 'consultation', 'जांच', 'checkup', 'appointment', 'prescription',
        'symptoms', 'लक्षण', 'disease', 'बीमारी', 'health', 'स्वास्थ्य', 'clinic'
    ]
    
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in medical_keywords)

def summarize_text_lightweight(text: str, max_length: int = 120) -> Dict[str, Any]:
    """Lightweight text summarization using extractive approach"""
    start_time = time.time()
    
    sentences = re.split(r'[.!?।]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    if len(sentences) <= 2:
        processing_time = time.time() - start_time
        return {
            'original_text': text,
            'summary': text,
            'summary_length': len(text),
            'original_length': len(text),
            'compression_ratio': 1.0,
            'method': 'no_summarization_needed',
            'key_sentences': sentences,
            'processing_time': processing_time
        }
    
    # Simple scoring based on word frequency and position
    word_freq = Counter()
    for sentence in sentences:
        words = sentence.lower().split()
        for word in words:
            if len(word) > 3 and word not in ['this', 'that', 'with', 'have', 'will']:
                word_freq[word] += 1
    
    # Score sentences
    sentence_scores = []
    for i, sentence in enumerate(sentences):
        score = 0
        words = sentence.lower().split()
        
        # Frequency score
        for word in words:
            score += word_freq.get(word, 0)
        
        # Position bonus (first and last sentences)
        if i == 0 or i == len(sentences) - 1:
            score += len(words) * 0.5
        
        # Length penalty for very short sentences
        if len(words) < 5:
            score *= 0.5
        
        sentence_scores.append((score, sentence))
    
    # Sort by score and select top sentences
    sentence_scores.sort(reverse=True)
    
    # Select sentences to fit max_length
    selected_sentences = []
    current_length = 0
    
    for score, sentence in sentence_scores:
        if current_length + len(sentence) <= max_length:
            selected_sentences.append(sentence)
            current_length += len(sentence)
        
        if len(selected_sentences) >= 3:  # Max 3 sentences
            break
    
    summary = ' '.join(selected_sentences)
    if not summary:
        summary = sentences[0] if sentences else text[:max_length]
    
    processing_time = time.time() - start_time
    
    return {
        'original_text': text,
        'summary': summary,
        'summary_length': len(summary),
        'original_length': len(text),
        'compression_ratio': round(len(summary) / len(text), 2),
        'method': 'extractive_frequency_based',
        'key_sentences': selected_sentences,
        'quality_score': round(min(len(selected_sentences) / 3, 1.0), 2),
        'processing_time': processing_time
    }

def fallback_sentiment_analysis(text: str, start_time: float) -> Dict[str, Any]:
    """Ultimate fallback sentiment analysis"""
    # Simple positive/negative word counting
    positive_words = ['good', 'great', 'excellent', 'happy', 'love', 'like', 'best', 'wonderful']
    negative_words = ['bad', 'terrible', 'hate', 'worst', 'awful', 'horrible', 'disappointed']
    
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        sentiment = 'positive'
        confidence = min(0.6 + pos_count * 0.1, 0.8)
    elif neg_count > pos_count:
        sentiment = 'negative'
        confidence = min(0.6 + neg_count * 0.1, 0.8)
    else:
        sentiment = 'neutral'
        confidence = 0.5
    
    processing_time = time.time() - start_time
    
    return {
        'sentiment': sentiment,
        'confidence': confidence,
        'scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
        'method': 'basic_fallback',
        'language_detected': 'unknown',
        'reasoning': 'Basic word counting fallback',
        'processing_time': processing_time
    }

def fallback_emotion_analysis(text: str, start_time: float) -> Dict[str, Any]:
    """Fallback emotion analysis"""
    processing_time = time.time() - start_time
    
    return {
        'text': text,
        'primary_emotion': 'neutral',
        'emotion_scores': {'neutral': 1.0},
        'confidence': 0.5,
        'intensity': 'low',
        'urgency_level': 'low',
        'medical_context': detect_medical_context(text),
        'method': 'fallback',
        'reasoning': 'Fallback emotion analysis',
        'processing_time': processing_time
    }

def generate_overall_summary(texts: List[str], max_length: int = 250) -> Dict[str, Any]:
    """Aggregate analysis and create an overall summary for a batch of comments."""
    start_time = time.time()
    if not texts:
        return {
            'overall_summary': 'No comments provided.',
            'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
            'emotion_distribution': {},
            'key_insights': {},
            'methods_used': ['aggregate_lightweight'],
            'processing_time': 0.0
        }

    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    emotion_counts: Dict[str, int] = {}
    language_counts: Dict[str, int] = {}
    confidences: List[float] = []

    # Analyze each text (lightweight methods)
    for text in texts:
        sentiment_result = analyze_sentiment_lightweight(text)
        sentiment = sentiment_result['sentiment']
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        confidences.append(sentiment_result.get('confidence', 0.5))

        emotion_result = analyze_emotion_lightweight(text)
        primary_emotion = emotion_result.get('primary_emotion', 'neutral')
        emotion_counts[primary_emotion] = emotion_counts.get(primary_emotion, 0) + 1

        lang = detect_language(text)
        language_counts[lang] = language_counts.get(lang, 0) + 1

    total = max(len(texts), 1)
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5

    # Determine dominant emotion & language
    dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else 'neutral'
    dominant_language = max(language_counts.items(), key=lambda x: x[1])[0] if language_counts else 'unknown'

    pos_pct = round((sentiment_counts['positive'] / total) * 100) if total else 0
    neg_pct = round((sentiment_counts['negative'] / total) * 100) if total else 0

    satisfaction_level = 'High' if pos_pct >= 60 else 'Medium' if pos_pct >= 40 else 'Low'
    primary_concerns = 'Negative experience' if neg_pct > pos_pct else 'General feedback'

    # Build concatenated text and summarize
    concatenated = '. '.join(t.strip() for t in texts if t.strip())[:4000]
    summary_data = summarize_text_lightweight(concatenated, max_length=max_length)
    summary_text = summary_data.get('summary', '')

    overall_summary_text = (
        f"Aggregated analysis of {total} comments: {pos_pct}% positive, {sentiment_counts['neutral']} neutral, "
        f"{sentiment_counts['negative']} negative. Dominant emotion is {dominant_emotion}. "
        f"Primary language: {dominant_language}. Average confidence {round(avg_confidence * 100)}%. "
        f"Satisfaction level: {satisfaction_level}."
    )

    # If extractive summary produced, append
    if summary_text and summary_text not in overall_summary_text:
        overall_summary_text += f" Summary: {summary_text}"

    processing_time = time.time() - start_time
    return {
        'overall_summary': overall_summary_text,
        'sentiment_distribution': sentiment_counts,
        'emotion_distribution': emotion_counts,
        'key_insights': {
            'satisfaction_level': satisfaction_level,
            'primary_concerns': primary_concerns,
            'dominant_emotion': dominant_emotion,
            'dominant_language': dominant_language,
            'average_confidence': round(avg_confidence, 3)
        },
        'methods_used': ['aggregate_lightweight', 'extractive_frequency_based'],
        'processing_time': processing_time
    }

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Lightweight Indian E-Consultation Analysis API",
        "version": "3.0.0-lightweight",
        "memory_optimized": True,
        "estimated_memory_usage": f"~{estimate_memory_usage()}MB",
        "models_loaded": list(ML_MODELS.keys()),
        "status": "ready" if MODEL_STATUS["loaded"] else "error"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if MODEL_STATUS["loaded"] else "unhealthy",
        "models_loaded": len(ML_MODELS),
        "estimated_memory": f"~{estimate_memory_usage()}MB",
        "error": MODEL_STATUS.get("error")
    }

@app.post("/analyze/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: TextRequest):
    """Analyze sentiment using lightweight models"""
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Empty text provided")
        
        result = analyze_sentiment_lightweight(request.text)
        
        return SentimentResponse(
            text=request.text,
            sentiment=result['sentiment'],
            confidence=result['confidence'],
            method=result['method'],
            scores=result.get('scores'),
            language_detected=result.get('language_detected'),
            reasoning=result.get('reasoning'),
            processing_time=result.get('processing_time')
        )
    
    except Exception as e:
        logger.error(f"Sentiment analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/emotion", response_model=EmotionResponse)
async def analyze_emotion(request: TextRequest):
    """Analyze emotions using lightweight models"""
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Empty text provided")
        
        result = analyze_emotion_lightweight(request.text)
        
        return EmotionResponse(**result)
    
    except Exception as e:
        logger.error(f"Emotion analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Emotion analysis failed: {str(e)}")

@app.post("/analyze/summarize", response_model=SummaryResponse)
async def summarize_text(request: SummaryRequest):
    """Summarize text using lightweight extractive method"""
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Empty text provided")
        
        result = summarize_text_lightweight(request.text, request.max_length)
        
        return SummaryResponse(**result)
    
    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

@app.post("/analyze/batch")
async def batch_analyze(request: dict):
    """Batch analysis endpoint - simplified for lightweight operation"""
    try:
        texts = request.get('texts', [])
        analyses = request.get('analyses', ['sentiment'])
        
        if not texts:
            raise HTTPException(status_code=400, detail="No texts provided")
        
        start_time = time.time()
        results = []
        
        for text in texts[:10]:  # Limit to 10 texts for memory efficiency
            result = {}
            
            if 'sentiment' in analyses:
                result['sentiment'] = analyze_sentiment_lightweight(text)
            
            if 'emotion' in analyses:
                result['emotions'] = analyze_emotion_lightweight(text)
            
            if 'summarize' in analyses and len(text) > 100:
                result['summary'] = summarize_text_lightweight(text, 100)['summary']
            
            results.append(result)
        
        processing_time = time.time() - start_time
        
        return {
            "count": len(results),
            "results": results,
            "processing_time": processing_time,
            "methods_used": ["lightweight_batch"],
            "memory_efficient": True
        }
    
    except Exception as e:
        logger.error(f"Batch analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@app.post("/analyze/overall", response_model=OverallSummaryResponse)
async def overall_summary(request: OverallSummaryRequest):
    """Generate an overall summary & distributions for a batch of comments."""
    try:
        if not request.texts or not isinstance(request.texts, list):
            raise HTTPException(status_code=400, detail="'texts' must be a non-empty list")
        result = generate_overall_summary(request.texts, request.max_length)
        return OverallSummaryResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Overall summary error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Overall summary failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)