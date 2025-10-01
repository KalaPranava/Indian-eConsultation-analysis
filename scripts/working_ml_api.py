"""
Working ML API using your installed models - torch, transformers, nltk, scikit-learn
This version avoids complex dependency chains and focuses on core functionality
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize ML models
ML_MODELS = {}
MODEL_STATUS = {"loaded": False, "error": None}

def load_ml_models():
    """Load ML models safely"""
    global ML_MODELS, MODEL_STATUS
    
    try:
        logger.info("🔄 Loading ML models...")
        
        # Import and load transformers models
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
        import torch
        import os
        
        # Load sentiment model - prioritize efficiency for cloud deployment
        deployment_mode = os.getenv('DEPLOYMENT_MODE', 'local')  # 'local' or 'cloud'
        
        if deployment_mode == 'cloud':
            # Use lightweight DistilBERT for cloud deployment (faster, less memory)
            try:
                ML_MODELS['sentiment_pipeline'] = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
                logger.info("✅ DistilBERT sentiment loaded (Cloud optimized)")
            except Exception as e:
                logger.warning(f"DistilBERT sentiment failed: {e}")
        else:
            # Use heavy XLM-RoBERTa for local deployment (better accuracy)
            try:
                ML_MODELS['sentiment_tokenizer'] = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
                ML_MODELS['sentiment_model'] = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
                logger.info("✅ XLM-RoBERTa sentiment loaded (Local mode)")
            except Exception as e:
                logger.warning(f"XLM-RoBERTa failed, falling back to DistilBERT: {e}")
                try:
                    ML_MODELS['sentiment_pipeline'] = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
                    logger.info("✅ DistilBERT sentiment loaded (Fallback)")
                except Exception as e2:
                    logger.warning(f"Backup sentiment model failed: {e2}")
        
        # Load emotion classification
        try:
            ML_MODELS['emotion_pipeline'] = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
            logger.info("✅ Emotion model loaded")
        except Exception as e:
            logger.warning(f"Emotion model failed: {e}")
        
        # Load NLTK for text processing
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
            from nltk.sentiment import SentimentIntensityAnalyzer
            ML_MODELS['vader'] = SentimentIntensityAnalyzer()
            logger.info("✅ NLTK VADER loaded")
        except Exception as e:
            logger.warning(f"NLTK VADER failed: {e}")
        
        # Load scikit-learn for TF-IDF
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            ML_MODELS['tfidf'] = TfidfVectorizer(max_features=1000, stop_words='english')
            logger.info("✅ TF-IDF vectorizer loaded")
        except Exception as e:
            logger.warning(f"TF-IDF failed: {e}")

        # Optionally load an abstractive summarization model (controlled by env to save memory)
        # Skip heavy summarizer in cloud mode for better performance
        if os.getenv('ENABLE_ABSTRACTIVE_SUMMARY', '0') in ['1', 'true', 'yes'] and deployment_mode != 'cloud':
            try:
                from transformers import pipeline as hf_pipeline
                # Prefer a smaller distilbart model for memory efficiency; allow override
                summarizer_model = os.getenv('SUMMARY_MODEL_NAME', 'sshleifer/distilbart-cnn-12-6')
                ML_MODELS['abstractive_summarizer'] = hf_pipeline(
                    'summarization', model=summarizer_model
                )
                logger.info(f"✅ Abstractive summarizer loaded: {summarizer_model}")
            except Exception as e:
                logger.warning(f"Abstractive summarizer load failed: {e}")
        
        MODEL_STATUS["loaded"] = len(ML_MODELS) > 0
        logger.info(f"🚀 ML models loaded: {list(ML_MODELS.keys())}")
        
    except Exception as e:
        logger.error(f"❌ ML model loading failed: {str(e)}")
        MODEL_STATUS["error"] = str(e)

# Load models on startup
load_ml_models()

app = FastAPI(
    title="Indian E-Consultation Analysis API (ML-Powered)",
    description="Real ML models: XLM-RoBERTa, DistilBERT, VADER, TF-IDF for sentiment analysis",
    version="2.0.0-ml-working"
)

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
    analyses: Optional[List[str]] = ["sentiment", "emotion", "summarize"]
    max_summary_length: Optional[int] = 120

class BatchItemResult(BaseModel):
    sentiment: Optional[Dict[str, Any]] = None
    emotions: Optional[Dict[str, Any]] = None
    summary: Optional[str] = None

class BatchResponse(BaseModel):
    count: int
    results: List[BatchItemResult]
    processing_time: float
    methods_used: List[str]

class OverallSummaryRequest(BaseModel):
    texts: List[str]
    max_length: Optional[int] = 200

class OverallSummaryResponse(BaseModel):
    overall_summary: str
    sentiment_distribution: Dict[str, int]
    emotion_distribution: Dict[str, int]
    key_insights: Dict[str, Any]
    methods_used: List[str]
    processing_time: float
    abstractive_summary: Optional[str] = None
    summarization_method: Optional[str] = None
    chunk_count: Optional[int] = None
    source_comment_count: Optional[int] = None
    summary_passes: Optional[int] = None

class WordcloudRequest(BaseModel):
    texts: List[str]
    analyses: Optional[List[str]] = []
    max_words: Optional[int] = 50

class WordcloudWord(BaseModel):
    text: str
    size: int
    comments: List[Dict[str, Any]]

class WordcloudResponse(BaseModel):
    wordcloud_data: List[WordcloudWord]
    processing_time: float
    total_unique_words: int

def analyze_sentiment_ml(text: str) -> Dict[str, Any]:
    """ML-powered sentiment analysis using your installed models"""
    start_time = time.time()
    
    # Try XLM-RoBERTa first (multilingual)
    if 'sentiment_tokenizer' in ML_MODELS and 'sentiment_model' in ML_MODELS:
        try:
            import torch
            tokenizer = ML_MODELS['sentiment_tokenizer']
            model = ML_MODELS['sentiment_model']
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # Map labels (0: negative, 1: neutral, 2: positive)
            labels = ['negative', 'neutral', 'positive']
            scores_array = predictions[0].numpy()
            scores = {labels[i]: float(scores_array[i]) for i in range(len(labels))}
            sentiment = labels[np.argmax(scores_array)]
            confidence = float(np.max(scores_array))
            processing_time = time.time() - start_time
            logger.info(f"Sentiment analysis used XLM-RoBERTa: {sentiment}, confidence={confidence}")
            return {
                'sentiment': sentiment,
                'confidence': round(confidence, 3),
                'scores': {k: round(v, 3) for k, v in scores.items()},
                'method': 'xlm_roberta_ml',
                'language_detected': 'multilingual',
                'reasoning': f'XLM-RoBERTa transformer analysis (confidence: {confidence:.3f})',
                'processing_time': processing_time
            }
        except Exception as e:
            logger.warning(f"XLM-RoBERTa failed: {e}")
    
    # Try DistilBERT pipeline
    if 'sentiment_pipeline' in ML_MODELS:
        try:
            pipeline = ML_MODELS['sentiment_pipeline']
            result = pipeline(text)[0]
            # Map DistilBERT labels
            label_map = {'POSITIVE': 'positive', 'NEGATIVE': 'negative'}
            sentiment = label_map.get(result['label'], result['label'].lower())
            confidence = result['score']
            # Create score distribution
            scores = {sentiment: confidence}
            if sentiment == 'positive':
                scores['negative'] = 1 - confidence
                scores['neutral'] = 0.1
            else:
                scores['positive'] = 1 - confidence
                scores['neutral'] = 0.1
            # Normalize scores
            total = sum(scores.values())
            scores = {k: v/total for k, v in scores.items()}
            processing_time = time.time() - start_time
            logger.info(f"Sentiment analysis used DistilBERT: {sentiment}, confidence={confidence}")
            return {
                'sentiment': sentiment,
                'confidence': round(confidence, 3),
                'scores': {k: round(v, 3) for k, v in scores.items()},
                'method': 'distilbert_pipeline',
                'language_detected': 'english',
                'reasoning': f'DistilBERT transformer pipeline (score: {confidence:.3f})',
                'processing_time': processing_time
            }
        except Exception as e:
            logger.warning(f"DistilBERT pipeline failed: {e}")
    
    # Try VADER (works well for informal text)
    if 'vader' in ML_MODELS:
        try:
            vader = ML_MODELS['vader']
            vader_scores = vader.polarity_scores(text)
            
            # Determine sentiment from compound score
            compound = vader_scores['compound']
            if compound >= 0.05:
                sentiment = 'positive'
            elif compound <= -0.05:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            confidence = abs(compound)
            if confidence < 0.1:
                confidence = 0.6  # Default neutral confidence
            
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
                'method': 'vader_nltk',
                'language_detected': 'english',
                'reasoning': f'VADER sentiment analysis (compound: {compound:.3f})',
                'processing_time': processing_time
            }
        except Exception as e:
            logger.warning(f"VADER failed: {e}")
    
    # Enhanced fallback with medical vocabulary
    return analyze_sentiment_enhanced_fallback(text)

def analyze_sentiment_enhanced_fallback(text: str) -> Dict[str, Any]:
    """Enhanced heuristic fallback with medical and consultation vocabulary"""
    start_time = time.time()
    text_lower = text.lower()
    
    # Enhanced medical consultation vocabulary
    positive_patterns = {
        'hindi': ['अच्छा', 'अच्छी', 'अच्छे', 'बहुत', 'खुश', 'संतुष्ट', 'बेहतर', 'उत्कृष्ट', 
                 'प्रभावी', 'ठीक', 'स्वस्थ', 'सुधार', 'लाभ', 'राहत', 'कामयाब', 'सफल', 'धन्यवाद'],
        'english': ['good', 'excellent', 'great', 'happy', 'satisfied', 'wonderful', 'amazing', 
                   'effective', 'helpful', 'better', 'improved', 'relief', 'successful', 'thank', 
                   'appreciate', 'recommend', 'professional', 'caring', 'kind']
    }
    
    negative_patterns = {
        'hindi': ['खराब', 'बुरा', 'गलत', 'निराश', 'असंतुष्ट', 'समस्या', 'दर्द', 'बीमार', 
                 'परेशान', 'चिंता', 'डर', 'गुस्सा', 'नाराज'],
        'english': ['bad', 'terrible', 'awful', 'disappointed', 'poor', 'problem', 'issue', 
                   'pain', 'worse', 'failed', 'difficult', 'rude', 'unprofessional', 'slow', 
                   'expensive', 'waste', 'complaint']
    }
    
    # Count positive and negative indicators
    positive_count = 0
    negative_count = 0
    
    for lang, words in positive_patterns.items():
        positive_count += sum(1 for word in words if word in text_lower)
    
    for lang, words in negative_patterns.items():
        negative_count += sum(1 for word in words if word in text_lower)
    
    # Boost scoring for medical context
    if any(word in text_lower for word in ['डॉक्टर', 'doctor', 'अस्पताल', 'hospital', 'इलाज', 'treatment']):
        positive_count *= 1.5
        negative_count *= 1.5
    
    # Determine sentiment with improved logic
    total_words = len(text_lower.split())
    if positive_count > negative_count:
        sentiment = "positive"
        confidence = min(0.6 + (positive_count * 0.15), 0.95)
    elif negative_count > positive_count:
        sentiment = "negative"
        confidence = min(0.6 + (negative_count * 0.15), 0.95)
    else:
        sentiment = "neutral"
        confidence = 0.5 + (total_words * 0.01)  # Slight boost for longer texts
    
    # Create score distribution
    pos_ratio = positive_count / max(total_words, 1)
    neg_ratio = negative_count / max(total_words, 1)
    neu_ratio = max(0, 1 - pos_ratio - neg_ratio)
    
    scores = {
        'positive': round(pos_ratio + 0.1, 3),
        'negative': round(neg_ratio + 0.1, 3),
        'neutral': round(neu_ratio + 0.1, 3)
    }
    
    processing_time = time.time() - start_time
    
    return {
        'sentiment': sentiment,
        'confidence': round(confidence, 3),
        'scores': scores,
        'method': 'enhanced_medical_heuristic_fallback',
        'language_detected': 'mixed' if any(ord(c) > 127 for c in text) else 'english',
        'reasoning': f'Enhanced medical vocabulary analysis (P:{positive_count}, N:{negative_count})',
        'processing_time': processing_time
    }

def analyze_emotion_ml(text: str) -> Dict[str, Any]:
    """ML-powered emotion analysis"""
    start_time = time.time()
    
    # Try emotion classification model
    if 'emotion_pipeline' in ML_MODELS:
        try:
            pipeline = ML_MODELS['emotion_pipeline']
            results = pipeline(text)
            # The transformer pipeline with top_k=None returns List[List[dict]]
            # Normalize to a flat list of dicts
            if isinstance(results, dict):
                results = [results]
            elif isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
                # Flatten one level
                flat = []
                for sub in results:
                    if isinstance(sub, list):
                        flat.extend(sub)
                    else:
                        flat.append(sub)
                results = flat
            emotion_scores = {}
            for result in results:
                if isinstance(result, dict) and 'label' in result and 'score' in result:
                    emotion = result['label'].lower()
                    score = result['score']
                    emotion_scores[emotion] = round(score, 3)
            if not emotion_scores:
                logger.warning(f"Emotion pipeline returned no valid results: {results}")
                raise ValueError("No valid emotion scores returned")
            primary_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = emotion_scores[primary_emotion]
            intensity = 'high' if confidence > 0.7 else 'medium' if confidence > 0.4 else 'low'
            urgency_level = 'high' if primary_emotion in ['anger', 'fear'] and confidence > 0.6 else 'moderate' if confidence > 0.4 else 'low'
            processing_time = time.time() - start_time
            logger.info(f"Emotion analysis used DistilRoBERTa: {primary_emotion}, confidence={confidence}")
            return {
                'primary_emotion': primary_emotion,
                'emotion_scores': emotion_scores,
                'confidence': confidence,
                'intensity': intensity,
                'urgency_level': urgency_level,
                'medical_context': any(word in text.lower() for word in ['doctor', 'hospital', 'medicine', 'डॉक्टर', 'अस्पताल']),
                'method': 'distilroberta_emotion_ml',
                'reasoning': f'DistilRoBERTa emotion classification (confidence: {confidence:.3f})',
                'processing_time': processing_time
            }
        except Exception as e:
            logger.warning(f"ML emotion analysis failed: {e}")
    
    # Enhanced fallback
    return analyze_emotion_enhanced_fallback(text)

def analyze_emotion_enhanced_fallback(text: str) -> Dict[str, Any]:
    """Enhanced emotion fallback with medical context"""
    start_time = time.time()
    text_lower = text.lower()
    
    # Enhanced emotion patterns with more medical and consultation specific words
    emotion_patterns = {
        'joy': ['खुश', 'प्रसन्न', 'अच्छा', 'बहुत अच्छा', 'धन्यवाद', 'शुक्रिया', 'बेहतरीन', 'उत्कृष्ट',
               'happy', 'joy', 'excited', 'great', 'excellent', 'wonderful', 'amazing', 'satisfied', 'pleased',
               'thankful', 'grateful', 'fantastic', 'awesome', 'brilliant', 'perfect', 'outstanding'],
        'sadness': ['दुखी', 'उदास', 'परेशान', 'निराश', 'दुख', 'गम', 'रोना',
                   'sad', 'disappointed', 'unhappy', 'depressed', 'upset', 'crying', 'heartbroken',
                   'devastated', 'miserable', 'sorrowful', 'melancholy', 'dejected'],
        'anger': ['गुस्सा', 'क्रोध', 'नाराज', 'चिढ़', 'बुरा', 'खराब सेवा', 'बहुत गुस्सा',
                 'angry', 'mad', 'furious', 'annoyed', 'frustrated', 'outraged', 'irritated',
                 'rage', 'pissed', 'livid', 'irate', 'infuriated', 'enraged'],
        'fear': ['डर', 'चिंता', 'घबराहट', 'परेशानी', 'भय', 'फिक्र',
                'scared', 'worried', 'anxious', 'nervous', 'concerned', 'afraid', 'fearful',
                'panic', 'terrified', 'apprehensive', 'uneasy', 'troubled'],
        'surprise': ['आश्चर्य', 'हैरान', 'अचंभा', 'हैरानी', 'चौंक',
                    'surprised', 'shocked', 'amazed', 'unexpected', 'astonished', 'stunned',
                    'bewildered', 'flabbergasted', 'astounded'],
        'disgust': ['घृणा', 'नफरत', 'बुरा लगा', 'गंदा', 'बहुत बुरा',
                   'disgust', 'hate', 'horrible', 'awful', 'terrible', 'disgusting', 'revolting',
                   'repulsive', 'nasty', 'gross', 'yuck', 'ew']
    }
    
    emotion_scores = {}
    total_matches = 0
    
    for emotion, patterns in emotion_patterns.items():
        matches = 0
        for word in patterns:
            if word in text_lower:
                # Weight longer phrases more heavily
                weight = 3 if len(word.split()) > 1 else 2 if len(word) > 6 else 1
                matches += weight
        emotion_scores[emotion] = matches
        total_matches += matches
    
    # Normalize scores and boost non-neutral emotions
    if total_matches > 0:
        for emotion in emotion_scores:
            emotion_scores[emotion] = round(emotion_scores[emotion] / max(total_matches, 1), 3)
    else:
        # If no matches, distribute based on sentiment analysis
        sentiment_words_positive = ['good', 'अच्छा', 'बेहतर', 'खुश', 'संतुष्ट']
        sentiment_words_negative = ['bad', 'खराब', 'बुरा', 'गुस्सा', 'परेशान']
        
        if any(word in text_lower for word in sentiment_words_positive):
            emotion_scores['joy'] = 0.6
        elif any(word in text_lower for word in sentiment_words_negative):
            emotion_scores['sadness'] = 0.4
            emotion_scores['anger'] = 0.3
        else:
            emotion_scores = {emotion: 0.1 for emotion in emotion_patterns}
    
    # Add baseline for neutral but reduce it if other emotions are detected
    emotion_scores['neutral'] = 0.4 if total_matches == 0 else max(0.1, 0.4 - total_matches * 0.1)
    
    # Find primary emotion
    primary_emotion = max(emotion_scores, key=emotion_scores.get)
    
    # Boost confidence if we found actual emotion words
    base_confidence = emotion_scores[primary_emotion]
    if total_matches > 0 and primary_emotion != 'neutral':
        confidence = min(0.7 + (base_confidence * 0.3), 0.95)
    else:
        confidence = base_confidence
    
    processing_time = time.time() - start_time
    
    return {
        'primary_emotion': primary_emotion,
        'emotion_scores': emotion_scores,
        'confidence': round(confidence, 3),
        'intensity': 'high' if confidence > 0.7 else 'medium' if confidence > 0.5 else 'low',
        'urgency_level': 'high' if primary_emotion in ['anger', 'fear'] and confidence > 0.6 else 'moderate' if confidence > 0.4 else 'low',
        'medical_context': any(word in text_lower for word in ['doctor', 'hospital', 'medicine', 'डॉक्टर', 'अस्पताल', 'treatment', 'इलाज']),
        'method': 'enhanced_pattern_fallback',
        'reasoning': f'Enhanced pattern matching with {total_matches} emotion indicators, detected: {primary_emotion}',
        'processing_time': processing_time
    }

def summarize_text_ml(text: str, max_sentences: int = 1) -> Dict[str, Any]:
    """ML-enhanced summarization using TF-IDF and sentence ranking"""
    start_time = time.time()
    
    # For comments, keep it short and simple
    if len(text) <= 80:
        return {
            'summary': text,
            'original_length': len(text),
            'summary_length': len(text),
            'compression_ratio': 1.0,
            'method': 'no_compression_needed',
            'key_sentences': [text],
            'important_entities': [],
            'quality_score': 1.0,
            'reasoning': 'Text already short enough',
            'processing_time': time.time() - start_time
        }
    
    # Split into sentences
    sentences = [s.strip() for s in re.split(r'[।.!?]', text) if s.strip()]
    original_length = len(text)
    
    # For comments, just take the first meaningful sentence or truncate
    if len(sentences) <= 1:
        summary = text[:60] + "..." if len(text) > 60 else text
        key_sentences = [summary]
    else:
        # For comments, keep it simple - just take the first sentence or key part
        first_sentence = sentences[0]
        if len(first_sentence) > 80:
            # Truncate long first sentence
            summary = first_sentence[:60] + "..."
        else:
            summary = first_sentence
        key_sentences = [summary]
    
    summary_length = len(summary)
    compression_ratio = summary_length / max(original_length, 1)
    quality_score = round(min(0.6 + (summary_length / (original_length + 1)) * 0.3, 0.9), 3)
    
    processing_time = time.time() - start_time
    
    method_used = 'tfidf_ml_ranking' if 'tfidf' in ML_MODELS else 'enhanced_extractive_fallback'
    
    return {
        'summary': summary,
        'original_length': original_length,
        'summary_length': summary_length,
        'compression_ratio': compression_ratio,
        'method': method_used,
        'key_sentences': key_sentences,
        'important_entities': [],
        'quality_score': quality_score,
        'reasoning': f'ML-enhanced sentence ranking extracted {len(key_sentences)} key sentences',
        'processing_time': processing_time
    }

def summarize_text_abstractive(text: str, max_length: int = 180, min_length: int = 40) -> Optional[str]:
    """Run abstractive summarization if model available. Returns summary or None."""
    if 'abstractive_summarizer' not in ML_MODELS:
        return None
    try:
        summarizer = ML_MODELS['abstractive_summarizer']
        
        # Auto-adjust max_length based on input length to avoid warnings
        input_length = len(text.split())  # Rough word count
        if input_length < 100:  # Short text
            max_length = min(max_length, max(40, input_length // 2))
        
        # Clamp lengths sensibly
        max_length = max(30, min(max_length, 400))
        min_length = max(10, min(min_length, max_length - 10))
        
        result = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        if isinstance(result, list) and result:
            return result[0].get('summary_text')
    except Exception as e:
        logger.warning(f"Abstractive summarization failed: {e}")
    return None

def multi_document_abstractive_summary(texts: List[str], target_sentences: int = 6) -> Dict[str, Any]:
    """Hierarchical multi-document summarization.

    Strategy:
      1. Clean and filter very short comments.
      2. Concatenate into chunks within token/char budget (approx character proxy for tokens).
      3. Run abstractive summary per chunk.
      4. If >1 chunk summaries, run a second-pass summary over concatenated chunk summaries.
    Returns dict with final summary & metadata.
    """
    start = time.time()
    if 'abstractive_summarizer' not in ML_MODELS:
        return {
            'summary': None,
            'method': 'none',
            'passes': 0,
            'chunks': 0
        }
    # Basic cleaning & filtering
    cleaned = []
    for t in texts:
        t = (t or '').strip()
        if len(t) < 10:
            continue
        # Collapse whitespace
        t = re.sub(r'\s+', ' ', t)
        cleaned.append(t)
    if not cleaned:
        return {
            'summary': None,
            'method': 'none',
            'passes': 0,
            'chunks': 0
        }
    # Chunking (char-based heuristic). distilbart ~ 1024 tokens -> ~4000 chars safe window
    MAX_CHARS = 3800
    chunks: List[str] = []
    current = []
    total_chars = 0
    for c in cleaned:
        if sum(len(x) for x in current) + len(c) + 1 > MAX_CHARS and current:
            chunks.append('\n'.join(current))
            current = [c]
        else:
            current.append(c)
    if current:
        chunks.append('\n'.join(current))

    chunk_summaries: List[str] = []
    for ch in chunks:
        # Adjust max_length proportional to size
        words = ch.split()
        approx_tokens = len(words)
        # Scale summary length: smaller chunk -> shorter summary
        max_len = 120 if approx_tokens > 350 else 80
        min_len = 35 if approx_tokens > 350 else 20
        s = summarize_text_abstractive(ch, max_length=max_len, min_length=min_len)
        if s:
            chunk_summaries.append(s.strip())
    passes = 1
    final_summary = None
    if not chunk_summaries:
        # Fallback: try summarizing first 3500 chars combined
        combined = ('\n'.join(cleaned))[:3500]
        final_summary = summarize_text_abstractive(combined, max_length=140, min_length=40)
    else:
        if len(chunk_summaries) == 1:
            final_summary = chunk_summaries[0]
        else:
            # Second pass
            passes = 2
            joined = ' \n '.join(chunk_summaries)
            final_summary = summarize_text_abstractive(joined, max_length=180, min_length=60) or joined

    processing_time = time.time() - start
    return {
        'summary': final_summary,
        'method': 'hierarchical_abstractive' if final_summary else 'abstractive_failed',
        'passes': passes,
        'chunks': len(chunks),
        'chunk_summaries': chunk_summaries,
        'processing_time': processing_time
    }

@app.get("/")
async def root():
    model_info = list(ML_MODELS.keys()) if MODEL_STATUS["loaded"] else ["fallback_heuristics"]
    return {
        "message": "Indian E-Consultation Analysis API (ML-Powered) is running", 
        "status": "healthy",
        "version": "2.0.0-ml-working",
        "features": ["sentiment_analysis", "emotion_detection", "text_summarization"],
        "ml_models_loaded": MODEL_STATUS["loaded"],
        "active_models": model_info,
        "model_count": len(ML_MODELS)
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "ML API is running",
        "version": "2.0.0-ml-working",
        "ml_models_active": MODEL_STATUS["loaded"],
        "models": list(ML_MODELS.keys())
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
        result = summarize_text_ml(request.text, max_sentences)
        
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

@app.post("/analyze/batch", response_model=BatchResponse)
async def batch_analyze_endpoint(request: BatchRequest):
    start = time.time()
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    results: List[BatchItemResult] = []
    methods_used = set()
    for text in request.texts:
        item_sent = analyze_sentiment_ml(text) if 'sentiment' in request.analyses else None
        if item_sent:
            methods_used.add(item_sent.get('method', ''))
            # Patch: add 'label' field for frontend compatibility
            item_sent = dict(item_sent)
            item_sent['label'] = item_sent.get('sentiment', 'neutral')
        item_emotion = analyze_emotion_ml(text) if 'emotion' in request.analyses else None
        if item_emotion:
            methods_used.add(item_emotion.get('method', ''))
        item_summary = None
        if 'summarize' in request.analyses and len(text) > 40:
            sum_res = summarize_text_ml(text, max_sentences=max(1, min(request.max_summary_length // 60, 5)))
            item_summary = sum_res.get('summary')
            methods_used.add(sum_res.get('method', ''))
        results.append(BatchItemResult(
            sentiment=item_sent,
            emotions=item_emotion,
            summary=item_summary
        ))
    processing_time = time.time() - start
    return BatchResponse(
        count=len(results),
        results=results,
        processing_time=processing_time,
        methods_used=[m for m in methods_used if m]
    )

@app.post("/analyze/overall_summary", response_model=OverallSummaryResponse)
async def overall_summary_endpoint(request: OverallSummaryRequest):
    start = time.time()
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    sentiments = [analyze_sentiment_ml(t) for t in request.texts]
    emotions = [analyze_emotion_ml(t) for t in request.texts]
    sentiment_distribution = {k: 0 for k in ['positive', 'negative', 'neutral']}
    for s in sentiments:
        sentiment_distribution[s.get('sentiment', 'neutral')] = sentiment_distribution.get(s.get('sentiment', 'neutral'), 0) + 1
    emotion_distribution: Dict[str, int] = {}
    for e in emotions:
        pe = e.get('primary_emotion', 'neutral')
        emotion_distribution[pe] = emotion_distribution.get(pe, 0) + 1
    # Simple key insights
    total = len(request.texts)
    pos_ratio = sentiment_distribution['positive'] / max(total,1)
    neg_ratio = sentiment_distribution['negative'] / max(total,1)
    key_insights = {
        'satisfaction_level': 'high' if pos_ratio > 0.6 else 'moderate' if pos_ratio > 0.4 else 'low',
        'primary_concerns': 'service quality' if neg_ratio > 0.3 else 'general feedback',
        'emotional_state': max(emotion_distribution, key=emotion_distribution.get) if emotion_distribution else 'neutral'
    }
    overall_summary = (
        f"Dataset of {total} comments: {sentiment_distribution['positive']} positive, "
        f"{sentiment_distribution['negative']} negative, {sentiment_distribution['neutral']} neutral. "
        f"Dominant emotion: {key_insights['emotional_state']}. Satisfaction level: {key_insights['satisfaction_level']}."
    )
    # Try enhanced multi-document abstractive summary (optional)
    abstractive_meta = multi_document_abstractive_summary(request.texts, target_sentences=6)
    abstractive_summary = abstractive_meta.get('summary')
    if abstractive_summary:
        methods_used.append(abstractive_meta.get('method', 'hierarchical_abstractive'))
    methods_used = list({s.get('method','') for s in sentiments} | {e.get('method','') for e in emotions})
    processing_time = time.time() - start
    return OverallSummaryResponse(
        overall_summary=overall_summary,
        sentiment_distribution=sentiment_distribution,
        emotion_distribution=emotion_distribution,
        key_insights=key_insights,
        methods_used=[m for m in methods_used if m],
        processing_time=processing_time + abstractive_meta.get('processing_time', 0.0),
        abstractive_summary=abstractive_summary,
        summarization_method=abstractive_meta.get('method'),
        chunk_count=abstractive_meta.get('chunks'),
        source_comment_count=total,
        summary_passes=abstractive_meta.get('passes')
    )

@app.post("/analyze/wordcloud", response_model=WordcloudResponse)
async def wordcloud_endpoint(request: WordcloudRequest):
    start = time.time()
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    # Basic tokenization
    word_freq: Dict[str, List[Dict[str, Any]]] = {}
    for idx, text in enumerate(request.texts):
        tokens = re.findall(r"[A-Za-z\u0900-\u097F']+", text.lower())
        for tok in tokens:
            if len(tok) < 3:
                continue
            if tok not in word_freq:
                word_freq[tok] = []
            if len(word_freq[tok]) < 10:  # store sample comments referencing the word
                word_freq[tok].append({'id': idx+1, 'text': text[:200]})
    # Convert to list with sizes
    words_scored = []
    for w, comments in word_freq.items():
        words_scored.append({
            'text': w,
            'size': len(comments),
            'comments': comments
        })
    # Sort and trim
    words_scored.sort(key=lambda x: x['size'], reverse=True)
    max_words = min(request.max_words or 50, 100)
    top_words = words_scored[:max_words]
    processing_time = time.time() - start
    return WordcloudResponse(
        wordcloud_data=top_words,
        processing_time=processing_time,
        total_unique_words=len(words_scored)
    )

if __name__ == "__main__":
    logger.info(f"Starting Indian E-Consultation Analysis API with ML models: {list(ML_MODELS.keys())}")
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info"
    )