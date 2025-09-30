"""
Advanced Emotion Detection using NLP Models and Multi-modal Analysis
Supports fine-grained emotion detection for Indian e-consultation contexts
"""
import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import nltk
from textblob import TextBlob
import spacy
from collections import Counter
import re
import time
from dataclasses import dataclass

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

logger = logging.getLogger(__name__)

@dataclass
class EmotionResult:
    """Structured result for emotion analysis"""
    primary_emotion: str
    confidence: float
    all_emotions: Dict[str, float]
    intensity: str  # 'low', 'medium', 'high'
    method: str
    processing_time: float
    reasoning: Optional[str] = None
    medical_context_detected: bool = False
    urgency_level: str = "normal"  # 'low', 'normal', 'high', 'urgent'

class AdvancedEmotionDetector:
    """
    Advanced emotion detection using transformer models and linguistic analysis
    Specialized for medical/consultation contexts
    """
    
    def __init__(self):
        self.emotion_pipeline = None
        self.nlp = None
        self.medical_keywords = self._load_medical_keywords()
        self.emotion_keywords = self._load_emotion_keywords()
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize emotion detection models"""
        try:
            logger.info("Loading advanced emotion detection models...")
            
            # Load emotion classification pipeline
            self.emotion_pipeline = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True
            )
            
            # Load spaCy for linguistic features
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy English model not found, using basic features")
                self.nlp = None
            
            logger.info("✅ Advanced emotion models loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load emotion models: {str(e)}")
            raise
    
    def _load_medical_keywords(self) -> Dict[str, List[str]]:
        """Load medical context keywords for better emotion detection"""
        return {
            'treatment': ['treatment', 'medicine', 'prescription', 'therapy', 'cure', 'healing',
                         'इलाज', 'दवा', 'चिकित्सा', 'उपचार'],
            'symptoms': ['pain', 'ache', 'fever', 'cough', 'headache', 'nausea',
                        'दर्द', 'बुखार', 'खांसी', 'सिरदर्द'],
            'healthcare': ['doctor', 'nurse', 'hospital', 'clinic', 'consultation',
                          'डॉक्टर', 'अस्पताल', 'क्लिनिक', 'परामर्श'],
            'urgency': ['emergency', 'urgent', 'critical', 'severe', 'acute',
                       'आपातकाल', 'तत्काल', 'गंभीर']
        }
    
    def _load_emotion_keywords(self) -> Dict[str, Dict[str, List[str]]]:
        """Load emotion-specific keywords for both Hindi and English"""
        return {
            'joy': {
                'hindi': ['खुश', 'प्रसन्न', 'संतुष्ट', 'आनंदित', 'हर्षित', 'खुशी'],
                'english': ['happy', 'joyful', 'pleased', 'delighted', 'satisfied', 'glad', 'cheerful']
            },
            'sadness': {
                'hindi': ['दुखी', 'उदास', 'निराश', 'गमगीन', 'व्यथित', 'दुख'],
                'english': ['sad', 'depressed', 'disappointed', 'melancholy', 'grief', 'sorrow']
            },
            'anger': {
                'hindi': ['गुस्सा', 'क्रोध', 'नाराज', 'रोष', 'क्रोधित', 'चिड़चिड़ाहट'],
                'english': ['angry', 'furious', 'annoyed', 'irritated', 'mad', 'rage', 'frustrated']
            },
            'fear': {
                'hindi': ['डर', 'भय', 'चिंता', 'घबराहट', 'आतंक', 'परेशानी'],
                'english': ['afraid', 'scared', 'worried', 'anxious', 'fearful', 'terrified', 'concerned']
            },
            'surprise': {
                'hindi': ['आश्चर्य', 'हैरान', 'अचंभा', 'विस्मय', 'चौंका'],
                'english': ['surprised', 'amazed', 'astonished', 'shocked', 'stunned', 'bewildered']
            },
            'love': {
                'hindi': ['प्रेम', 'स्नेह', 'मोहब्बत', 'प्यार', 'कृतज्ञता', 'आभार'],
                'english': ['love', 'affection', 'gratitude', 'thankful', 'appreciate', 'grateful']
            },
            'disgust': {
                'hindi': ['घृणा', 'नफरत', 'अरुचि', 'विरक्ति'],
                'english': ['disgusted', 'revolted', 'repulsed', 'nauseated', 'sickened']
            }
        }
    
    def analyze(self, text: str, context: str = "medical") -> EmotionResult:
        """
        Analyze emotions in text with medical context awareness
        
        Args:
            text: Input text for emotion analysis
            context: Context hint ('medical', 'general')
        
        Returns:
            EmotionResult with detailed emotion analysis
        """
        start_time = time.time()
        
        try:
            # Multi-modal emotion detection
            transformer_emotions = self._analyze_with_transformer(text)
            keyword_emotions = self._analyze_with_keywords(text)
            linguistic_features = self._extract_linguistic_features(text)
            medical_context = self._detect_medical_context(text)
            
            # Combine analyses
            combined_emotions = self._combine_emotion_analyses(
                transformer_emotions, keyword_emotions, linguistic_features, medical_context
            )
            
            # Determine primary emotion and confidence
            primary_emotion = max(combined_emotions, key=combined_emotions.get)
            confidence = combined_emotions[primary_emotion]
            
            # Calculate intensity and urgency
            intensity = self._calculate_intensity(text, combined_emotions, linguistic_features)
            urgency = self._assess_urgency(text, combined_emotions, medical_context)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(
                transformer_emotions, keyword_emotions, medical_context, linguistic_features
            )
            
            processing_time = time.time() - start_time
            
            return EmotionResult(
                primary_emotion=primary_emotion,
                confidence=confidence,
                all_emotions=combined_emotions,
                intensity=intensity,
                method="advanced_multimodal",
                processing_time=processing_time,
                reasoning=reasoning,
                medical_context_detected=medical_context['detected'],
                urgency_level=urgency
            )
            
        except Exception as e:
            logger.error(f"Advanced emotion analysis failed: {str(e)}")
            # Fallback to simple keyword analysis
            return self._fallback_analysis(text, time.time() - start_time, str(e))
    
    def _analyze_with_transformer(self, text: str) -> Dict[str, float]:
        """Analyze emotions using transformer model"""
        try:
            if self.emotion_pipeline is None:
                return self._get_default_emotions()
            
            results = self.emotion_pipeline(text)
            
            # Convert to our standard emotion format
            emotion_scores = {}
            for result in results[0]:  # First (and only) text
                label = result['label'].lower()
                score = result['score']
                
                # Map model labels to our emotion categories
                if label in ['joy', 'happiness']:
                    emotion_scores['joy'] = score
                elif label in ['sadness', 'sorrow']:
                    emotion_scores['sadness'] = score
                elif label in ['anger', 'rage']:
                    emotion_scores['anger'] = score
                elif label in ['fear', 'anxiety']:
                    emotion_scores['fear'] = score
                elif label in ['surprise', 'amazement']:
                    emotion_scores['surprise'] = score
                elif label in ['love', 'affection']:
                    emotion_scores['love'] = score
                elif label in ['disgust', 'contempt']:
                    emotion_scores['disgust'] = score
                else:
                    # Handle other emotions by mapping to closest category
                    emotion_scores.setdefault('neutral', 0)
                    emotion_scores['neutral'] += score
            
            # Ensure all basic emotions are present
            for emotion in ['joy', 'sadness', 'anger', 'fear', 'surprise', 'love', 'disgust']:
                if emotion not in emotion_scores:
                    emotion_scores[emotion] = 0.1
            
            return emotion_scores
            
        except Exception as e:
            logger.warning(f"Transformer emotion analysis failed: {e}")
            return self._get_default_emotions()
    
    def _analyze_with_keywords(self, text: str) -> Dict[str, float]:
        """Analyze emotions using keyword matching for Hindi/English"""
        text_lower = text.lower()
        emotion_scores = {emotion: 0.0 for emotion in self.emotion_keywords.keys()}
        
        for emotion, keywords in self.emotion_keywords.items():
            # Count Hindi keywords
            hindi_matches = sum(1 for word in keywords['hindi'] if word in text_lower)
            english_matches = sum(1 for word in keywords['english'] if word in text_lower)
            
            total_matches = hindi_matches + english_matches
            
            # Calculate score with diminishing returns
            if total_matches > 0:
                emotion_scores[emotion] = min(0.3 + (total_matches * 0.2), 0.9)
        
        # If no strong emotions detected, set neutral baseline
        if max(emotion_scores.values()) < 0.3:
            emotion_scores['neutral'] = 0.5
        
        return emotion_scores
    
    def _extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """Extract linguistic features that indicate emotional state"""
        features = {
            'exclamation_marks': text.count('!'),
            'question_marks': text.count('?'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'word_count': len(text.split()),
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
            'punctuation_density': sum(1 for c in text if c in '.,!?;:') / max(len(text), 1)
        }
        
        # Analyze with spaCy if available
        if self.nlp:
            try:
                doc = self.nlp(text)
                features.update({
                    'sentiment_polarity': self._calculate_spacy_sentiment(doc),
                    'entity_count': len(doc.ents),
                    'adjective_count': sum(1 for token in doc if token.pos_ == 'ADJ'),
                    'adverb_count': sum(1 for token in doc if token.pos_ == 'ADV')
                })
            except:
                pass
        
        return features
    
    def _calculate_spacy_sentiment(self, doc) -> float:
        """Calculate sentiment polarity using spaCy linguistic features"""
        # Simple heuristic based on linguistic patterns
        positive_patterns = ['very good', 'excellent', 'amazing', 'wonderful']
        negative_patterns = ['very bad', 'terrible', 'awful', 'horrible']
        
        text_lower = doc.text.lower()
        pos_score = sum(1 for pattern in positive_patterns if pattern in text_lower)
        neg_score = sum(1 for pattern in negative_patterns if pattern in text_lower)
        
        if pos_score > neg_score:
            return min(0.5 + (pos_score * 0.2), 1.0)
        elif neg_score > pos_score:
            return max(-0.5 - (neg_score * 0.2), -1.0)
        else:
            return 0.0
    
    def _detect_medical_context(self, text: str) -> Dict[str, any]:
        """Detect medical context and related concerns"""
        text_lower = text.lower()
        context = {
            'detected': False,
            'categories': [],
            'urgency_indicators': 0,
            'treatment_mentions': 0,
            'symptom_mentions': 0
        }
        
        for category, keywords in self.medical_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > 0:
                context['detected'] = True
                context['categories'].append(category)
                
                if category == 'urgency':
                    context['urgency_indicators'] += matches
                elif category == 'treatment':
                    context['treatment_mentions'] += matches
                elif category == 'symptoms':
                    context['symptom_mentions'] += matches
        
        return context
    
    def _combine_emotion_analyses(self, transformer_emotions: Dict, keyword_emotions: Dict, 
                                linguistic_features: Dict, medical_context: Dict) -> Dict[str, float]:
        """Combine multiple emotion analysis results"""
        # Weights for different analysis methods
        transformer_weight = 0.5
        keyword_weight = 0.3
        linguistic_weight = 0.2
        
        combined = {}
        all_emotions = set(transformer_emotions.keys()) | set(keyword_emotions.keys())
        
        for emotion in all_emotions:
            score = 0.0
            
            # Transformer score
            if emotion in transformer_emotions:
                score += transformer_weight * transformer_emotions[emotion]
            
            # Keyword score
            if emotion in keyword_emotions:
                score += keyword_weight * keyword_emotions[emotion]
            
            # Linguistic adjustment
            if emotion == 'anger' and linguistic_features.get('exclamation_marks', 0) > 1:
                score += linguistic_weight * 0.3
            elif emotion == 'fear' and linguistic_features.get('question_marks', 0) > 1:
                score += linguistic_weight * 0.3
            elif emotion == 'joy' and linguistic_features.get('exclamation_marks', 0) > 0:
                score += linguistic_weight * 0.2
            
            # Medical context adjustments
            if medical_context['detected']:
                if emotion == 'fear' and medical_context['urgency_indicators'] > 0:
                    score *= 1.3  # Amplify fear in urgent medical contexts
                elif emotion == 'sadness' and medical_context['symptom_mentions'] > 0:
                    score *= 1.2  # Amplify sadness when symptoms mentioned
                elif emotion == 'joy' and medical_context['treatment_mentions'] > 0:
                    score *= 1.2  # Amplify joy when treatment is mentioned
            
            combined[emotion] = min(score, 0.95)  # Cap at 95%
        
        # Normalize scores
        total = sum(combined.values())
        if total > 1:
            combined = {k: v/total for k, v in combined.items()}
        
        return combined
    
    def _calculate_intensity(self, text: str, emotions: Dict[str, float], 
                           linguistic_features: Dict) -> str:
        """Calculate emotional intensity based on various factors"""
        max_emotion_score = max(emotions.values()) if emotions else 0
        
        # Factors that increase intensity
        intensity_factors = [
            linguistic_features.get('exclamation_marks', 0) * 0.1,
            linguistic_features.get('uppercase_ratio', 0) * 0.2,
            (max_emotion_score - 0.5) * 2 if max_emotion_score > 0.5 else 0
        ]
        
        intensity_score = sum(intensity_factors)
        
        if intensity_score > 0.7:
            return "high"
        elif intensity_score > 0.3:
            return "medium"
        else:
            return "low"
    
    def _assess_urgency(self, text: str, emotions: Dict[str, float], 
                       medical_context: Dict) -> str:
        """Assess urgency level based on emotions and medical context"""
        urgency_score = 0
        
        # High urgency emotions
        if emotions.get('fear', 0) > 0.7:
            urgency_score += 2
        if emotions.get('anger', 0) > 0.8:
            urgency_score += 1
        if emotions.get('sadness', 0) > 0.8:
            urgency_score += 1
        
        # Medical context urgency
        if medical_context['detected']:
            urgency_score += medical_context['urgency_indicators'] * 2
            if medical_context['symptom_mentions'] > 2:
                urgency_score += 1
        
        # Keywords that indicate urgency
        urgent_keywords = ['emergency', 'urgent', 'help', 'critical', 'severe', 'pain',
                          'आपातकाल', 'तत्काल', 'मदद', 'गंभीर', 'दर्द']
        text_lower = text.lower()
        urgent_keyword_count = sum(1 for keyword in urgent_keywords if keyword in text_lower)
        urgency_score += urgent_keyword_count
        
        if urgency_score >= 4:
            return "urgent"
        elif urgency_score >= 2:
            return "high"
        elif urgency_score >= 1:
            return "normal"
        else:
            return "low"
    
    def _generate_reasoning(self, transformer_emotions: Dict, keyword_emotions: Dict,
                           medical_context: Dict, linguistic_features: Dict) -> str:
        """Generate human-readable reasoning for emotion detection"""
        reasoning_parts = []
        
        # Transformer analysis
        if transformer_emotions:
            top_transformer = max(transformer_emotions, key=transformer_emotions.get)
            reasoning_parts.append(f"Transformer: {top_transformer}({transformer_emotions[top_transformer]:.2f})")
        
        # Keyword analysis
        significant_keywords = {k: v for k, v in keyword_emotions.items() if v > 0.3}
        if significant_keywords:
            top_keyword = max(significant_keywords, key=significant_keywords.get)
            reasoning_parts.append(f"Keywords: {top_keyword}({significant_keywords[top_keyword]:.2f})")
        
        # Medical context
        if medical_context['detected']:
            reasoning_parts.append(f"Medical context: {', '.join(medical_context['categories'])}")
        
        # Linguistic features
        notable_features = []
        if linguistic_features.get('exclamation_marks', 0) > 1:
            notable_features.append("high exclamation")
        if linguistic_features.get('uppercase_ratio', 0) > 0.3:
            notable_features.append("caps usage")
        if notable_features:
            reasoning_parts.append(f"Linguistic: {', '.join(notable_features)}")
        
        return " | ".join(reasoning_parts) if reasoning_parts else "Basic analysis"
    
    def _fallback_analysis(self, text: str, processing_time: float, error: str) -> EmotionResult:
        """Fallback analysis when advanced models fail"""
        # Simple keyword-based fallback
        keyword_emotions = self._analyze_with_keywords(text)
        primary_emotion = max(keyword_emotions, key=keyword_emotions.get) if keyword_emotions else 'neutral'
        
        return EmotionResult(
            primary_emotion=primary_emotion,
            confidence=0.5,
            all_emotions=keyword_emotions or {'neutral': 1.0},
            intensity="medium",
            method="fallback_keywords",
            processing_time=processing_time,
            reasoning=f"Fallback analysis due to: {error}",
            medical_context_detected=False,
            urgency_level="normal"
        )
    
    def _get_default_emotions(self) -> Dict[str, float]:
        """Get default emotion scores when analysis fails"""
        return {
            'joy': 0.15,
            'sadness': 0.15,
            'anger': 0.15,
            'fear': 0.15,
            'surprise': 0.10,
            'love': 0.10,
            'disgust': 0.10,
            'neutral': 0.10
        }

# Factory function
def create_emotion_detector() -> AdvancedEmotionDetector:
    """Create and return advanced emotion detector"""
    try:
        return AdvancedEmotionDetector()
    except Exception as e:
        logger.error(f"Failed to create advanced emotion detector: {str(e)}")
        return None