"""
Unified inference pipeline for sentiment analysis, summarization, and emotion detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from pathlib import Path
import yaml
import time
from datetime import datetime

from ..data import DataLoader, TextPreprocessor
from ..models import (
    SentimentAnalyzer, HybridSummarizer, EmotionDetector,
    TextRankSummarizer, AbstractiveSummarizer
)

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    Unified inference pipeline for all analysis tasks.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the inference pipeline.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize components with error handling
        try:
            self.preprocessor = TextPreprocessor(download_nltk_data=False)
            logger.info("Text preprocessor initialized successfully")
        except Exception as e:
            logger.warning(f"Text preprocessor initialization warning: {e}")
            self.preprocessor = None
        
        # Model instances (lazy loading with fallbacks)
        self.sentiment_analyzer = None
        self.emotion_detector = None
        self.summarizer = None
        
        # Fallback mode for when models can't be loaded
        self.fallback_mode = False
        
        # Performance tracking
        self.inference_stats = {
            'total_requests': 0,
            'sentiment_requests': 0,
            'emotion_requests': 0,
            'summarization_requests': 0,
            'avg_processing_time': 0,
            'last_request_time': None
        }
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()
            
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'models': {
                'sentiment': {
                    'name': 'ai4bharat/indic-bert',
                    'cache_dir': './models/sentiment'
                },
                'emotion': {
                    'name': 'cardiffnlp/twitter-roberta-base-emotion',
                    'cache_dir': './models/emotion'
                },
                'summarization': {
                    'extractive': {'method': 'textrank'},
                    'abstractive': {
                        'name': 't5-small',
                        'cache_dir': './models/summarization'
                    }
                }
            },
            'preprocessing': {
                'clean_html': True,
                'remove_urls': True,
                'remove_mentions': True,
                'min_length': 10,
                'max_length': 1000
            }
        }
        
    def _initialize_sentiment_analyzer(self):
        """Initialize sentiment analyzer if not already loaded."""
        if self.sentiment_analyzer is None:
            try:
                logger.info("Initializing sentiment analyzer")
                sentiment_config = self.config.get('models', {}).get('sentiment', {})
                
                self.sentiment_analyzer = SentimentAnalyzer(
                    model_name=sentiment_config.get('name', 'ai4bharat/indic-bert'),
                    cache_dir=sentiment_config.get('cache_dir', './models/sentiment')
                )
                self.sentiment_analyzer.load_model()
                logger.info("Sentiment analyzer initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize sentiment analyzer: {e}")
                self.sentiment_analyzer = None
                self.fallback_mode = True
            
    def _initialize_emotion_detector(self):
        """Initialize emotion detector if not already loaded."""
        if self.emotion_detector is None:
            try:
                logger.info("Initializing emotion detector")
                emotion_config = self.config.get('models', {}).get('emotion', {})
                
                self.emotion_detector = EmotionDetector(
                    model_name=emotion_config.get('name', 'cardiffnlp/twitter-roberta-base-emotion'),
                    cache_dir=emotion_config.get('cache_dir', './models/emotion')
                )
                self.emotion_detector.load_model()
                logger.info("Emotion detector initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize emotion detector: {e}")
                self.emotion_detector = None
                self.fallback_mode = True
            
    def _initialize_summarizer(self):
        """Initialize summarizer if not already loaded."""
        if self.summarizer is None:
            try:
                logger.info("Initializing summarizer")
                summarization_config = self.config.get('models', {}).get('summarization', {})
                
                # Initialize extractive summarizer
                extractive_summarizer = TextRankSummarizer()
                
                # Initialize abstractive summarizer
                abstractive_config = summarization_config.get('abstractive', {})
                abstractive_summarizer = AbstractiveSummarizer(
                    model_name=abstractive_config.get('name', 't5-small'),
                    cache_dir=abstractive_config.get('cache_dir', './models/summarization')
                )
                
                # Create hybrid summarizer
                self.summarizer = HybridSummarizer(
                    extractive_summarizer=extractive_summarizer,
                    abstractive_summarizer=abstractive_summarizer
                )
                logger.info("Summarizer initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize summarizer: {e}")
                self.summarizer = None
                self.fallback_mode = True
            
    def preprocess_text(self, 
                       text: str,
                       custom_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Preprocess a single text.
        
        Args:
            text (str): Input text
            custom_config (Dict[str, Any]): Custom preprocessing configuration
            
        Returns:
            Dict[str, Any]: Preprocessing results
        """
        preprocessing_config = self.config.get('preprocessing', {})
        if custom_config:
            preprocessing_config.update(custom_config)
            
        return self.preprocessor.preprocess_text(text, **preprocessing_config)
        
    def analyze_sentiment(self, 
                         texts: Union[str, List[str]],
                         preprocess: bool = True,
                         return_probabilities: bool = False,
                         batch_size: int = 32) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Analyze sentiment of text(s).
        
        Args:
            texts (Union[str, List[str]]): Input text(s)
            preprocess (bool): Whether to preprocess texts
            return_probabilities (bool): Whether to return probabilities
            batch_size (int): Batch size for processing
            
        Returns:
            Union[Dict[str, Any], List[Dict[str, Any]]]: Sentiment analysis results
        """
        start_time = time.time()
        
        # Initialize analyzer
        self._initialize_sentiment_analyzer()
        
        # Handle single text
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
            
        # Fallback if analyzer failed to initialize
        if self.sentiment_analyzer is None:
            logger.warning("Using fallback sentiment analysis")
            results = []
            for text in texts:
                result = self._fallback_sentiment_analysis(text)
                results.append(result)
            return results[0] if single_input else results
            
        # Preprocess if requested
        processed_texts = []
        preprocessing_results = []
        
        for text in texts:
            if preprocess:
                preprocess_result = self.preprocess_text(text)
                processed_texts.append(preprocess_result['processed'])
                preprocessing_results.append(preprocess_result)
            else:
                processed_texts.append(text)
                preprocessing_results.append({'processed': text, 'is_valid': True})
                
        # Filter valid texts
        valid_texts = []
        valid_indices = []
        
        for i, (processed_text, preprocess_result) in enumerate(zip(processed_texts, preprocessing_results)):
            if preprocess_result.get('is_valid', True) and processed_text.strip():
                valid_texts.append(processed_text)
                valid_indices.append(i)
                
        # Analyze sentiment
        results = []
        if valid_texts:
            predictions = self.sentiment_analyzer.predict(
                valid_texts, 
                return_probabilities=return_probabilities,
                batch_size=batch_size
            )
            
            # Map results back to original indices
            valid_idx = 0
            for i in range(len(texts)):
                if i in valid_indices:
                    result = predictions[valid_idx]
                    result['original_text'] = texts[i]
                    if preprocess:
                        result['processed_text'] = processed_texts[i]
                        result['preprocessing'] = preprocessing_results[i]
                    results.append(result)
                    valid_idx += 1
                else:
                    # Invalid text
                    results.append({
                        'label': 'unknown',
                        'confidence': 0.0,
                        'original_text': texts[i],
                        'error': 'Invalid text after preprocessing'
                    })
        else:
            # No valid texts
            for text in texts:
                results.append({
                    'label': 'unknown',
                    'confidence': 0.0,
                    'original_text': text,
                    'error': 'No valid text to analyze'
                })
                
        # Update stats
        processing_time = time.time() - start_time
        self._update_stats('sentiment', len(texts), processing_time)
        
        return results[0] if single_input else results
        
    def detect_emotions(self, 
                       texts: Union[str, List[str]],
                       preprocess: bool = True,
                       top_k: int = 3,
                       threshold: float = 0.1,
                       batch_size: int = 32) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Detect emotions in text(s).
        
        Args:
            texts (Union[str, List[str]]): Input text(s)
            preprocess (bool): Whether to preprocess texts
            top_k (int): Number of top emotions to return
            threshold (float): Minimum confidence threshold
            batch_size (int): Batch size for processing
            
        Returns:
            Union[Dict[str, Any], List[Dict[str, Any]]]: Emotion detection results
        """
        start_time = time.time()
        
        # Initialize detector
        self._initialize_emotion_detector()
        
        # Handle single text
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
            
        # Preprocess if requested
        processed_texts = []
        preprocessing_results = []
        
        for text in texts:
            if preprocess:
                preprocess_result = self.preprocess_text(text)
                processed_texts.append(preprocess_result['processed'])
                preprocessing_results.append(preprocess_result)
            else:
                processed_texts.append(text)
                preprocessing_results.append({'processed': text, 'is_valid': True})
                
        # Filter valid texts
        valid_texts = []
        valid_indices = []
        
        for i, (processed_text, preprocess_result) in enumerate(zip(processed_texts, preprocessing_results)):
            if preprocess_result.get('is_valid', True) and processed_text.strip():
                valid_texts.append(processed_text)
                valid_indices.append(i)
                
        # Detect emotions
        results = []
        if valid_texts:
            predictions = self.emotion_detector.predict(
                valid_texts,
                top_k=top_k,
                threshold=threshold,
                batch_size=batch_size
            )
            
            # Map results back to original indices
            valid_idx = 0
            for i in range(len(texts)):
                if i in valid_indices:
                    result = predictions[valid_idx]
                    result['original_text'] = texts[i]
                    if preprocess:
                        result['processed_text'] = processed_texts[i]
                        result['preprocessing'] = preprocessing_results[i]
                    results.append(result)
                    valid_idx += 1
                else:
                    # Invalid text
                    results.append({
                        'primary_emotion': 'unknown',
                        'primary_confidence': 0.0,
                        'all_emotions': [],
                        'original_text': texts[i],
                        'error': 'Invalid text after preprocessing'
                    })
        else:
            # No valid texts
            for text in texts:
                results.append({
                    'primary_emotion': 'unknown',
                    'primary_confidence': 0.0,
                    'all_emotions': [],
                    'original_text': text,
                    'error': 'No valid text to analyze'
                })
                
        # Update stats
        processing_time = time.time() - start_time
        self._update_stats('emotion', len(texts), processing_time)
        
        return results[0] if single_input else results
        
    def generate_summary(self, 
                        texts: Union[str, List[str]],
                        method: str = "extractive",
                        num_sentences: int = 3,
                        max_length: int = 128,
                        min_length: int = 30,
                        preprocess: bool = True) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Generate summaries for text(s).
        
        Args:
            texts (Union[str, List[str]]): Input text(s)
            method (str): Summarization method ('extractive', 'abstractive', 'hybrid')
            num_sentences (int): Number of sentences for extractive summarization
            max_length (int): Maximum length for abstractive summarization
            min_length (int): Minimum length for abstractive summarization
            preprocess (bool): Whether to preprocess texts
            
        Returns:
            Union[Dict[str, Any], List[Dict[str, Any]]]: Summarization results
        """
        start_time = time.time()
        
        # Initialize summarizer
        self._initialize_summarizer()
        
        # Handle single text
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
            
        # Process texts
        results = []
        for text in texts:
            # Preprocess if requested
            if preprocess:
                preprocess_result = self.preprocess_text(text)
                processed_text = preprocess_result['processed']
            else:
                processed_text = text
                preprocess_result = {'processed': text, 'is_valid': True}
                
            # Generate summary
            if preprocess_result.get('is_valid', True) and processed_text.strip():
                try:
                    if method == "extractive":
                        summary_result = self.summarizer.extractive.summarize(
                            processed_text, num_sentences=num_sentences
                        )
                    elif method == "abstractive":
                        summary_result = self.summarizer.abstractive.summarize(
                            processed_text, max_length=max_length, min_length=min_length
                        )
                    elif method == "hybrid":
                        summary_result = self.summarizer.summarize(
                            processed_text, method="hybrid",
                            extractive_sentences=num_sentences,
                            max_length=max_length, min_length=min_length
                        )
                    else:
                        raise ValueError(f"Unknown summarization method: {method}")
                        
                    summary_result['original_text'] = text
                    summary_result['method'] = method
                    if preprocess:
                        summary_result['processed_text'] = processed_text
                        summary_result['preprocessing'] = preprocess_result
                        
                    results.append(summary_result)
                    
                except Exception as e:
                    logger.error(f"Error generating summary: {e}")
                    results.append({
                        'summary': '',
                        'original_text': text,
                        'method': method,
                        'error': str(e)
                    })
            else:
                # Invalid text
                results.append({
                    'summary': '',
                    'original_text': text,
                    'method': method,
                    'error': 'Invalid text after preprocessing'
                })
                
        # Update stats
        processing_time = time.time() - start_time
        self._update_stats('summarization', len(texts), processing_time)
        
        return results[0] if single_input else results
        
    def analyze_comprehensive(self, 
                            texts: Union[str, List[str]],
                            include_sentiment: bool = True,
                            include_emotion: bool = True,
                            include_summary: bool = True,
                            summary_method: str = "extractive",
                            preprocess: bool = True,
                            **kwargs) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Perform comprehensive analysis including sentiment, emotion, and summarization.
        
        Args:
            texts (Union[str, List[str]]): Input text(s)
            include_sentiment (bool): Whether to include sentiment analysis
            include_emotion (bool): Whether to include emotion detection
            include_summary (bool): Whether to include summarization
            summary_method (str): Summarization method
            preprocess (bool): Whether to preprocess texts
            **kwargs: Additional arguments for specific analyses
            
        Returns:
            Union[Dict[str, Any], List[Dict[str, Any]]]: Comprehensive analysis results
        """
        start_time = time.time()
        
        # Handle single text
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
            
        results = []
        
        for text in texts:
            result = {
                'original_text': text,
                'timestamp': datetime.now().isoformat()
            }
            
            # Sentiment analysis
            if include_sentiment:
                try:
                    sentiment_result = self.analyze_sentiment(
                        text, preprocess=preprocess, **kwargs.get('sentiment_kwargs', {})
                    )
                    result['sentiment'] = sentiment_result
                except Exception as e:
                    logger.error(f"Error in sentiment analysis: {e}")
                    result['sentiment'] = {'error': str(e)}
                    
            # Emotion detection
            if include_emotion:
                try:
                    emotion_result = self.detect_emotions(
                        text, preprocess=preprocess, **kwargs.get('emotion_kwargs', {})
                    )
                    result['emotion'] = emotion_result
                except Exception as e:
                    logger.error(f"Error in emotion detection: {e}")
                    result['emotion'] = {'error': str(e)}
                    
            # Summarization
            if include_summary:
                try:
                    summary_result = self.generate_summary(
                        text, method=summary_method, preprocess=preprocess,
                        **kwargs.get('summary_kwargs', {})
                    )
                    result['summary'] = summary_result
                except Exception as e:
                    logger.error(f"Error in summarization: {e}")
                    result['summary'] = {'error': str(e)}
                    
            results.append(result)
            
        # Update stats
        processing_time = time.time() - start_time
        self._update_stats('comprehensive', len(texts), processing_time)
        
        return results[0] if single_input else results
        
    def process_dataframe(self, 
                         df: pd.DataFrame,
                         text_column: str,
                         include_sentiment: bool = True,
                         include_emotion: bool = True,
                         include_summary: bool = True,
                         batch_size: int = 32,
                         **kwargs) -> pd.DataFrame:
        """
        Process a pandas DataFrame with comprehensive analysis.
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Column containing texts
            include_sentiment (bool): Whether to include sentiment analysis
            include_emotion (bool): Whether to include emotion detection
            include_summary (bool): Whether to include summarization
            batch_size (int): Batch size for processing
            **kwargs: Additional arguments
            
        Returns:
            pd.DataFrame: Dataframe with analysis results
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in dataframe")
            
        df_copy = df.copy()
        
        # Get texts
        texts = df_copy[text_column].fillna('').tolist()
        
        logger.info(f"Processing {len(texts)} texts from dataframe")
        
        # Sentiment analysis
        if include_sentiment:
            logger.info("Adding sentiment analysis")
            sentiment_results = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_results = self.analyze_sentiment(
                    batch_texts, **kwargs.get('sentiment_kwargs', {})
                )
                sentiment_results.extend(batch_results)
                
            # Add to dataframe
            df_copy['sentiment_label'] = [r.get('label', 'unknown') for r in sentiment_results]
            df_copy['sentiment_confidence'] = [r.get('confidence', 0.0) for r in sentiment_results]
            
        # Emotion detection
        if include_emotion:
            logger.info("Adding emotion detection")
            emotion_results = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_results = self.detect_emotions(
                    batch_texts, **kwargs.get('emotion_kwargs', {})
                )
                emotion_results.extend(batch_results)
                
            # Add to dataframe
            df_copy['primary_emotion'] = [r.get('primary_emotion', 'unknown') for r in emotion_results]
            df_copy['emotion_confidence'] = [r.get('primary_confidence', 0.0) for r in emotion_results]
            
        # Summarization
        if include_summary:
            logger.info("Adding summarization")
            summary_results = []
            for text in texts:
                summary_result = self.generate_summary(
                    text, **kwargs.get('summary_kwargs', {})
                )
                summary_results.append(summary_result)
                
            # Add to dataframe
            df_copy['summary'] = [r.get('summary', '') for r in summary_results]
            df_copy['compression_ratio'] = [r.get('compression_ratio', 0.0) for r in summary_results]
            
        logger.info(f"DataFrame processing completed")
        return df_copy
        
    def _update_stats(self, task_type: str, count: int, processing_time: float):
        """Update inference statistics."""
        self.inference_stats['total_requests'] += count
        self.inference_stats[f'{task_type}_requests'] += count
        
        # Update average processing time
        total_time = (
            self.inference_stats['avg_processing_time'] * 
            (self.inference_stats['total_requests'] - count)
        ) + processing_time
        self.inference_stats['avg_processing_time'] = total_time / self.inference_stats['total_requests']
        
        self.inference_stats['last_request_time'] = datetime.now().isoformat()
        
    def _fallback_sentiment_analysis(self, text: str) -> Dict[str, Any]:
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
        
        if positive_count > negative_count:
            label = "positive"
            confidence = min(0.7 + (positive_count * 0.1), 0.95)
        elif negative_count > positive_count:
            label = "negative"
            confidence = min(0.7 + (negative_count * 0.1), 0.95)
        else:
            label = "neutral"
            confidence = 0.6
            
        return {
            'label': label,
            'confidence': confidence,
            'original_text': text,
            'method': 'fallback_heuristic'
        }
        
    def _fallback_emotion_analysis(self, text: str) -> Dict[str, Any]:
        """Fallback emotion analysis using simple heuristics."""
        text_lower = text.lower()
        
        emotion_words = {
            'joy': ['खुश', 'प्रसन्न', 'happy', 'joy', 'excited', 'great'],
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
            
        return {
            'primary_emotion': primary_emotion,
            'emotion_scores': emotion_scores,
            'original_text': text,
            'method': 'fallback_heuristic'
        }
        
    def _fallback_summarization(self, text: str, max_sentences: int = 2) -> Dict[str, Any]:
        """Fallback summarization using simple sentence extraction."""
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= max_sentences:
            summary = text
        else:
            # Take first and last sentences as summary
            summary = '. '.join(sentences[:max_sentences]) + '.'
            
        return {
            'summary': summary,
            'original_length': len(text),
            'summary_length': len(summary),
            'compression_ratio': len(summary) / len(text) if len(text) > 0 else 0,
            'method': 'fallback_extraction'
        }
        
    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        stats = self.inference_stats.copy()
        stats['fallback_mode'] = self.fallback_mode
        return stats
        
    def reset_stats(self):
        """Reset inference statistics."""
        self.inference_stats = {
            'total_requests': 0,
            'sentiment_requests': 0,
            'emotion_requests': 0,
            'summarization_requests': 0,
            'avg_processing_time': 0,
            'last_request_time': None
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    pipeline = InferencePipeline()
    
    sample_texts = [
        "यह सेवा बहुत अच्छी है! डॉक्टर ने बहुत अच्छी सलाह दी।",
        "The consultation was helpful but the wait time was too long.",
        "I am very disappointed with the service. The doctor was rude."
    ]
    
    print("=== Testing Inference Pipeline ===")
    
    try:
        # Test sentiment analysis
        print("\n--- Sentiment Analysis ---")
        sentiment_results = pipeline.analyze_sentiment(sample_texts[0], return_probabilities=True)
        print(f"Text: {sentiment_results['original_text']}")
        print(f"Sentiment: {sentiment_results['label']} (confidence: {sentiment_results['confidence']:.3f})")
        
        # Test comprehensive analysis
        print("\n--- Comprehensive Analysis ---")
        comprehensive_result = pipeline.analyze_comprehensive(
            sample_texts[0],
            include_summary=True,
            summary_method="extractive"
        )
        
        print(f"Sentiment: {comprehensive_result['sentiment']['label']}")
        print(f"Primary Emotion: {comprehensive_result['emotion']['primary_emotion']}")
        print(f"Summary: {comprehensive_result['summary']['summary']}")
        
        # Show stats
        print(f"\n--- Pipeline Stats ---")
        stats = pipeline.get_stats()
        print(f"Total requests: {stats['total_requests']}")
        print(f"Average processing time: {stats['avg_processing_time']:.3f}s")
        
    except Exception as e:
        print(f"Pipeline test failed: {e}")
        print("Note: Make sure required models are available.")