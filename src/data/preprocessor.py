"""
Text preprocessing utilities for Indian e-consultation comments.
"""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

# Optional imports with graceful degradation
try:  # langdetect is recommended but optional
    from langdetect import detect, DetectorFactory
    from langdetect.lang_detect_exception import LangDetectException
    DetectorFactory.seed = 0
    _LANGDETECT_AVAILABLE = True
except ImportError:  # Fallback if not installed
    detect = None  # type: ignore
    class LangDetectException(Exception):  # type: ignore
        pass
    _LANGDETECT_AVAILABLE = False

try:  # spaCy optional
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    spacy = None  # type: ignore
    _SPACY_AVAILABLE = False

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from bs4 import BeautifulSoup
import unicodedata

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    A comprehensive text preprocessor for Indian e-consultation comments.
    Handles Hindi, English, and code-mixed content.
    """
    
    def __init__(self, download_nltk_data: bool = True):
        """
        Initialize the text preprocessor.
        
        Args:
            download_nltk_data (bool): Whether to download required NLTK data
        """
        self.hindi_stopwords = set([
            'और', 'है', 'हैं', 'था', 'थे', 'को', 'की', 'के', 'से', 'में', 'पर', 'यह', 'वह',
            'एक', 'दो', 'तीन', 'कि', 'जो', 'तो', 'ही', 'भी', 'नहीं', 'अभी', 'यहाँ', 'वहाँ',
            'कब', 'कैसे', 'क्यों', 'कहाँ', 'जब', 'तब', 'अगर', 'लेकिन', 'या', 'बहुत', 'सभी'
        ])
        
        # Download required NLTK data
        if download_nltk_data:
            self._download_nltk_requirements()
            
        # Initialize English stopwords
        try:
            self.english_stopwords = set(stopwords.words('english'))
        except LookupError:
            logger.warning("English stopwords not found, using empty set")
            self.english_stopwords = set()
            
        # Load spaCy model if available
        self.nlp = None
        if _SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy English model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
        else:
            logger.info("spaCy not installed; lemmatization / POS features disabled.")
            
    def _download_nltk_requirements(self):
        """Download required NLTK data."""
        nltk_downloads = [
            'punkt', 'stopwords', 'vader_lexicon', 
            'averaged_perceptron_tagger', 'wordnet'
        ]
        
        for item in nltk_downloads:
            try:
                nltk.download(item, quiet=True)
            except Exception as e:
                logger.warning(f"Could not download NLTK data '{item}': {e}")
                
    def detect_language(self, text: str) -> str:
        """
        Detect the primary language of the text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Language code ('hi', 'en', or 'mixed')
        """
        if not text or len(text.strip()) < 3:
            return 'unknown'
            
        # Character heuristic first (fast & works even without langdetect)
        hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = hindi_chars + english_chars

        heuristic_lang = 'unknown'
        if total_chars > 0:
            hindi_ratio = hindi_chars / total_chars
            english_ratio = english_chars / total_chars
            if hindi_ratio > 0.2 and english_ratio > 0.2:
                heuristic_lang = 'mixed'
            elif hindi_ratio > 0.6:
                heuristic_lang = 'hi'
            elif english_ratio > 0.6:
                heuristic_lang = 'en'
            else:
                heuristic_lang = 'mixed'

        if not _LANGDETECT_AVAILABLE:
            return heuristic_lang

        # If langdetect available, refine result
        try:
            clean_text = re.sub(r'[^\w\s]', ' ', text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            if len(clean_text) < 3:
                return heuristic_lang
            detected_lang = detect(clean_text)
            if detected_lang in ['hi', 'en']:
                # Prefer explicit detection over heuristic unless heuristic says mixed
                if heuristic_lang != 'mixed':
                    return detected_lang
            return heuristic_lang
        except LangDetectException:
            return heuristic_lang
                
    def clean_html(self, text: str) -> str:
        """
        Remove HTML tags and entities from text.
        
        Args:
            text (str): Input text with HTML
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
            
        # Use BeautifulSoup to remove HTML tags
        soup = BeautifulSoup(text, 'html.parser')
        cleaned = soup.get_text()
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
        
    def normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode characters.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Normalized text
        """
        if not text:
            return ""
            
        # Normalize Unicode characters
        normalized = unicodedata.normalize('NFKC', text)
        
        return normalized
        
    def remove_urls(self, text: str) -> str:
        """
        Remove URLs from text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text without URLs
        """
        if not text:
            return ""
            
        # Pattern for URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        
        # Remove URLs
        cleaned = re.sub(url_pattern, '', text)
        
        # Remove www. patterns
        cleaned = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', cleaned)
        
        # Clean up extra spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
        
    def remove_mentions_hashtags(self, text: str) -> str:
        """
        Remove social media mentions and hashtags.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text without mentions and hashtags
        """
        if not text:
            return ""
            
        # Remove mentions (@username)
        cleaned = re.sub(r'@[A-Za-z0-9_]+', '', text)
        
        # Remove hashtags (#hashtag)
        cleaned = re.sub(r'#[A-Za-z0-9_]+', '', cleaned)
        
        # Clean up extra spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
        
    def remove_extra_whitespace(self, text: str) -> str:
        """
        Remove extra whitespace and normalize spacing.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with normalized spacing
        """
        if not text:
            return ""
            
        # Replace multiple spaces with single space
        cleaned = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        cleaned = cleaned.strip()
        
        return cleaned
        
    def remove_special_characters(self, text: str, keep_devanagari: bool = True) -> str:
        """
        Remove special characters while preserving necessary punctuation.
        
        Args:
            text (str): Input text
            keep_devanagari (bool): Whether to keep Devanagari characters
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
            
        if keep_devanagari:
            # Keep alphanumeric, Devanagari, and basic punctuation
            pattern = r'[^a-zA-Z0-9\u0900-\u097F\s.!?,:;()-]'
        else:
            # Keep only alphanumeric and basic punctuation
            pattern = r'[^a-zA-Z0-9\s.!?,:;()-]'
            
        cleaned = re.sub(pattern, '', text)
        
        # Clean up extra spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
        
    def remove_stopwords(self, text: str, language: str = 'auto') -> str:
        """
        Remove stopwords based on detected or specified language.
        
        Args:
            text (str): Input text
            language (str): Language code ('hi', 'en', 'mixed', or 'auto')
            
        Returns:
            str: Text without stopwords
        """
        if not text:
            return ""
            
        if language == 'auto':
            language = self.detect_language(text)
            
        # Tokenize text
        words = word_tokenize(text.lower())
        
        # Select appropriate stopwords
        if language == 'hi':
            stopwords_set = self.hindi_stopwords
        elif language == 'en':
            stopwords_set = self.english_stopwords
        elif language == 'mixed':
            stopwords_set = self.hindi_stopwords.union(self.english_stopwords)
        else:
            return text  # Return original if language is unknown
            
        # Filter out stopwords
        filtered_words = [word for word in words if word not in stopwords_set]
        
        return ' '.join(filtered_words)
        
    def preprocess_text(self, 
                       text: str, 
                       clean_html: bool = True,
                       normalize_unicode: bool = True,
                       remove_urls: bool = True,
                       remove_mentions: bool = True,
                       remove_special_chars: bool = True,
                       remove_stopwords: bool = False,
                       min_length: int = 3,
                       max_length: int = 1000) -> Dict[str, Any]:
        """
        Comprehensive text preprocessing pipeline.
        
        Args:
            text (str): Input text
            clean_html (bool): Whether to remove HTML tags
            normalize_unicode (bool): Whether to normalize Unicode
            remove_urls (bool): Whether to remove URLs
            remove_mentions (bool): Whether to remove mentions/hashtags
            remove_special_chars (bool): Whether to remove special characters
            remove_stopwords (bool): Whether to remove stopwords
            min_length (int): Minimum text length
            max_length (int): Maximum text length
            
        Returns:
            Dict[str, Any]: Preprocessing results
        """
        if not text or not isinstance(text, str):
            return {
                'original': text,
                'processed': '',
                'language': 'unknown',
                'length': 0,
                'is_valid': False,
                'steps_applied': []
            }
            
        processed = text
        steps_applied = []
        
        # Detect original language
        original_language = self.detect_language(text)
        
        # Apply preprocessing steps
        if clean_html:
            processed = self.clean_html(processed)
            steps_applied.append('clean_html')
            
        if normalize_unicode:
            processed = self.normalize_unicode(processed)
            steps_applied.append('normalize_unicode')
            
        if remove_urls:
            processed = self.remove_urls(processed)
            steps_applied.append('remove_urls')
            
        if remove_mentions:
            processed = self.remove_mentions_hashtags(processed)
            steps_applied.append('remove_mentions')
            
        if remove_special_chars:
            processed = self.remove_special_characters(processed)
            steps_applied.append('remove_special_chars')
            
        # Always remove extra whitespace
        processed = self.remove_extra_whitespace(processed)
        steps_applied.append('remove_extra_whitespace')
        
        if remove_stopwords:
            processed = self.remove_stopwords(processed, original_language)
            steps_applied.append('remove_stopwords')
            
        # Check validity
        processed_length = len(processed)
        is_valid = min_length <= processed_length <= max_length
        
        return {
            'original': text,
            'processed': processed,
            'language': original_language,
            'length': processed_length,
            'is_valid': is_valid,
            'steps_applied': steps_applied
        }
        
    def process_dataframe(self, 
                         df: pd.DataFrame, 
                         text_columns: List[str],
                         **preprocessing_kwargs) -> pd.DataFrame:
        """
        Process text in a pandas DataFrame.
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_columns (List[str]): Columns containing text to process
            **preprocessing_kwargs: Arguments for preprocess_text
            
        Returns:
            pd.DataFrame: Processed dataframe
        """
        df_copy = df.copy()
        
        for column in text_columns:
            if column not in df_copy.columns:
                logger.warning(f"Column '{column}' not found in dataframe")
                continue
                
            logger.info(f"Processing text in column '{column}'")
            
            # Apply preprocessing
            results = df_copy[column].apply(
                lambda x: self.preprocess_text(x, **preprocessing_kwargs) if pd.notna(x) else {
                    'original': x, 'processed': '', 'language': 'unknown', 
                    'length': 0, 'is_valid': False, 'steps_applied': []
                }
            )
            
            # Extract results into separate columns
            df_copy[f'{column}_processed'] = results.apply(lambda x: x['processed'])
            df_copy[f'{column}_language'] = results.apply(lambda x: x['language'])
            df_copy[f'{column}_length'] = results.apply(lambda x: x['length'])
            df_copy[f'{column}_is_valid'] = results.apply(lambda x: x['is_valid'])
            
        return df_copy
        
    def remove_duplicates(self, 
                         df: pd.DataFrame, 
                         text_column: str = 'comment',
                         processed_column: str = None) -> pd.DataFrame:
        """
        Remove duplicate texts from dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Column name for original text
            processed_column (str): Column name for processed text (optional)
            
        Returns:
            pd.DataFrame: Dataframe without duplicates
        """
        if processed_column and processed_column in df.columns:
            # Remove duplicates based on processed text
            df_dedup = df.drop_duplicates(subset=[processed_column])
            logger.info(f"Removed {len(df) - len(df_dedup)} duplicates based on processed text")
        else:
            # Remove duplicates based on original text
            df_dedup = df.drop_duplicates(subset=[text_column])
            logger.info(f"Removed {len(df) - len(df_dedup)} duplicates based on original text")
            
        return df_dedup.reset_index(drop=True)
        
    def get_preprocessing_stats(self, df: pd.DataFrame, text_column: str) -> Dict[str, Any]:
        """
        Get statistics about the preprocessing results.
        
        Args:
            df (pd.DataFrame): Processed dataframe
            text_column (str): Base text column name
            
        Returns:
            Dict[str, Any]: Statistics
        """
        stats = {}
        
        # Basic stats
        stats['total_samples'] = len(df)
        
        # Language distribution
        if f'{text_column}_language' in df.columns:
            lang_counts = df[f'{text_column}_language'].value_counts()
            stats['language_distribution'] = lang_counts.to_dict()
            
        # Validity stats
        if f'{text_column}_is_valid' in df.columns:
            valid_count = df[f'{text_column}_is_valid'].sum()
            stats['valid_samples'] = int(valid_count)
            stats['invalid_samples'] = int(len(df) - valid_count)
            stats['validity_rate'] = float(valid_count / len(df))
            
        # Length stats
        if f'{text_column}_length' in df.columns:
            lengths = df[f'{text_column}_length']
            stats['length_stats'] = {
                'mean': float(lengths.mean()),
                'median': float(lengths.median()),
                'min': int(lengths.min()),
                'max': int(lengths.max()),
                'std': float(lengths.std())
            }
            
        return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    preprocessor = TextPreprocessor()
    
    sample_texts = [
        "यह सेवा बहुत अच्छी है! <br> डॉक्टर ने excellent advice दी।",
        "The consultation was great. डॉक्टर साहब बहुत knowledgeable हैं।",
        "Visit https://example.com for more info @doctor #healthcare",
        "Service     बहुत   अच्छी   है   !!!!"
    ]
    
    for text in sample_texts:
        result = preprocessor.preprocess_text(text)
        print(f"Original: {result['original']}")
        print(f"Processed: {result['processed']}")
        print(f"Language: {result['language']}")
        print(f"Valid: {result['is_valid']}")
        print("-" * 50)