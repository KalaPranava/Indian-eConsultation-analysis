"""
Text summarization models for Indian e-consultation comments.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
import re
from collections import Counter
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, 
    pipeline, TrainingArguments, Trainer
)
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk
from pathlib import Path

logger = logging.getLogger(__name__)


class TextRankSummarizer:
    """
    Extractive text summarization using TextRank algorithm.
    """
    
    def __init__(self, 
                 language: str = 'english',
                 stopwords_lang: str = 'english',
                 damping: float = 0.85,
                 min_diff: float = 1e-5,
                 steps: int = 10):
        """
        Initialize TextRank Summarizer.
        
        Args:
            language (str): Language for sentence tokenization
            stopwords_lang (str): Language for stopwords
            damping (float): Damping parameter for PageRank
            min_diff (float): Minimum difference for convergence
            steps (int): Maximum number of iterations
        """
        self.language = language
        self.damping = damping
        self.min_diff = min_diff
        self.steps = steps
        
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except Exception as e:
            logger.warning(f"Could not download NLTK data: {e}")
            
        # Load stopwords
        try:
            self.stop_words = set(stopwords.words(stopwords_lang))
        except LookupError:
            logger.warning(f"Stopwords for {stopwords_lang} not found, using empty set")
            self.stop_words = set()
            
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for summarization.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'[@#]\w+', '', text)
        
        return text
        
    def _sentence_similarity(self, sent1: str, sent2: str) -> float:
        """
        Calculate similarity between two sentences.
        
        Args:
            sent1 (str): First sentence
            sent2 (str): Second sentence
            
        Returns:
            float: Similarity score
        """
        # Tokenize and clean
        words1 = [word.lower() for word in word_tokenize(sent1) 
                 if word.lower() not in self.stop_words and word.isalnum()]
        words2 = [word.lower() for word in word_tokenize(sent2) 
                 if word.lower() not in self.stop_words and word.isalnum()]
        
        if not words1 or not words2:
            return 0.0
            
        # Calculate Jaccard similarity
        intersection = set(words1) & set(words2)
        union = set(words1) | set(words2)
        
        if not union:
            return 0.0
            
        return len(intersection) / len(union)
        
    def _build_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """
        Build similarity matrix for sentences.
        
        Args:
            sentences (List[str]): List of sentences
            
        Returns:
            np.ndarray: Similarity matrix
        """
        n = len(sentences)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    similarity_matrix[i][j] = self._sentence_similarity(sentences[i], sentences[j])
                    
        return similarity_matrix
        
    def _apply_pagerank(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """
        Apply PageRank algorithm to similarity matrix.
        
        Args:
            similarity_matrix (np.ndarray): Sentence similarity matrix
            
        Returns:
            np.ndarray: PageRank scores
        """
        n = similarity_matrix.shape[0]
        
        # Normalize similarity matrix
        for i in range(n):
            row_sum = similarity_matrix[i].sum()
            if row_sum > 0:
                similarity_matrix[i] = similarity_matrix[i] / row_sum
                
        # Initialize scores
        scores = np.ones(n) / n
        
        # Iterate
        for _ in range(self.steps):
            new_scores = (1 - self.damping) / n + self.damping * np.dot(similarity_matrix.T, scores)
            
            if np.sum(np.abs(new_scores - scores)) < self.min_diff:
                break
                
            scores = new_scores
            
        return scores
        
    def summarize(self, 
                 text: str, 
                 num_sentences: int = 3,
                 min_sentence_length: int = 10) -> Dict[str, Any]:
        """
        Generate extractive summary using TextRank.
        
        Args:
            text (str): Input text
            num_sentences (int): Number of sentences in summary
            min_sentence_length (int): Minimum sentence length
            
        Returns:
            Dict[str, Any]: Summary results
        """
        if not text or len(text.strip()) < min_sentence_length:
            return {
                'summary': '',
                'sentences': [],
                'scores': [],
                'original_length': len(text),
                'summary_length': 0,
                'compression_ratio': 0.0
            }
            
        # Preprocess text
        preprocessed_text = self._preprocess_text(text)
        
        # Split into sentences
        sentences = sent_tokenize(preprocessed_text)
        
        # Filter short sentences
        sentences = [sent for sent in sentences if len(sent.strip()) >= min_sentence_length]
        
        if len(sentences) <= num_sentences:
            summary = ' '.join(sentences)
            return {
                'summary': summary,
                'sentences': sentences,
                'scores': [1.0] * len(sentences),
                'original_length': len(text),
                'summary_length': len(summary),
                'compression_ratio': len(summary) / len(text) if len(text) > 0 else 0.0
            }
            
        # Build similarity matrix
        similarity_matrix = self._build_similarity_matrix(sentences)
        
        # Apply PageRank
        scores = self._apply_pagerank(similarity_matrix)
        
        # Get top sentences
        top_indices = scores.argsort()[-num_sentences:][::-1]
        top_indices = sorted(top_indices)  # Maintain original order
        
        summary_sentences = [sentences[i] for i in top_indices]
        summary = ' '.join(summary_sentences)
        
        return {
            'summary': summary,
            'sentences': summary_sentences,
            'scores': scores[top_indices].tolist(),
            'original_length': len(text),
            'summary_length': len(summary),
            'compression_ratio': len(summary) / len(text) if len(text) > 0 else 0.0
        }
        
    def summarize_multiple(self, 
                          texts: List[str], 
                          num_sentences: int = 3) -> List[Dict[str, Any]]:
        """
        Summarize multiple texts.
        
        Args:
            texts (List[str]): List of input texts
            num_sentences (int): Number of sentences per summary
            
        Returns:
            List[Dict[str, Any]]: List of summary results
        """
        return [self.summarize(text, num_sentences) for text in texts]


class AbstractiveSummarizer:
    """
    Abstractive text summarization using transformer models.
    """
    
    def __init__(self, 
                 model_name: str = "t5-small",
                 cache_dir: str = "models/summarization",
                 device: str = None):
        """
        Initialize abstractive summarizer.
        
        Args:
            model_name (str): Name of the pre-trained model
            cache_dir (str): Directory to cache models
            device (str): Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Initialize model components
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.is_trained = False
        
    def load_model(self, model_path: str = None):
        """
        Load a pre-trained or fine-tuned model.
        
        Args:
            model_path (str): Path to the model (optional)
        """
        try:
            if model_path:
                logger.info(f"Loading model from {model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
                self.is_trained = True
            else:
                logger.info(f"Loading pre-trained model: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir
                )
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir
                )
                
            self.model.to(self.device)
            
            # Create pipeline
            self.pipeline = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
            
    def summarize(self, 
                 text: str,
                 max_length: int = 128,
                 min_length: int = 30,
                 do_sample: bool = False,
                 num_beams: int = 4,
                 temperature: float = 1.0) -> Dict[str, Any]:
        """
        Generate abstractive summary.
        
        Args:
            text (str): Input text
            max_length (int): Maximum summary length
            min_length (int): Minimum summary length
            do_sample (bool): Whether to use sampling
            num_beams (int): Number of beams for beam search
            temperature (float): Temperature for sampling
            
        Returns:
            Dict[str, Any]: Summary results
        """
        if not self.pipeline:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        if not text or len(text.strip()) < min_length:
            return {
                'summary': '',
                'original_length': len(text),
                'summary_length': 0,
                'compression_ratio': 0.0
            }
            
        try:
            # Prepare input for T5 models
            if "t5" in self.model_name.lower():
                input_text = f"summarize: {text}"
            else:
                input_text = text
                
            # Generate summary
            result = self.pipeline(
                input_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                num_beams=num_beams,
                temperature=temperature,
                truncation=True
            )
            
            summary = result[0]['summary_text']
            
            return {
                'summary': summary,
                'original_length': len(text),
                'summary_length': len(summary),
                'compression_ratio': len(summary) / len(text) if len(text) > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error during summarization: {str(e)}")
            return {
                'summary': '',
                'original_length': len(text),
                'summary_length': 0,
                'compression_ratio': 0.0,
                'error': str(e)
            }
            
    def summarize_multiple(self, 
                          texts: List[str],
                          batch_size: int = 8,
                          **kwargs) -> List[Dict[str, Any]]:
        """
        Summarize multiple texts in batches.
        
        Args:
            texts (List[str]): List of input texts
            batch_size (int): Batch size for processing
            **kwargs: Additional arguments for summarize()
            
        Returns:
            List[Dict[str, Any]]: List of summary results
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = [self.summarize(text, **kwargs) for text in batch]
            results.extend(batch_results)
            
        return results
        
    def save_model(self, save_path: str):
        """
        Save the trained model.
        
        Args:
            save_path (str): Path to save the model
        """
        if not self.model or not self.tokenizer:
            raise ValueError("No model to save. Train or load a model first.")
            
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        logger.info(f"Model saved to {save_path}")


class HybridSummarizer:
    """
    Hybrid summarizer combining extractive and abstractive approaches.
    """
    
    def __init__(self, 
                 extractive_summarizer: TextRankSummarizer = None,
                 abstractive_summarizer: AbstractiveSummarizer = None):
        """
        Initialize hybrid summarizer.
        
        Args:
            extractive_summarizer: Extractive summarizer instance
            abstractive_summarizer: Abstractive summarizer instance
        """
        self.extractive = extractive_summarizer or TextRankSummarizer()
        self.abstractive = abstractive_summarizer or AbstractiveSummarizer()
        
    def summarize(self, 
                 text: str,
                 method: str = "extractive",
                 **kwargs) -> Dict[str, Any]:
        """
        Generate summary using specified method.
        
        Args:
            text (str): Input text
            method (str): Method to use ('extractive', 'abstractive', or 'hybrid')
            **kwargs: Additional arguments for summarization
            
        Returns:
            Dict[str, Any]: Summary results
        """
        if method == "extractive":
            return self.extractive.summarize(text, **kwargs)
        elif method == "abstractive":
            if not self.abstractive.pipeline:
                self.abstractive.load_model()
            return self.abstractive.summarize(text, **kwargs)
        elif method == "hybrid":
            return self._hybrid_summarize(text, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
            
    def _hybrid_summarize(self, 
                         text: str,
                         extractive_sentences: int = 5,
                         **abstractive_kwargs) -> Dict[str, Any]:
        """
        Generate hybrid summary (extractive + abstractive).
        
        Args:
            text (str): Input text
            extractive_sentences (int): Number of sentences for extractive step
            **abstractive_kwargs: Arguments for abstractive summarization
            
        Returns:
            Dict[str, Any]: Summary results
        """
        # Step 1: Extractive summarization to reduce text length
        extractive_result = self.extractive.summarize(text, num_sentences=extractive_sentences)
        
        if not extractive_result['summary']:
            return extractive_result
            
        # Step 2: Abstractive summarization on extracted sentences
        if not self.abstractive.pipeline:
            self.abstractive.load_model()
            
        abstractive_result = self.abstractive.summarize(
            extractive_result['summary'], 
            **abstractive_kwargs
        )
        
        # Combine results
        return {
            'summary': abstractive_result['summary'],
            'extractive_summary': extractive_result['summary'],
            'extractive_sentences': extractive_result['sentences'],
            'original_length': len(text),
            'summary_length': len(abstractive_result['summary']),
            'compression_ratio': len(abstractive_result['summary']) / len(text) if len(text) > 0 else 0.0,
            'method': 'hybrid'
        }
        
    def summarize_dataframe(self, 
                           df: pd.DataFrame,
                           text_column: str,
                           method: str = "extractive",
                           batch_size: int = 32,
                           **kwargs) -> pd.DataFrame:
        """
        Summarize texts in a DataFrame.
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Column containing texts
            method (str): Summarization method
            batch_size (int): Batch size for processing
            **kwargs: Additional arguments for summarization
            
        Returns:
            pd.DataFrame: Dataframe with summaries
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in dataframe")
            
        df_copy = df.copy()
        
        # Get valid texts
        valid_mask = df_copy[text_column].notna() & (df_copy[text_column] != '')
        valid_texts = df_copy.loc[valid_mask, text_column].tolist()
        
        if not valid_texts:
            logger.warning("No valid texts found for summarization")
            df_copy['summary'] = ''
            df_copy['compression_ratio'] = 0.0
            return df_copy
            
        # Summarize
        logger.info(f"Summarizing {len(valid_texts)} texts using {method} method")
        
        if method == "extractive":
            summaries = [self.extractive.summarize(text, **kwargs) for text in valid_texts]
        elif method == "abstractive":
            if not self.abstractive.pipeline:
                self.abstractive.load_model()
            summaries = self.abstractive.summarize_multiple(valid_texts, batch_size, **kwargs)
        else:  # hybrid
            summaries = [self._hybrid_summarize(text, **kwargs) for text in valid_texts]
            
        # Add summaries to dataframe
        df_copy['summary'] = ''
        df_copy['compression_ratio'] = 0.0
        
        # Fill summaries for valid rows
        summary_idx = 0
        for idx, is_valid in valid_mask.items():
            if is_valid:
                summary_result = summaries[summary_idx]
                df_copy.loc[idx, 'summary'] = summary_result['summary']
                df_copy.loc[idx, 'compression_ratio'] = summary_result['compression_ratio']
                summary_idx += 1
                
        return df_copy


def create_sample_summarization_data():
    """Create sample data for summarization testing."""
    sample_texts = [
        """
        मेरा ऑनलाइन consultation का experience बहुत अच्छा रहा। डॉक्टर ने बहुत patience से मेरी सारी problems को सुना। 
        उन्होंने detailed में मेरी condition explain की और proper medication suggest की। Video call की quality भी अच्छी थी। 
        Overall, यह service बहुत helpful है, especially उन लोगों के लिए जो hospital नहीं जा सकते।
        """,
        """
        The online consultation platform is very user-friendly. I was able to book an appointment easily and the doctor 
        joined the call on time. The doctor was very professional and knowledgeable. They listened to all my concerns 
        carefully and provided detailed explanations. The prescription was also sent digitally which was very convenient. 
        I would definitely recommend this service to others.
        """,
        """
        Initial में मुझे थोड़ी technical difficulties face करनी पड़ीं platform use करने में। But customer support team ने 
        quickly help की। Doctor consultation के time properly सब कुछ work कर रहा था। Doctor ने अच्छी advice दी लेकिन 
        follow-up के लिए physical visit recommend किया। Overall decent experience रहा।
        """
    ]
    
    return sample_texts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    sample_texts = create_sample_summarization_data()
    
    # Test TextRank summarizer
    textrank = TextRankSummarizer()
    
    print("=== TextRank Summarization ===")
    for i, text in enumerate(sample_texts, 1):
        result = textrank.summarize(text, num_sentences=2)
        print(f"\nText {i}:")
        print(f"Original ({result['original_length']} chars): {text.strip()[:100]}...")
        print(f"Summary ({result['summary_length']} chars): {result['summary']}")
        print(f"Compression ratio: {result['compression_ratio']:.2f}")
        
    # Test abstractive summarizer (requires model loading)
    print("\n=== Abstractive Summarization ===")
    try:
        abstractive = AbstractiveSummarizer(model_name="t5-small")
        abstractive.load_model()
        
        for i, text in enumerate(sample_texts[:1], 1):  # Test with first text only
            result = abstractive.summarize(text, max_length=50, min_length=20)
            print(f"\nText {i}:")
            print(f"Original ({result['original_length']} chars): {text.strip()[:100]}...")
            print(f"Summary ({result['summary_length']} chars): {result['summary']}")
            print(f"Compression ratio: {result['compression_ratio']:.2f}")
            
    except Exception as e:
        print(f"Abstractive summarization test failed: {e}")
        
    print("\n=== Hybrid Summarization ===")
    try:
        hybrid = HybridSummarizer()
        
        for i, text in enumerate(sample_texts[:1], 1):  # Test with first text only
            result = hybrid.summarize(text, method="hybrid", max_length=50)
            print(f"\nText {i}:")
            print(f"Original ({result['original_length']} chars): {text.strip()[:100]}...")
            print(f"Extractive: {result['extractive_summary']}")
            print(f"Final Summary: {result['summary']}")
            print(f"Compression ratio: {result['compression_ratio']:.2f}")
            
    except Exception as e:
        print(f"Hybrid summarization test failed: {e}")