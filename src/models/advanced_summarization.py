"""
Advanced Text Summarization using Transformer Models and Hybrid Approaches
Supports extractive and abstractive summarization for Indian e-consultation texts
"""
import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from transformers import (
    pipeline, AutoTokenizer, AutoModelForSeq2SeqLM,
    T5Tokenizer, T5ForConditionalGeneration,
    BartTokenizer, BartForConditionalGeneration
)
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import re
from collections import Counter
from dataclasses import dataclass
import time

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

logger = logging.getLogger(__name__)

@dataclass
class SummaryResult:
    """Structured result for text summarization"""
    summary: str
    original_length: int
    summary_length: int
    compression_ratio: float
    method_used: str
    processing_time: float
    key_sentences: List[str]
    important_entities: List[str]
    summary_quality_score: float
    reasoning: Optional[str] = None

class AdvancedTextSummarizer:
    """
    Advanced text summarization using multiple transformer models and hybrid approaches
    Optimized for Indian e-consultation and medical texts
    """
    
    def __init__(self):
        self.models = {}
        self.stop_words = self._get_stop_words()
        self.medical_terms = self._load_medical_terms()
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize summarization models"""
        try:
            logger.info("Loading advanced summarization models...")
            
            # T5 model for abstractive summarization
            try:
                self.models['t5_tokenizer'] = T5Tokenizer.from_pretrained('t5-small')
                self.models['t5_model'] = T5ForConditionalGeneration.from_pretrained('t5-small')
                logger.info("✅ T5 model loaded")
            except Exception as e:
                logger.warning(f"T5 model loading failed: {e}")
            
            # BART model for abstractive summarization
            try:
                self.models['bart_pipeline'] = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    max_length=150,
                    min_length=30,
                    do_sample=False
                )
                logger.info("✅ BART model loaded")
            except Exception as e:
                logger.warning(f"BART model loading failed: {e}")
            
            # DistilBART for faster processing
            try:
                self.models['distilbart_pipeline'] = pipeline(
                    "summarization",
                    model="sshleifer/distilbart-cnn-12-6",
                    max_length=100,
                    min_length=20
                )
                logger.info("✅ DistilBART model loaded")
            except Exception as e:
                logger.warning(f"DistilBART model loading failed: {e}")
            
            # TF-IDF vectorizer for extractive summarization
            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=1000,
                ngram_range=(1, 2)
            )
            
            logger.info("✅ Advanced summarization models initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize summarization models: {str(e)}")
            raise
    
    def _get_stop_words(self) -> set:
        """Get combined stop words for Hindi and English"""
        try:
            english_stops = set(stopwords.words('english'))
        except:
            english_stops = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        
        hindi_stops = {
            'है', 'हैं', 'था', 'थे', 'की', 'के', 'का', 'में', 'से', 'को', 'और', 'या', 
            'यह', 'वह', 'एक', 'तो', 'ही', 'भी', 'पर', 'तक', 'जो', 'कि', 'लिए'
        }
        
        return english_stops | hindi_stops
    
    def _load_medical_terms(self) -> Dict[str, List[str]]:
        """Load medical terminology for context-aware summarization"""
        return {
            'symptoms': ['fever', 'pain', 'cough', 'headache', 'nausea', 'fatigue',
                        'बुखार', 'दर्द', 'खांसी', 'सिरदर्द', 'जी मचलना', 'थकान'],
            'treatments': ['medicine', 'prescription', 'therapy', 'surgery', 'treatment',
                          'दवा', 'इलाज', 'चिकित्सा', 'उपचार', 'नुस्खा'],
            'healthcare': ['doctor', 'nurse', 'hospital', 'clinic', 'consultation',
                          'डॉक्टर', 'नर्स', 'अस्पताल', 'क्लिनिक', 'सलाह'],
            'outcomes': ['recovery', 'improvement', 'cure', 'healing', 'better',
                        'ठीक', 'सुधार', 'स्वस्थ', 'बेहतर', 'अच्छा']
        }
    
    def summarize(self, text: str, method: str = "hybrid", 
                 max_length: int = 100, target_sentences: int = 3) -> SummaryResult:
        """
        Generate summary using specified method
        
        Args:
            text: Input text to summarize
            method: 'extractive', 'abstractive', 'hybrid', 'auto'
            max_length: Maximum summary length in characters
            target_sentences: Target number of sentences for extractive methods
        
        Returns:
            SummaryResult with comprehensive summary information
        """
        start_time = time.time()
        original_length = len(text)
        
        try:
            # Choose appropriate method based on text length and content
            if method == "auto":
                method = self._choose_optimal_method(text)
            
            # Generate summary based on method
            if method == "extractive":
                result = self._extractive_summarization(text, target_sentences)
            elif method == "abstractive":
                result = self._abstractive_summarization(text, max_length)
            elif method == "hybrid":
                result = self._hybrid_summarization(text, max_length, target_sentences)
            else:
                # Fallback to TextRank extractive
                result = self._textrank_summarization(text, target_sentences)
            
            # Calculate metrics
            summary_length = len(result['summary'])
            compression_ratio = summary_length / max(original_length, 1)
            
            # Extract additional information
            key_sentences = result.get('key_sentences', [])
            important_entities = self._extract_important_entities(text, result['summary'])
            quality_score = self._calculate_summary_quality(text, result['summary'])
            
            processing_time = time.time() - start_time
            
            return SummaryResult(
                summary=result['summary'],
                original_length=original_length,
                summary_length=summary_length,
                compression_ratio=compression_ratio,
                method_used=result['method'],
                processing_time=processing_time,
                key_sentences=key_sentences,
                important_entities=important_entities,
                summary_quality_score=quality_score,
                reasoning=result.get('reasoning', '')
            )
            
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            # Return fallback summary
            return self._fallback_summarization(text, time.time() - start_time, str(e))
    
    def _choose_optimal_method(self, text: str) -> str:
        """Choose optimal summarization method based on text characteristics"""
        word_count = len(text.split())
        sentence_count = len(sent_tokenize(text))
        
        # Check if text contains medical terminology
        has_medical_context = self._has_medical_context(text)
        
        # Decision logic
        if word_count < 50:
            return "extractive"  # Too short for abstractive
        elif word_count > 500:
            return "hybrid"  # Best for long texts
        elif has_medical_context and sentence_count > 3:
            return "extractive"  # Preserve medical accuracy
        else:
            return "abstractive"  # Good for general content
    
    def _has_medical_context(self, text: str) -> bool:
        """Check if text has significant medical context"""
        text_lower = text.lower()
        medical_count = 0
        
        for category, terms in self.medical_terms.items():
            medical_count += sum(1 for term in terms if term in text_lower)
        
        return medical_count >= 2
    
    def _extractive_summarization(self, text: str, target_sentences: int) -> Dict:
        """Extractive summarization using TF-IDF and sentence ranking"""
        sentences = sent_tokenize(text)
        
        if len(sentences) <= target_sentences:
            return {
                'summary': text.strip(),
                'method': 'extractive_full_text',
                'key_sentences': sentences,
                'reasoning': 'Text too short to summarize'
            }
        
        try:
            # Create sentence embeddings using TF-IDF
            sentence_vectors = self.tfidf_vectorizer.fit_transform(sentences)
            
            # Calculate sentence importance scores
            sentence_scores = self._calculate_sentence_importance(sentences, sentence_vectors)
            
            # Select top sentences
            ranked_indices = np.argsort(sentence_scores)[-target_sentences:]
            selected_sentences = [sentences[i] for i in sorted(ranked_indices)]
            
            summary = ' '.join(selected_sentences)
            
            return {
                'summary': summary,
                'method': 'extractive_tfidf',
                'key_sentences': selected_sentences,
                'reasoning': f'Selected {len(selected_sentences)} most important sentences'
            }
            
        except Exception as e:
            logger.warning(f"TF-IDF extractive failed, using simple method: {e}")
            return self._simple_extractive_summarization(text, target_sentences)
    
    def _calculate_sentence_importance(self, sentences: List[str], 
                                     sentence_vectors) -> np.ndarray:
        """Calculate importance scores for sentences"""
        scores = np.zeros(len(sentences))
        
        # TF-IDF based scoring
        if sentence_vectors.shape[0] > 1:
            # Calculate average similarity to all other sentences
            similarity_matrix = cosine_similarity(sentence_vectors)
            scores = np.mean(similarity_matrix, axis=1)
        
        # Boost scores for sentences with medical terms
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            
            # Medical term boost
            medical_score = 0
            for terms in self.medical_terms.values():
                medical_score += sum(0.1 for term in terms if term in sentence_lower)
            scores[i] += medical_score
            
            # Position boost (first and last sentences often important)
            if i == 0 or i == len(sentences) - 1:
                scores[i] += 0.1
            
            # Length penalty for very short or very long sentences
            word_count = len(sentence.split())
            if word_count < 5:
                scores[i] -= 0.2
            elif word_count > 30:
                scores[i] -= 0.1
        
        return scores
    
    def _simple_extractive_summarization(self, text: str, target_sentences: int) -> Dict:
        """Simple extractive summarization fallback"""
        sentences = sent_tokenize(text)
        
        if len(sentences) <= target_sentences:
            return {
                'summary': text.strip(),
                'method': 'extractive_simple_full',
                'key_sentences': sentences,
                'reasoning': 'Used all sentences (text too short)'
            }
        
        # Simple heuristic: take first, middle, and last sentences
        indices = []
        if target_sentences >= 1:
            indices.append(0)  # First sentence
        if target_sentences >= 2:
            indices.append(len(sentences) - 1)  # Last sentence
        if target_sentences >= 3:
            indices.append(len(sentences) // 2)  # Middle sentence
        
        # Add more sentences from beginning if needed
        while len(indices) < target_sentences and len(indices) < len(sentences):
            next_idx = len(indices)
            if next_idx not in indices:
                indices.append(next_idx)
        
        selected_sentences = [sentences[i] for i in sorted(indices)]
        summary = ' '.join(selected_sentences)
        
        return {
            'summary': summary,
            'method': 'extractive_simple_heuristic',
            'key_sentences': selected_sentences,
            'reasoning': f'Selected {len(selected_sentences)} sentences using position heuristic'
        }
    
    def _abstractive_summarization(self, text: str, max_length: int) -> Dict:
        """Abstractive summarization using transformer models"""
        # Try different models in order of preference
        model_attempts = [
            ('distilbart_pipeline', 'DistilBART'),
            ('bart_pipeline', 'BART'),
            ('t5', 'T5')
        ]
        
        for model_key, model_name in model_attempts:
            try:
                if model_key == 't5':
                    return self._t5_summarization(text, max_length)
                elif model_key in self.models:
                    return self._pipeline_summarization(text, max_length, model_key, model_name)
            except Exception as e:
                logger.warning(f"{model_name} summarization failed: {e}")
                continue
        
        # If all abstractive methods fail, fall back to extractive
        logger.warning("All abstractive methods failed, falling back to extractive")
        return self._extractive_summarization(text, 2)
    
    def _t5_summarization(self, text: str, max_length: int) -> Dict:
        """T5-based abstractive summarization"""
        if 't5_model' not in self.models or 't5_tokenizer' not in self.models:
            raise Exception("T5 models not loaded")
        
        tokenizer = self.models['t5_tokenizer']
        model = self.models['t5_model']
        
        # Prepare input
        input_text = f"summarize: {text}"
        inputs = tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        # Generate summary
        with torch.no_grad():
            summary_ids = model.generate(
                inputs,
                max_length=max_length // 4,  # Rough token estimate
                min_length=10,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
        
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return {
            'summary': summary,
            'method': 'abstractive_t5',
            'key_sentences': [summary],
            'reasoning': 'T5 generated abstractive summary'
        }
    
    def _pipeline_summarization(self, text: str, max_length: int, 
                               model_key: str, model_name: str) -> Dict:
        """Pipeline-based abstractive summarization"""
        pipeline_model = self.models[model_key]
        
        # Adjust length parameters based on input
        min_length = min(20, max_length // 4)
        target_length = min(max_length // 2, 150)
        
        # Generate summary
        result = pipeline_model(
            text,
            max_length=target_length,
            min_length=min_length,
            do_sample=False
        )
        
        summary = result[0]['summary_text']
        
        return {
            'summary': summary,
            'method': f'abstractive_{model_name.lower()}',
            'key_sentences': [summary],
            'reasoning': f'{model_name} generated abstractive summary'
        }
    
    def _hybrid_summarization(self, text: str, max_length: int, 
                             target_sentences: int) -> Dict:
        """Hybrid summarization combining extractive and abstractive methods"""
        try:
            # First, use extractive to identify key sentences
            extractive_result = self._extractive_summarization(text, target_sentences + 1)
            key_content = extractive_result['summary']
            
            # Then, use abstractive to refine the summary
            if len(key_content) > max_length:
                abstractive_result = self._abstractive_summarization(key_content, max_length)
                final_summary = abstractive_result['summary']
                method_used = f"hybrid_{abstractive_result['method']}"
            else:
                final_summary = key_content
                method_used = "hybrid_extractive_only"
            
            return {
                'summary': final_summary,
                'method': method_used,
                'key_sentences': extractive_result['key_sentences'],
                'reasoning': 'Extractive selection + abstractive refinement'
            }
            
        except Exception as e:
            logger.warning(f"Hybrid summarization failed: {e}")
            # Fall back to extractive only
            return self._extractive_summarization(text, target_sentences)
    
    def _textrank_summarization(self, text: str, target_sentences: int) -> Dict:
        """TextRank algorithm for extractive summarization"""
        try:
            sentences = sent_tokenize(text)
            
            if len(sentences) <= target_sentences:
                return {
                    'summary': text.strip(),
                    'method': 'textrank_full_text',
                    'key_sentences': sentences,
                    'reasoning': 'Text too short for TextRank'
                }
            
            # Create sentence similarity matrix
            similarity_matrix = self._create_similarity_matrix(sentences)
            
            # Apply PageRank algorithm
            nx_graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(nx_graph)
            
            # Select top sentences
            ranked_sentences = sorted(((scores[i], i) for i in range(len(sentences))), reverse=True)
            selected_indices = [idx for _, idx in ranked_sentences[:target_sentences]]
            selected_sentences = [sentences[i] for i in sorted(selected_indices)]
            
            summary = ' '.join(selected_sentences)
            
            return {
                'summary': summary,
                'method': 'extractive_textrank',
                'key_sentences': selected_sentences,
                'reasoning': f'TextRank selected {len(selected_sentences)} top-ranked sentences'
            }
            
        except Exception as e:
            logger.warning(f"TextRank failed: {e}")
            return self._simple_extractive_summarization(text, target_sentences)
    
    def _create_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """Create sentence similarity matrix for TextRank"""
        # Simple word overlap similarity
        def sentence_similarity(sent1, sent2):
            words1 = set(word_tokenize(sent1.lower())) - self.stop_words
            words2 = set(word_tokenize(sent2.lower())) - self.stop_words
            
            if not words1 or not words2:
                return 0
            
            intersection = words1.intersection(words2)
            return len(intersection) / (len(words1) + len(words2) - len(intersection))
        
        n = len(sentences)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    similarity_matrix[i][j] = sentence_similarity(sentences[i], sentences[j])
        
        return similarity_matrix
    
    def _extract_important_entities(self, original_text: str, summary: str) -> List[str]:
        """Extract important entities preserved in summary"""
        # Simple entity extraction using common patterns
        entities = []
        
        # Medical entities
        medical_patterns = [
            r'\b(?:Dr|Doctor|डॉक्टर)\.?\s+([A-Za-z]+)',  # Doctor names
            r'\b(\d+)\s*(?:mg|ml|tablet|दवा)',  # Dosages
            r'\b([A-Za-z]+(?:itis|osis|emia|pathy))',  # Medical conditions
        ]
        
        text_to_search = original_text + " " + summary
        
        for pattern in medical_patterns:
            matches = re.findall(pattern, text_to_search, re.IGNORECASE)
            entities.extend(matches)
        
        # Remove duplicates and return top 5
        unique_entities = list(set(entities))
        return unique_entities[:5]
    
    def _calculate_summary_quality(self, original_text: str, summary: str) -> float:
        """Calculate summary quality score based on various metrics"""
        if not summary or not original_text:
            return 0.0
        
        score = 0.0
        
        # Length appropriateness (0.3 weight)
        length_ratio = len(summary) / len(original_text)
        if 0.1 <= length_ratio <= 0.5:
            score += 0.3
        elif 0.05 <= length_ratio <= 0.8:
            score += 0.2
        else:
            score += 0.1
        
        # Content preservation (0.4 weight)
        original_words = set(word_tokenize(original_text.lower())) - self.stop_words
        summary_words = set(word_tokenize(summary.lower())) - self.stop_words
        
        if original_words:
            content_overlap = len(original_words.intersection(summary_words)) / len(original_words)
            score += 0.4 * content_overlap
        
        # Medical term preservation (0.2 weight)
        medical_terms_in_original = []
        medical_terms_in_summary = []
        
        for terms in self.medical_terms.values():
            for term in terms:
                if term.lower() in original_text.lower():
                    medical_terms_in_original.append(term)
                if term.lower() in summary.lower():
                    medical_terms_in_summary.append(term)
        
        if medical_terms_in_original:
            medical_preservation = len(medical_terms_in_summary) / len(medical_terms_in_original)
            score += 0.2 * medical_preservation
        else:
            score += 0.2  # No medical terms to preserve
        
        # Readability (0.1 weight)
        sentences = sent_tokenize(summary)
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            if 5 <= avg_sentence_length <= 20:  # Optimal sentence length
                score += 0.1
        
        return min(score, 1.0)
    
    def _fallback_summarization(self, text: str, processing_time: float, error: str) -> SummaryResult:
        """Fallback summarization when all methods fail"""
        sentences = sent_tokenize(text)
        
        # Take first two sentences as fallback
        if len(sentences) >= 2:
            summary = sentences[0] + " " + sentences[1]
        elif len(sentences) == 1:
            summary = sentences[0][:200] + "..." if len(sentences[0]) > 200 else sentences[0]
        else:
            summary = text[:100] + "..." if len(text) > 100 else text
        
        return SummaryResult(
            summary=summary,
            original_length=len(text),
            summary_length=len(summary),
            compression_ratio=len(summary) / max(len(text), 1),
            method_used="fallback_simple",
            processing_time=processing_time,
            key_sentences=[summary],
            important_entities=[],
            summary_quality_score=0.3,
            reasoning=f"Fallback summary due to: {error}"
        )

# Factory function
def create_text_summarizer() -> AdvancedTextSummarizer:
    """Create and return advanced text summarizer"""
    try:
        return AdvancedTextSummarizer()
    except Exception as e:
        logger.error(f"Failed to create advanced text summarizer: {str(e)}")
        return None