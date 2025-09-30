"""
Emotion detection model for Indian e-consultation comments.
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
import logging
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, TrainingArguments, Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from pathlib import Path

logger = logging.getLogger(__name__)


class EmotionDetector:
    """
    Emotion detection using transformer models.
    """
    
    def __init__(self, 
                 model_name: str = "cardiffnlp/twitter-roberta-base-emotion",
                 cache_dir: str = "models/emotion",
                 device: str = None):
        """
        Initialize the emotion detector.
        
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
        
        # Standard emotion labels (can be customized based on model)
        self.emotion_labels = ["joy", "anger", "fear", "sadness", "surprise", "neutral"]
        
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
                self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
                self.is_trained = True
            else:
                logger.info(f"Loading pre-trained model: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir
                )
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir
                )
                
            self.model.to(self.device)
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True
            )
            
            # Update emotion labels based on model config
            if hasattr(self.model.config, 'id2label'):
                self.emotion_labels = list(self.model.config.id2label.values())
                
            logger.info(f"Model loaded successfully with emotions: {self.emotion_labels}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
            
    def predict(self, 
               texts: Union[str, List[str]], 
               top_k: int = 3,
               threshold: float = 0.1,
               batch_size: int = 32) -> Union[Dict, List[Dict]]:
        """
        Predict emotions for given texts.
        
        Args:
            texts (Union[str, List[str]]): Input text(s)
            top_k (int): Number of top emotions to return
            threshold (float): Minimum confidence threshold
            batch_size (int): Batch size for prediction
            
        Returns:
            Union[Dict, List[Dict]]: Prediction results
        """
        if not self.pipeline:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
            
        try:
            # Predict in batches
            all_results = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_results = self.pipeline(batch)
                all_results.extend(batch_results)
                
            # Process results
            processed_results = []
            for result in all_results:
                # Sort by confidence
                sorted_emotions = sorted(result, key=lambda x: x['score'], reverse=True)
                
                # Filter by threshold and get top_k
                filtered_emotions = [
                    {'emotion': item['label'], 'confidence': item['score']} 
                    for item in sorted_emotions[:top_k] 
                    if item['score'] >= threshold
                ]
                
                if not filtered_emotions:
                    # If no emotions meet threshold, return the top one
                    filtered_emotions = [
                        {'emotion': sorted_emotions[0]['label'], 'confidence': sorted_emotions[0]['score']}
                    ]
                    
                processed_results.append({
                    'primary_emotion': filtered_emotions[0]['emotion'],
                    'primary_confidence': filtered_emotions[0]['confidence'],
                    'all_emotions': filtered_emotions,
                    'emotion_scores': {item['label']: item['score'] for item in result}
                })
                
            return processed_results[0] if single_input else processed_results
            
        except Exception as e:
            logger.error(f"Error during emotion prediction: {str(e)}")
            raise
            
    def predict_dataframe(self, 
                         df: pd.DataFrame, 
                         text_column: str,
                         batch_size: int = 32,
                         **kwargs) -> pd.DataFrame:
        """
        Predict emotions for texts in a DataFrame.
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Column containing texts
            batch_size (int): Batch size for prediction
            **kwargs: Additional arguments for predict()
            
        Returns:
            pd.DataFrame: Dataframe with emotion predictions
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in dataframe")
            
        df_copy = df.copy()
        
        # Get valid texts
        valid_mask = df_copy[text_column].notna() & (df_copy[text_column] != '')
        valid_texts = df_copy.loc[valid_mask, text_column].tolist()
        
        if not valid_texts:
            logger.warning("No valid texts found for emotion prediction")
            df_copy['primary_emotion'] = 'unknown'
            df_copy['emotion_confidence'] = 0.0
            df_copy['emotion_scores'] = None
            return df_copy
            
        # Predict
        logger.info(f"Predicting emotions for {len(valid_texts)} texts")
        predictions = self.predict(valid_texts, batch_size=batch_size, **kwargs)
        
        # Add predictions to dataframe
        df_copy['primary_emotion'] = 'unknown'
        df_copy['emotion_confidence'] = 0.0
        df_copy['emotion_scores'] = None
        df_copy['all_emotions'] = None
        
        # Fill predictions for valid rows
        pred_idx = 0
        for idx, is_valid in valid_mask.items():
            if is_valid:
                pred = predictions[pred_idx]
                df_copy.loc[idx, 'primary_emotion'] = pred['primary_emotion']
                df_copy.loc[idx, 'emotion_confidence'] = pred['primary_confidence']
                df_copy.loc[idx, 'emotion_scores'] = pred['emotion_scores']
                df_copy.loc[idx, 'all_emotions'] = pred['all_emotions']
                pred_idx += 1
                
        return df_copy
        
    def analyze_emotion_distribution(self, 
                                   df: pd.DataFrame, 
                                   emotion_column: str = 'primary_emotion') -> Dict[str, Any]:
        """
        Analyze emotion distribution in a dataset.
        
        Args:
            df (pd.DataFrame): Dataframe with emotion predictions
            emotion_column (str): Column containing emotion labels
            
        Returns:
            Dict[str, Any]: Emotion distribution analysis
        """
        if emotion_column not in df.columns:
            raise ValueError(f"Column '{emotion_column}' not found in dataframe")
            
        # Basic distribution
        emotion_counts = df[emotion_column].value_counts()
        emotion_percentages = df[emotion_column].value_counts(normalize=True) * 100
        
        # Confidence statistics
        confidence_stats = {}
        if 'emotion_confidence' in df.columns:
            for emotion in emotion_counts.index:
                emotion_data = df[df[emotion_column] == emotion]['emotion_confidence']
                confidence_stats[emotion] = {
                    'mean_confidence': float(emotion_data.mean()),
                    'median_confidence': float(emotion_data.median()),
                    'min_confidence': float(emotion_data.min()),
                    'max_confidence': float(emotion_data.max()),
                    'std_confidence': float(emotion_data.std())
                }
                
        return {
            'total_samples': len(df),
            'unique_emotions': len(emotion_counts),
            'emotion_counts': emotion_counts.to_dict(),
            'emotion_percentages': emotion_percentages.to_dict(),
            'most_common_emotion': emotion_counts.index[0] if len(emotion_counts) > 0 else None,
            'confidence_stats': confidence_stats
        }
        
    def get_emotion_insights(self, 
                           df: pd.DataFrame, 
                           text_column: str = 'comment',
                           emotion_column: str = 'primary_emotion',
                           confidence_column: str = 'emotion_confidence',
                           min_confidence: float = 0.5) -> Dict[str, Any]:
        """
        Get detailed insights about emotions in the dataset.
        
        Args:
            df (pd.DataFrame): Dataframe with predictions
            text_column (str): Column containing original texts
            emotion_column (str): Column containing emotion labels
            confidence_column (str): Column containing confidence scores
            min_confidence (float): Minimum confidence for high-confidence samples
            
        Returns:
            Dict[str, Any]: Detailed emotion insights
        """
        # Filter high-confidence predictions
        high_conf_df = df[df[confidence_column] >= min_confidence] if confidence_column in df.columns else df
        
        insights = {
            'total_samples': len(df),
            'high_confidence_samples': len(high_conf_df),
            'high_confidence_ratio': len(high_conf_df) / len(df) if len(df) > 0 else 0
        }
        
        # Emotion distribution for high-confidence samples
        if len(high_conf_df) > 0:
            emotion_dist = self.analyze_emotion_distribution(high_conf_df, emotion_column)
            insights['high_confidence_distribution'] = emotion_dist
            
            # Sample texts for each emotion
            sample_texts = {}
            for emotion in emotion_dist['emotion_counts'].keys():
                emotion_samples = high_conf_df[
                    high_conf_df[emotion_column] == emotion
                ][text_column].head(3).tolist()
                sample_texts[emotion] = emotion_samples
                
            insights['sample_texts'] = sample_texts
            
        # Overall distribution
        overall_dist = self.analyze_emotion_distribution(df, emotion_column)
        insights['overall_distribution'] = overall_dist
        
        return insights
        
    def evaluate(self, 
                texts: List[str], 
                true_emotions: List[str],
                **predict_kwargs) -> Dict[str, Any]:
        """
        Evaluate the emotion detection model.
        
        Args:
            texts (List[str]): List of texts
            true_emotions (List[str]): List of true emotion labels
            **predict_kwargs: Additional arguments for predict()
            
        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        # Predict
        predictions = self.predict(texts, **predict_kwargs)
        predicted_emotions = [pred['primary_emotion'] for pred in predictions]
        
        # Calculate metrics
        accuracy = accuracy_score(true_emotions, predicted_emotions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_emotions, predicted_emotions, average='weighted', zero_division=0
        )
        
        # Classification report
        class_report = classification_report(
            true_emotions, predicted_emotions, 
            output_dict=True, zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'classification_report': class_report,
            'predicted_emotions': predicted_emotions
        }
        
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


class MultiLabelEmotionDetector(EmotionDetector):
    """
    Multi-label emotion detection for texts that may express multiple emotions.
    """
    
    def __init__(self, 
                 model_name: str = "cardiffnlp/twitter-roberta-base-emotion",
                 cache_dir: str = "models/emotion",
                 device: str = None,
                 threshold: float = 0.3):
        """
        Initialize multi-label emotion detector.
        
        Args:
            model_name (str): Name of the pre-trained model
            cache_dir (str): Directory to cache models
            device (str): Device to use
            threshold (float): Threshold for multi-label classification
        """
        super().__init__(model_name, cache_dir, device)
        self.threshold = threshold
        
    def predict_multilabel(self, 
                          texts: Union[str, List[str]], 
                          threshold: float = None,
                          batch_size: int = 32) -> Union[Dict, List[Dict]]:
        """
        Predict multiple emotions for given texts.
        
        Args:
            texts (Union[str, List[str]]): Input text(s)
            threshold (float): Confidence threshold for multi-label
            batch_size (int): Batch size for prediction
            
        Returns:
            Union[Dict, List[Dict]]: Multi-label prediction results
        """
        if threshold is None:
            threshold = self.threshold
            
        # Get single-label predictions with all scores
        predictions = self.predict(
            texts, 
            top_k=len(self.emotion_labels), 
            threshold=0.0,  # Get all scores
            batch_size=batch_size
        )
        
        if isinstance(texts, str):
            predictions = [predictions]
            single_input = True
        else:
            single_input = False
            
        # Process for multi-label
        multilabel_results = []
        for pred in predictions:
            # Get emotions above threshold
            selected_emotions = [
                emotion_info for emotion_info in pred['all_emotions']
                if emotion_info['confidence'] >= threshold
            ]
            
            if not selected_emotions:
                # If no emotions meet threshold, take the top one
                selected_emotions = [pred['all_emotions'][0]]
                
            multilabel_results.append({
                'emotions': [e['emotion'] for e in selected_emotions],
                'confidences': [e['confidence'] for e in selected_emotions],
                'emotion_confidence_pairs': selected_emotions,
                'all_scores': pred['emotion_scores']
            })
            
        return multilabel_results[0] if single_input else multilabel_results


def create_sample_emotion_data():
    """Create sample data for emotion detection testing."""
    sample_data = [
        ("यह service बहुत अच्छी है! मैं बहुत खुश हूं।", "joy"),
        ("डॉक्टर ने बहुत अच्छी care की। Thank you so much!", "joy"),
        ("Wait time बहुत ज्यादा था। Very frustrating experience.", "anger"),
        ("Doctor ने properly check नहीं किया। I am angry about this.", "anger"),
        ("मुझे बहुत चिंता हो रही है। Is this treatment safe?", "fear"),
        ("I am worried about the side effects of this medicine.", "fear"),
        ("Service normal है। Nothing special but okay.", "neutral"),
        ("बहुत disappointing experience था। Feeling sad.", "sadness"),
        ("Unexpected results मिले! यह तो amazing है!", "surprise")
    ]
    
    texts, emotions = zip(*sample_data)
    return list(texts), list(emotions)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    sample_texts, sample_emotions = create_sample_emotion_data()
    
    try:
        # Initialize emotion detector
        detector = EmotionDetector()
        detector.load_model()
        
        print("=== Emotion Detection Results ===")
        
        # Test single predictions
        for text in sample_texts[:3]:
            result = detector.predict(text, top_k=3)
            print(f"\nText: {text}")
            print(f"Primary emotion: {result['primary_emotion']} (confidence: {result['primary_confidence']:.3f})")
            print(f"All emotions: {result['all_emotions']}")
            
        # Test multi-label detection
        print("\n=== Multi-label Emotion Detection ===")
        multilabel_detector = MultiLabelEmotionDetector(threshold=0.2)
        multilabel_detector.load_model()
        
        multilabel_result = multilabel_detector.predict_multilabel(sample_texts[0])
        print(f"\nText: {sample_texts[0]}")
        print(f"Detected emotions: {multilabel_result['emotions']}")
        print(f"Confidences: {[f'{c:.3f}' for c in multilabel_result['confidences']]}")
        
        # Test with DataFrame
        print("\n=== DataFrame Processing ===")
        df = pd.DataFrame({
            'comment': sample_texts[:5],
            'true_emotion': sample_emotions[:5]
        })
        
        df_with_emotions = detector.predict_dataframe(df, 'comment')
        print(f"Processed {len(df_with_emotions)} samples")
        
        # Analyze distribution
        distribution = detector.analyze_emotion_distribution(df_with_emotions)
        print(f"Emotion distribution: {distribution['emotion_percentages']}")
        
    except Exception as e:
        print(f"Emotion detection test failed: {e}")
        print("Note: Make sure you have the required models and dependencies installed.")