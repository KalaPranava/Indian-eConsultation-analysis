"""
Sentiment analysis models for Indian e-consultation comments.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, pipeline
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Sentiment analyzer using transformer models, optimized for Indian languages.
    """
    
    def __init__(self, 
                 model_name: str = "ai4bharat/indic-bert",
                 cache_dir: str = "models/sentiment",
                 device: str = None):
        """
        Initialize the sentiment analyzer.
        
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
        
        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.is_trained = False
        
        # Label mappings
        self.label2id = {"negative": 0, "neutral": 1, "positive": 2}
        self.id2label = {v: k for k, v in self.label2id.items()}
        
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
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    num_labels=len(self.label2id),
                    id2label=self.id2label,
                    label2id=self.label2id
                )
                self.is_trained = True
            else:
                logger.info(f"Loading pre-trained model: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir
                )
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=len(self.label2id),
                    id2label=self.id2label,
                    label2id=self.label2id,
                    cache_dir=self.cache_dir
                )
                
            self.model.to(self.device)
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
            
    def prepare_data(self, 
                    texts: List[str], 
                    labels: List[str] = None,
                    max_length: int = 512) -> Dict[str, torch.Tensor]:
        """
        Prepare data for training or inference.
        
        Args:
            texts (List[str]): List of input texts
            labels (List[str]): List of labels (optional)
            max_length (int): Maximum sequence length
            
        Returns:
            Dict[str, torch.Tensor]: Tokenized data
        """
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Add labels if provided
        if labels:
            label_ids = [self.label2id[label] for label in labels]
            encodings["labels"] = torch.tensor(label_ids, dtype=torch.long)
            
        return encodings
        
    def predict(self, 
               texts: Union[str, List[str]], 
               return_probabilities: bool = False,
               batch_size: int = 32) -> Union[Dict, List[Dict]]:
        """
        Predict sentiment for given texts.
        
        Args:
            texts (Union[str, List[str]]): Input text(s)
            return_probabilities (bool): Whether to return probabilities
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
            results = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_results = self.pipeline(batch, return_all_scores=return_probabilities)
                results.extend(batch_results)
                
            # Process results
            processed_results = []
            for result in results:
                if return_probabilities:
                    # Convert to more readable format
                    scores = {item['label']: item['score'] for item in result}
                    predicted_label = max(scores, key=scores.get)
                    processed_results.append({
                        'label': predicted_label,
                        'confidence': scores[predicted_label],
                        'scores': scores
                    })
                else:
                    processed_results.append({
                        'label': result['label'],
                        'confidence': result['score']
                    })
                    
            return processed_results[0] if single_input else processed_results
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
            
    def predict_dataframe(self, 
                         df: pd.DataFrame, 
                         text_column: str,
                         batch_size: int = 32) -> pd.DataFrame:
        """
        Predict sentiment for texts in a DataFrame.
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Column containing texts
            batch_size (int): Batch size for prediction
            
        Returns:
            pd.DataFrame: Dataframe with predictions
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in dataframe")
            
        df_copy = df.copy()
        
        # Get valid texts
        valid_mask = df_copy[text_column].notna() & (df_copy[text_column] != '')
        valid_texts = df_copy.loc[valid_mask, text_column].tolist()
        
        if not valid_texts:
            logger.warning("No valid texts found for prediction")
            df_copy['predicted_sentiment'] = 'unknown'
            df_copy['sentiment_confidence'] = 0.0
            return df_copy
            
        # Predict
        logger.info(f"Predicting sentiment for {len(valid_texts)} texts")
        predictions = self.predict(valid_texts, return_probabilities=True, batch_size=batch_size)
        
        # Add predictions to dataframe
        df_copy['predicted_sentiment'] = 'unknown'
        df_copy['sentiment_confidence'] = 0.0
        df_copy['sentiment_scores'] = None
        
        # Fill predictions for valid rows
        pred_idx = 0
        for idx, is_valid in valid_mask.items():
            if is_valid:
                pred = predictions[pred_idx]
                df_copy.loc[idx, 'predicted_sentiment'] = pred['label']
                df_copy.loc[idx, 'sentiment_confidence'] = pred['confidence']
                df_copy.loc[idx, 'sentiment_scores'] = pred['scores']
                pred_idx += 1
                
        return df_copy
        
    def evaluate(self, texts: List[str], labels: List[str]) -> Dict[str, float]:
        """
        Evaluate the model on given texts and labels.
        
        Args:
            texts (List[str]): List of texts
            labels (List[str]): List of true labels
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        # Predict
        predictions = self.predict(texts)
        predicted_labels = [pred['label'] for pred in predictions]
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predicted_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predicted_labels, average='weighted'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            labels, predicted_labels, average=None, labels=list(self.label2id.keys())
        )
        
        # Confusion matrix
        cm = confusion_matrix(labels, predicted_labels, labels=list(self.label2id.keys()))
        
        results = {
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': {}
        }
        
        # Add per-class metrics
        for i, label in enumerate(self.label2id.keys()):
            results['per_class_metrics'][label] = {
                'precision': precision_per_class[i],
                'recall': recall_per_class[i],
                'f1': f1_per_class[i]
            }
            
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


class SentimentDataset(torch.utils.data.Dataset):
    """
    Dataset class for sentiment analysis.
    """
    
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
        
    def __len__(self):
        return len(self.encodings['input_ids'])


class SentimentTrainer:
    """
    Trainer class for fine-tuning sentiment analysis models.
    """
    
    def __init__(self, analyzer: SentimentAnalyzer):
        self.analyzer = analyzer
        self.trainer = None
        
    def prepare_datasets(self, 
                        train_texts: List[str], 
                        train_labels: List[str],
                        val_texts: List[str] = None,
                        val_labels: List[str] = None,
                        max_length: int = 512) -> Tuple[SentimentDataset, Optional[SentimentDataset]]:
        """
        Prepare training and validation datasets.
        
        Args:
            train_texts (List[str]): Training texts
            train_labels (List[str]): Training labels
            val_texts (List[str]): Validation texts (optional)
            val_labels (List[str]): Validation labels (optional)
            max_length (int): Maximum sequence length
            
        Returns:
            Tuple[SentimentDataset, Optional[SentimentDataset]]: Train and validation datasets
        """
        # Prepare training data
        train_encodings = self.analyzer.prepare_data(train_texts, train_labels, max_length)
        train_dataset = SentimentDataset(train_encodings)
        
        # Prepare validation data if provided
        val_dataset = None
        if val_texts and val_labels:
            val_encodings = self.analyzer.prepare_data(val_texts, val_labels, max_length)
            val_dataset = SentimentDataset(val_encodings)
            
        return train_dataset, val_dataset
        
    def train(self,
             train_texts: List[str],
             train_labels: List[str],
             val_texts: List[str] = None,
             val_labels: List[str] = None,
             output_dir: str = "models/sentiment/trained",
             num_train_epochs: int = 3,
             learning_rate: float = 2e-5,
             batch_size: int = 16,
             warmup_steps: int = 500,
             weight_decay: float = 0.01,
             save_best_model: bool = True) -> Dict[str, Any]:
        """
        Fine-tune the sentiment analysis model.
        
        Args:
            train_texts (List[str]): Training texts
            train_labels (List[str]): Training labels
            val_texts (List[str]): Validation texts (optional)
            val_labels (List[str]): Validation labels (optional)
            output_dir (str): Output directory for trained model
            num_train_epochs (int): Number of training epochs
            learning_rate (float): Learning rate
            batch_size (int): Training batch size
            warmup_steps (int): Number of warmup steps
            weight_decay (float): Weight decay
            save_best_model (bool): Whether to save the best model
            
        Returns:
            Dict[str, Any]: Training results
        """
        if not self.analyzer.model or not self.analyzer.tokenizer:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        logger.info("Preparing datasets for training")
        train_dataset, val_dataset = self.prepare_datasets(
            train_texts, train_labels, val_texts, val_labels
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            logging_dir=f'{output_dir}/logs',
            evaluation_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch" if save_best_model else "no",
            load_best_model_at_end=save_best_model,
            metric_for_best_model="eval_loss" if val_dataset else None,
            greater_is_better=False,
            save_total_limit=2,
            logging_steps=100,
            report_to=None  # Disable wandb/tensorboard logging
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.analyzer.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.analyzer.tokenizer,
        )
        
        logger.info("Starting training")
        train_result = self.trainer.train()
        
        # Save the final model
        if save_best_model:
            self.trainer.save_model()
            logger.info(f"Model saved to {output_dir}")
            
        self.analyzer.is_trained = True
        
        return {
            'train_loss': train_result.training_loss,
            'train_runtime': train_result.metrics['train_runtime'],
            'train_samples_per_second': train_result.metrics['train_samples_per_second'],
            'train_steps_per_second': train_result.metrics['train_steps_per_second']
        }


def create_sample_sentiment_data():
    """Create sample sentiment data for testing."""
    sample_data = [
        ("यह सेवा बहुत अच्छी है", "positive"),
        ("डॉक्टर ने बहुत अच्छी सलाह दी", "positive"), 
        ("बहुत helpful consultation था", "positive"),
        ("Service okay है but improve करने की जरूरत है", "neutral"),
        ("Wait time थोड़ा ज्यादा था", "neutral"),
        ("Average experience रहा", "neutral"),
        ("बहुत bad experience था", "negative"),
        ("Doctor रूखे थे और properly check नहीं किया", "negative"),
        ("Waste of time और money", "negative"),
        ("Excellent service! डॉक्टर बहुत knowledgeable हैं", "positive"),
        ("Not satisfied with consultation", "negative")
    ]
    
    texts, labels = zip(*sample_data)
    return list(texts), list(labels)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    analyzer = SentimentAnalyzer()
    analyzer.load_model()
    
    # Create sample data
    texts, labels = create_sample_sentiment_data()
    
    # Predict on sample texts
    predictions = analyzer.predict(texts[:3], return_probabilities=True)
    
    for text, pred in zip(texts[:3], predictions):
        print(f"Text: {text}")
        print(f"Predicted: {pred['label']} (confidence: {pred['confidence']:.3f})")
        print("-" * 50)