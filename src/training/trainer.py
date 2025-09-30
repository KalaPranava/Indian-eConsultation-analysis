"""
Model training utilities and orchestration.
"""

import os
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from datetime import datetime
import json

from ..data import DataLoader, TextPreprocessor
from ..models import SentimentAnalyzer, SentimentTrainer, EmotionDetector

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Orchestrates training of different models with unified configuration.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the model trainer.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize components
        self.data_loader = DataLoader()
        self.preprocessor = TextPreprocessor()
        
        # Create directories
        self.models_dir = Path("models")
        self.logs_dir = Path("logs")
        self.results_dir = Path("results")
        
        for dir_path in [self.models_dir, self.logs_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            # Return default config
            return self._get_default_config()
            
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'models': {
                'sentiment': {
                    'name': 'ai4bharat/indic-bert',
                    'cache_dir': './models/sentiment',
                    'max_length': 512,
                    'batch_size': 16,
                    'learning_rate': 2e-5,
                    'epochs': 3
                },
                'emotion': {
                    'name': 'cardiffnlp/twitter-roberta-base-emotion',
                    'cache_dir': './models/emotion',
                    'max_length': 512,
                    'batch_size': 16,
                    'learning_rate': 2e-5,
                    'epochs': 3
                }
            },
            'training': {
                'validation_split': 0.2,
                'test_split': 0.1,
                'random_seed': 42,
                'early_stopping_patience': 3,
                'save_best_model': True
            },
            'preprocessing': {
                'clean_html': True,
                'remove_urls': True,
                'remove_mentions': True,
                'min_length': 10,
                'max_length': 1000
            }
        }
        
    def prepare_data(self, 
                    data_path: str, 
                    text_column: str = 'comment',
                    label_column: str = 'label',
                    preprocess: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load and prepare data for training.
        
        Args:
            data_path (str): Path to the data file
            text_column (str): Name of the text column
            label_column (str): Name of the label column
            preprocess (bool): Whether to preprocess the text
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: Prepared data and statistics
        """
        logger.info(f"Loading data from {data_path}")
        
        # Load data based on file extension
        if data_path.endswith('.csv'):
            df = self.data_loader.load_csv(data_path)
        elif data_path.endswith('.json'):
            data = self.data_loader.load_json(data_path)
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
            
        logger.info(f"Loaded {len(df)} samples")
        
        # Validate required columns
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in data")
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in data")
            
        # Remove missing values
        initial_count = len(df)
        df = df.dropna(subset=[text_column, label_column])
        logger.info(f"Removed {initial_count - len(df)} samples with missing values")
        
        # Preprocess text if requested
        if preprocess:
            logger.info("Preprocessing text data")
            preprocessing_config = self.config.get('preprocessing', {})
            
            df_processed = self.preprocessor.process_dataframe(
                df, [text_column], **preprocessing_config
            )
            
            # Use processed text
            processed_column = f"{text_column}_processed"
            if processed_column in df_processed.columns:
                # Filter valid samples
                valid_mask = df_processed[f"{text_column}_is_valid"]
                df = df_processed[valid_mask].copy()
                df[text_column] = df[processed_column]
                logger.info(f"After preprocessing: {len(df)} valid samples")
                
        # Remove duplicates
        df = self.preprocessor.remove_duplicates(df, text_column)
        
        # Get statistics
        stats = {
            'total_samples': len(df),
            'unique_labels': df[label_column].nunique(),
            'label_distribution': df[label_column].value_counts().to_dict(),
            'text_length_stats': {
                'mean': df[text_column].str.len().mean(),
                'median': df[text_column].str.len().median(),
                'min': df[text_column].str.len().min(),
                'max': df[text_column].str.len().max()
            }
        }
        
        logger.info(f"Data preparation complete: {stats}")
        return df, stats
        
    def split_data(self, 
                  df: pd.DataFrame,
                  text_column: str = 'comment',
                  label_column: str = 'label',
                  stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of the text column
            label_column (str): Name of the label column
            stratify (bool): Whether to stratify split
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, test sets
        """
        training_config = self.config.get('training', {})
        val_split = training_config.get('validation_split', 0.2)
        test_split = training_config.get('test_split', 0.1)
        random_seed = training_config.get('random_seed', 42)
        
        stratify_column = df[label_column] if stratify else None
        
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_split,
            random_state=random_seed,
            stratify=stratify_column
        )
        
        # Second split: separate train and validation
        if val_split > 0:
            stratify_column_train = train_val_df[label_column] if stratify else None
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=val_split / (1 - test_split),  # Adjust for remaining data
                random_state=random_seed,
                stratify=stratify_column_train
            )
        else:
            train_df = train_val_df
            val_df = pd.DataFrame()
            
        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
        
    def train_sentiment_model(self, 
                            data_path: str,
                            text_column: str = 'comment',
                            label_column: str = 'sentiment',
                            model_name: str = None,
                            output_dir: str = None) -> Dict[str, Any]:
        """
        Train sentiment analysis model.
        
        Args:
            data_path (str): Path to training data
            text_column (str): Name of text column
            label_column (str): Name of label column
            model_name (str): Model name (overrides config)
            output_dir (str): Output directory (overrides config)
            
        Returns:
            Dict[str, Any]: Training results
        """
        logger.info("Starting sentiment model training")
        
        # Load configuration
        sentiment_config = self.config.get('models', {}).get('sentiment', {})
        training_config = self.config.get('training', {})
        
        model_name = model_name or sentiment_config.get('name', 'ai4bharat/indic-bert')
        output_dir = output_dir or str(self.models_dir / 'sentiment' / 'trained')
        
        # Prepare data
        df, data_stats = self.prepare_data(data_path, text_column, label_column)
        train_df, val_df, test_df = self.split_data(df, text_column, label_column)
        
        # Initialize model
        analyzer = SentimentAnalyzer(
            model_name=model_name,
            cache_dir=sentiment_config.get('cache_dir', './models/sentiment')
        )
        analyzer.load_model()
        
        # Initialize trainer
        trainer = SentimentTrainer(analyzer)
        
        # Prepare training data
        train_texts = train_df[text_column].tolist()
        train_labels = train_df[label_column].tolist()
        
        val_texts = val_df[text_column].tolist() if len(val_df) > 0 else None
        val_labels = val_df[label_column].tolist() if len(val_df) > 0 else None
        
        # Train model
        training_args = {
            'output_dir': output_dir,
            'num_train_epochs': sentiment_config.get('epochs', 3),
            'learning_rate': sentiment_config.get('learning_rate', 2e-5),
            'batch_size': sentiment_config.get('batch_size', 16),
            'save_best_model': training_config.get('save_best_model', True)
        }
        
        training_results = trainer.train(
            train_texts, train_labels,
            val_texts, val_labels,
            **training_args
        )
        
        # Evaluate on test set
        test_results = {}
        if len(test_df) > 0:
            test_texts = test_df[text_column].tolist()
            test_labels = test_df[label_column].tolist()
            test_results = analyzer.evaluate(test_texts, test_labels)
            
        # Save results
        results = {
            'model_type': 'sentiment',
            'model_name': model_name,
            'data_stats': data_stats,
            'training_results': training_results,
            'test_results': test_results,
            'config': sentiment_config,
            'timestamp': datetime.now().isoformat()
        }
        
        results_file = self.results_dir / f"sentiment_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Sentiment model training completed. Results saved to {results_file}")
        return results
        
    def train_emotion_model(self, 
                          data_path: str,
                          text_column: str = 'comment',
                          label_column: str = 'emotion',
                          model_name: str = None,
                          output_dir: str = None) -> Dict[str, Any]:
        """
        Train emotion detection model.
        
        Args:
            data_path (str): Path to training data
            text_column (str): Name of text column
            label_column (str): Name of label column
            model_name (str): Model name (overrides config)
            output_dir (str): Output directory (overrides config)
            
        Returns:
            Dict[str, Any]: Training results
        """
        logger.info("Starting emotion model training")
        
        # Load configuration
        emotion_config = self.config.get('models', {}).get('emotion', {})
        training_config = self.config.get('training', {})
        
        model_name = model_name or emotion_config.get('name', 'cardiffnlp/twitter-roberta-base-emotion')
        output_dir = output_dir or str(self.models_dir / 'emotion' / 'trained')
        
        # Prepare data
        df, data_stats = self.prepare_data(data_path, text_column, label_column)
        train_df, val_df, test_df = self.split_data(df, text_column, label_column)
        
        # Initialize model
        detector = EmotionDetector(
            model_name=model_name,
            cache_dir=emotion_config.get('cache_dir', './models/emotion')
        )
        detector.load_model()
        
        # Evaluate on test set (for pre-trained models)
        test_results = {}
        if len(test_df) > 0:
            test_texts = test_df[text_column].tolist()
            test_labels = test_df[label_column].tolist()
            test_results = detector.evaluate(test_texts, test_labels)
            
        # Save results
        results = {
            'model_type': 'emotion',
            'model_name': model_name,
            'data_stats': data_stats,
            'test_results': test_results,
            'config': emotion_config,
            'timestamp': datetime.now().isoformat()
        }
        
        results_file = self.results_dir / f"emotion_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Emotion model evaluation completed. Results saved to {results_file}")
        return results
        
    def compare_models(self, 
                      results_files: List[str]) -> Dict[str, Any]:
        """
        Compare results from multiple model training runs.
        
        Args:
            results_files (List[str]): List of result file paths
            
        Returns:
            Dict[str, Any]: Comparison results
        """
        comparison = {
            'models': [],
            'best_model': None,
            'metrics_comparison': {}
        }
        
        best_score = 0
        best_model_idx = 0
        
        for i, file_path in enumerate(results_files):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                    
                model_info = {
                    'file': file_path,
                    'model_type': results.get('model_type'),
                    'model_name': results.get('model_name'),
                    'timestamp': results.get('timestamp')
                }
                
                # Extract test metrics
                test_results = results.get('test_results', {})
                if 'f1_weighted' in test_results:
                    model_info['f1_score'] = test_results['f1_weighted']
                    if test_results['f1_weighted'] > best_score:
                        best_score = test_results['f1_weighted']
                        best_model_idx = i
                        
                if 'accuracy' in test_results:
                    model_info['accuracy'] = test_results['accuracy']
                    
                comparison['models'].append(model_info)
                
            except Exception as e:
                logger.error(f"Error loading results from {file_path}: {e}")
                
        if comparison['models']:
            comparison['best_model'] = comparison['models'][best_model_idx]
            
        return comparison


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for different model types.
    """
    
    @staticmethod
    def sentiment_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
        """Calculate sentiment analysis metrics."""
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
    @staticmethod
    def emotion_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
        """Calculate emotion detection metrics."""
        return EvaluationMetrics.sentiment_metrics(y_true, y_pred)  # Same metrics
        
    @staticmethod
    def summarization_metrics(summaries: List[str], 
                            references: List[str] = None) -> Dict[str, float]:
        """Calculate summarization metrics."""
        if not summaries:
            return {'avg_length': 0, 'coverage': 0}
            
        avg_length = np.mean([len(summary.split()) for summary in summaries])
        
        metrics = {'avg_length': avg_length}
        
        # If references are provided, calculate ROUGE-like metrics (simplified)
        if references:
            coverage_scores = []
            for summary, reference in zip(summaries, references):
                summary_words = set(summary.lower().split())
                reference_words = set(reference.lower().split())
                
                if reference_words:
                    coverage = len(summary_words & reference_words) / len(reference_words)
                    coverage_scores.append(coverage)
                    
            metrics['coverage'] = np.mean(coverage_scores) if coverage_scores else 0
            
        return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    trainer = ModelTrainer()
    
    # Create sample data for testing
    from ..data.loader import create_sample_data
    create_sample_data()
    
    print("Model trainer initialized successfully")
    print(f"Configuration: {trainer.config}")
    
    # Test data preparation
    try:
        df, stats = trainer.prepare_data("data/sample/comments.csv", 'comment', 'sentiment')
        print(f"Data preparation test successful: {stats}")
        
        train_df, val_df, test_df = trainer.split_data(df, 'comment', 'sentiment')
        print(f"Data split test successful - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
    except Exception as e:
        print(f"Test failed: {e}")