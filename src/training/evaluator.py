"""
Model evaluation utilities and metrics calculation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation with metrics calculation and visualization.
    """
    
    def __init__(self, output_dir: str = "results/evaluation"):
        """
        Initialize the model evaluator.
        
        Args:
            output_dir (str): Directory to save evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate_classification(self, 
                              y_true: List[str], 
                              y_pred: List[str],
                              y_prob: List[List[float]] = None,
                              labels: List[str] = None,
                              model_name: str = "model") -> Dict[str, Any]:
        """
        Comprehensive evaluation for classification models.
        
        Args:
            y_true (List[str]): True labels
            y_pred (List[str]): Predicted labels
            y_prob (List[List[float]]): Prediction probabilities (optional)
            labels (List[str]): List of class labels
            model_name (str): Name of the model being evaluated
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        if not y_true or not y_pred:
            raise ValueError("True and predicted labels cannot be empty")
            
        if len(y_true) != len(y_pred):
            raise ValueError("True and predicted labels must have same length")
            
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = (
            precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        )
        
        # Unique labels
        unique_labels = sorted(list(set(y_true + y_pred)))
        if labels is None:
            labels = unique_labels
            
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Classification report
        class_report = classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        )
        
        # AUC score (if probabilities provided)
        auc_scores = {}
        if y_prob is not None and len(labels) > 1:
            try:
                # Convert to binary format for multi-class AUC
                from sklearn.preprocessing import LabelBinarizer
                from sklearn.metrics import roc_auc_score
                
                lb = LabelBinarizer()
                y_true_binary = lb.fit_transform(y_true)
                
                if y_true_binary.shape[1] > 1:  # Multi-class
                    auc_scores['macro'] = roc_auc_score(
                        y_true_binary, y_prob, average='macro', multi_class='ovr'
                    )
                    auc_scores['weighted'] = roc_auc_score(
                        y_true_binary, y_prob, average='weighted', multi_class='ovr'
                    )
                else:  # Binary
                    auc_scores['binary'] = roc_auc_score(y_true_binary, [p[1] for p in y_prob])
                    
            except Exception as e:
                logger.warning(f"Could not calculate AUC scores: {e}")
                
        # Compile results
        results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'sample_count': len(y_true),
            'metrics': {
                'accuracy': float(accuracy),
                'precision_weighted': float(precision),
                'recall_weighted': float(recall),
                'f1_weighted': float(f1),
                'support_total': int(np.sum(support))
            },
            'per_class_metrics': {},
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'auc_scores': auc_scores,
            'labels': labels
        }
        
        # Add per-class metrics
        for i, label in enumerate(labels):
            if i < len(precision_per_class):
                results['per_class_metrics'][label] = {
                    'precision': float(precision_per_class[i]),
                    'recall': float(recall_per_class[i]),
                    'f1_score': float(f1_per_class[i]),
                    'support': int(support_per_class[i])
                }
                
        return results
        
    def evaluate_sentiment_analysis(self, 
                                   y_true: List[str], 
                                   y_pred: List[str],
                                   y_prob: List[List[float]] = None,
                                   model_name: str = "sentiment_model") -> Dict[str, Any]:
        """
        Evaluate sentiment analysis model.
        
        Args:
            y_true (List[str]): True sentiment labels
            y_pred (List[str]): Predicted sentiment labels
            y_prob (List[List[float]]): Prediction probabilities
            model_name (str): Model name
            
        Returns:
            Dict[str, Any]: Sentiment evaluation results
        """
        sentiment_labels = ['negative', 'neutral', 'positive']
        results = self.evaluate_classification(
            y_true, y_pred, y_prob, sentiment_labels, model_name
        )
        
        # Add sentiment-specific metrics
        results['sentiment_specific'] = {
            'positive_precision': results['per_class_metrics'].get('positive', {}).get('precision', 0),
            'negative_precision': results['per_class_metrics'].get('negative', {}).get('precision', 0),
            'neutral_f1': results['per_class_metrics'].get('neutral', {}).get('f1_score', 0)
        }
        
        return results
        
    def evaluate_emotion_detection(self, 
                                 y_true: List[str], 
                                 y_pred: List[str],
                                 y_prob: List[List[float]] = None,
                                 model_name: str = "emotion_model") -> Dict[str, Any]:
        """
        Evaluate emotion detection model.
        
        Args:
            y_true (List[str]): True emotion labels
            y_pred (List[str]): Predicted emotion labels
            y_prob (List[List[float]]): Prediction probabilities
            model_name (str): Model name
            
        Returns:
            Dict[str, Any]: Emotion evaluation results
        """
        emotion_labels = ['joy', 'anger', 'fear', 'sadness', 'surprise', 'neutral']
        results = self.evaluate_classification(
            y_true, y_pred, y_prob, emotion_labels, model_name
        )
        
        # Add emotion-specific metrics
        emotion_groups = {
            'positive_emotions': ['joy', 'surprise'],
            'negative_emotions': ['anger', 'fear', 'sadness'],
            'neutral_emotions': ['neutral']
        }
        
        for group_name, group_emotions in emotion_groups.items():
            group_metrics = []
            for emotion in group_emotions:
                if emotion in results['per_class_metrics']:
                    group_metrics.append(results['per_class_metrics'][emotion]['f1_score'])
                    
            if group_metrics:
                results['emotion_specific'] = results.get('emotion_specific', {})
                results['emotion_specific'][f'{group_name}_avg_f1'] = float(np.mean(group_metrics))
                
        return results
        
    def evaluate_summarization(self, 
                             summaries: List[str],
                             original_texts: List[str],
                             reference_summaries: List[str] = None,
                             model_name: str = "summarization_model") -> Dict[str, Any]:
        """
        Evaluate text summarization model.
        
        Args:
            summaries (List[str]): Generated summaries
            original_texts (List[str]): Original texts
            reference_summaries (List[str]): Reference summaries (optional)
            model_name (str): Model name
            
        Returns:
            Dict[str, Any]: Summarization evaluation results
        """
        if len(summaries) != len(original_texts):
            raise ValueError("Summaries and original texts must have same length")
            
        results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'sample_count': len(summaries),
            'metrics': {}
        }
        
        # Length-based metrics
        original_lengths = [len(text.split()) for text in original_texts]
        summary_lengths = [len(summary.split()) for summary in summaries]
        compression_ratios = [
            s_len / o_len if o_len > 0 else 0 
            for s_len, o_len in zip(summary_lengths, original_lengths)
        ]
        
        results['metrics'].update({
            'avg_original_length': float(np.mean(original_lengths)),
            'avg_summary_length': float(np.mean(summary_lengths)),
            'avg_compression_ratio': float(np.mean(compression_ratios)),
            'compression_ratio_std': float(np.std(compression_ratios))
        })
        
        # Content overlap metrics (simple)
        overlap_scores = []
        for summary, original in zip(summaries, original_texts):
            summary_words = set(summary.lower().split())
            original_words = set(original.lower().split())
            
            if original_words:
                overlap = len(summary_words & original_words) / len(original_words)
                overlap_scores.append(overlap)
            else:
                overlap_scores.append(0)
                
        results['metrics']['avg_content_overlap'] = float(np.mean(overlap_scores))
        
        # Reference-based metrics (if available)
        if reference_summaries:
            if len(reference_summaries) != len(summaries):
                logger.warning("Reference summaries count doesn't match generated summaries")
            else:
                rouge_scores = self._calculate_simple_rouge(summaries, reference_summaries)
                results['metrics'].update(rouge_scores)
                
        return results
        
    def _calculate_simple_rouge(self, 
                               summaries: List[str], 
                               references: List[str]) -> Dict[str, float]:
        """
        Calculate simplified ROUGE scores.
        
        Args:
            summaries (List[str]): Generated summaries
            references (List[str]): Reference summaries
            
        Returns:
            Dict[str, float]: ROUGE scores
        """
        rouge_1_scores = []
        rouge_2_scores = []
        rouge_l_scores = []
        
        for summary, reference in zip(summaries, references):
            # ROUGE-1 (unigram overlap)
            summary_words = summary.lower().split()
            reference_words = reference.lower().split()
            
            if reference_words:
                common_words = set(summary_words) & set(reference_words)
                rouge_1 = len(common_words) / len(set(reference_words))
                rouge_1_scores.append(rouge_1)
            else:
                rouge_1_scores.append(0)
                
            # ROUGE-2 (bigram overlap) - simplified
            summary_bigrams = set(zip(summary_words[:-1], summary_words[1:]))
            reference_bigrams = set(zip(reference_words[:-1], reference_words[1:]))
            
            if reference_bigrams:
                common_bigrams = summary_bigrams & reference_bigrams
                rouge_2 = len(common_bigrams) / len(reference_bigrams)
                rouge_2_scores.append(rouge_2)
            else:
                rouge_2_scores.append(0)
                
            # ROUGE-L (longest common subsequence) - simplified as word overlap
            rouge_l_scores.append(rouge_1_scores[-1])  # Simplified
            
        return {
            'rouge_1': float(np.mean(rouge_1_scores)),
            'rouge_2': float(np.mean(rouge_2_scores)),
            'rouge_l': float(np.mean(rouge_l_scores))
        }
        
    def plot_confusion_matrix(self, 
                            confusion_matrix: List[List[int]], 
                            labels: List[str],
                            model_name: str = "model",
                            save_path: str = None) -> str:
        """
        Plot and save confusion matrix.
        
        Args:
            confusion_matrix (List[List[int]]): Confusion matrix
            labels (List[str]): Class labels
            model_name (str): Model name for title
            save_path (str): Path to save the plot
            
        Returns:
            str: Path to saved plot
        """
        plt.figure(figsize=(10, 8))
        
        # Convert to numpy array
        cm = np.array(confusion_matrix)
        
        # Create heatmap
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )
        
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / f"{model_name}_confusion_matrix.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {save_path}")
        return str(save_path)
        
    def plot_metrics_comparison(self, 
                               results_list: List[Dict[str, Any]],
                               metric_name: str = 'f1_weighted',
                               save_path: str = None) -> str:
        """
        Plot comparison of metrics across different models.
        
        Args:
            results_list (List[Dict[str, Any]]): List of evaluation results
            metric_name (str): Metric to compare
            save_path (str): Path to save the plot
            
        Returns:
            str: Path to saved plot
        """
        model_names = []
        metric_values = []
        
        for result in results_list:
            model_names.append(result.get('model_name', 'Unknown'))
            metric_value = result.get('metrics', {}).get(metric_name, 0)
            metric_values.append(metric_value)
            
        plt.figure(figsize=(12, 6))
        bars = plt.bar(model_names, metric_values, color='skyblue', edgecolor='navy')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
                    
        plt.title(f'Model Comparison - {metric_name.replace("_", " ").title()}')
        plt.xlabel('Models')
        plt.ylabel(metric_name.replace('_', ' ').title())
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / f"models_comparison_{metric_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Metrics comparison plot saved to {save_path}")
        return str(save_path)
        
    def save_results(self, 
                    results: Dict[str, Any], 
                    filename: str = None) -> str:
        """
        Save evaluation results to JSON file.
        
        Args:
            results (Dict[str, Any]): Evaluation results
            filename (str): Output filename
            
        Returns:
            str: Path to saved file
        """
        if filename is None:
            model_name = results.get('model_name', 'model')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{model_name}_evaluation_{timestamp}.json"
            
        save_path = self.output_dir / filename
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Evaluation results saved to {save_path}")
        return str(save_path)
        
    def generate_report(self, 
                       results: Dict[str, Any],
                       include_plots: bool = True) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            results (Dict[str, Any]): Evaluation results
            include_plots (bool): Whether to include plots
            
        Returns:
            str: Path to generated report
        """
        model_name = results.get('model_name', 'model')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f"{model_name}_report_{timestamp}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Evaluation Report: {model_name}\n\n")
            f.write(f"**Generated:** {results.get('timestamp', 'Unknown')}\n")
            f.write(f"**Sample Count:** {results.get('sample_count', 'Unknown')}\n\n")
            
            # Overall metrics
            f.write("## Overall Metrics\n\n")
            metrics = results.get('metrics', {})
            for metric, value in metrics.items():
                if isinstance(value, float):
                    f.write(f"- **{metric.replace('_', ' ').title()}:** {value:.4f}\n")
                else:
                    f.write(f"- **{metric.replace('_', ' ').title()}:** {value}\n")
            f.write("\n")
            
            # Per-class metrics
            if 'per_class_metrics' in results:
                f.write("## Per-Class Metrics\n\n")
                for class_name, class_metrics in results['per_class_metrics'].items():
                    f.write(f"### {class_name}\n")
                    for metric, value in class_metrics.items():
                        f.write(f"- **{metric.replace('_', ' ').title()}:** {value:.4f}\n")
                    f.write("\n")
                    
            # Confusion matrix
            if 'confusion_matrix' in results and include_plots:
                cm_path = self.plot_confusion_matrix(
                    results['confusion_matrix'],
                    results.get('labels', []),
                    model_name
                )
                f.write("## Confusion Matrix\n\n")
                f.write(f"![Confusion Matrix]({cm_path})\n\n")
                
        logger.info(f"Evaluation report generated: {report_path}")
        return str(report_path)


def create_sample_evaluation():
    """Create sample evaluation data for testing."""
    # Sample predictions
    y_true = ['positive', 'negative', 'neutral', 'positive', 'negative'] * 20
    y_pred = ['positive', 'negative', 'neutral', 'neutral', 'negative'] * 20
    
    # Sample probabilities (3 classes)
    np.random.seed(42)
    y_prob = []
    for true_label in y_true:
        if true_label == 'positive':
            probs = [0.1, 0.2, 0.7]  # [neg, neu, pos]
        elif true_label == 'negative':
            probs = [0.7, 0.2, 0.1]
        else:  # neutral
            probs = [0.2, 0.7, 0.1]
            
        # Add some noise
        noise = np.random.normal(0, 0.1, 3)
        probs = np.array(probs) + noise
        probs = np.maximum(probs, 0)  # Ensure non-negative
        probs = probs / np.sum(probs)  # Normalize
        y_prob.append(probs.tolist())
        
    return y_true, y_pred, y_prob


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create sample evaluation
    y_true, y_pred, y_prob = create_sample_evaluation()
    
    # Test evaluator
    evaluator = ModelEvaluator()
    
    # Test sentiment evaluation
    print("=== Testing Sentiment Evaluation ===")
    sentiment_results = evaluator.evaluate_sentiment_analysis(
        y_true, y_pred, y_prob, "test_sentiment_model"
    )
    print(f"Accuracy: {sentiment_results['metrics']['accuracy']:.4f}")
    print(f"F1 Score: {sentiment_results['metrics']['f1_weighted']:.4f}")
    
    # Save results
    results_path = evaluator.save_results(sentiment_results)
    print(f"Results saved to: {results_path}")
    
    # Generate report
    report_path = evaluator.generate_report(sentiment_results)
    print(f"Report generated: {report_path}")
    
    print("Model evaluator test completed successfully!")