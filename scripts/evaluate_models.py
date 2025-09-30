"""
Script for evaluating trained models.
"""

import argparse
import logging
import sys
import json
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training import ModelEvaluator
from src.models import SentimentAnalyzer, EmotionDetector, HybridSummarizer
from src.data import DataLoader, TextPreprocessor

logger = logging.getLogger(__name__)


def evaluate_sentiment_model(model_path, data_path, text_column, label_column, output_dir):
    """Evaluate sentiment analysis model."""
    logger.info("Evaluating sentiment analysis model")
    
    # Load data
    loader = DataLoader()
    if data_path.endswith('.csv'):
        df = loader.load_csv(data_path)
    else:
        data = loader.load_json(data_path)
        df = pd.DataFrame(data)
        
    # Preprocess
    preprocessor = TextPreprocessor()
    df_processed = preprocessor.process_dataframe(df, [text_column])
    
    # Filter valid samples
    valid_mask = df_processed[f"{text_column}_is_valid"]
    df_valid = df_processed[valid_mask].copy()
    
    texts = df_valid[f"{text_column}_processed"].tolist()
    labels = df_valid[label_column].tolist()
    
    # Load model
    analyzer = SentimentAnalyzer()
    if Path(model_path).exists():
        analyzer.load_model(model_path)
    else:
        analyzer.load_model()  # Load default model
        
    # Predict
    predictions = analyzer.predict(texts, return_probabilities=True)
    predicted_labels = [pred['label'] for pred in predictions]
    probabilities = [list(pred['scores'].values()) for pred in predictions]
    
    # Evaluate
    evaluator = ModelEvaluator(output_dir)
    results = evaluator.evaluate_sentiment_analysis(
        labels, predicted_labels, probabilities, "sentiment_model"
    )
    
    # Save results
    results_path = evaluator.save_results(results)
    
    # Generate report
    report_path = evaluator.generate_report(results)
    
    logger.info(f"Sentiment evaluation completed")
    logger.info(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    logger.info(f"F1 Score: {results['metrics']['f1_weighted']:.4f}")
    logger.info(f"Results saved to: {results_path}")
    logger.info(f"Report saved to: {report_path}")
    
    return results


def evaluate_emotion_model(model_path, data_path, text_column, label_column, output_dir):
    """Evaluate emotion detection model."""
    logger.info("Evaluating emotion detection model")
    
    # Load data
    loader = DataLoader()
    if data_path.endswith('.csv'):
        df = loader.load_csv(data_path)
    else:
        data = loader.load_json(data_path)
        df = pd.DataFrame(data)
        
    # Preprocess
    preprocessor = TextPreprocessor()
    df_processed = preprocessor.process_dataframe(df, [text_column])
    
    # Filter valid samples
    valid_mask = df_processed[f"{text_column}_is_valid"]
    df_valid = df_processed[valid_mask].copy()
    
    texts = df_valid[f"{text_column}_processed"].tolist()
    labels = df_valid[label_column].tolist()
    
    # Load model
    detector = EmotionDetector()
    if Path(model_path).exists():
        detector.load_model(model_path)
    else:
        detector.load_model()  # Load default model
        
    # Predict
    predictions = detector.predict(texts, top_k=6, threshold=0.0)
    predicted_labels = [pred['primary_emotion'] for pred in predictions]
    probabilities = [list(pred['emotion_scores'].values()) for pred in predictions]
    
    # Evaluate
    evaluator = ModelEvaluator(output_dir)
    results = evaluator.evaluate_emotion_detection(
        labels, predicted_labels, probabilities, "emotion_model"
    )
    
    # Save results
    results_path = evaluator.save_results(results)
    
    # Generate report
    report_path = evaluator.generate_report(results)
    
    logger.info(f"Emotion evaluation completed")
    logger.info(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    logger.info(f"F1 Score: {results['metrics']['f1_weighted']:.4f}")
    logger.info(f"Results saved to: {results_path}")
    logger.info(f"Report saved to: {report_path}")
    
    return results


def evaluate_summarization_model(data_path, text_column, summary_column, output_dir, method="extractive"):
    """Evaluate summarization model."""
    logger.info(f"Evaluating {method} summarization")
    
    # Load data
    loader = DataLoader()
    if data_path.endswith('.csv'):
        df = loader.load_csv(data_path)
    else:
        data = loader.load_json(data_path)
        df = pd.DataFrame(data)
        
    # Preprocess
    preprocessor = TextPreprocessor()
    df_processed = preprocessor.process_dataframe(df, [text_column])
    
    # Filter valid samples
    valid_mask = df_processed[f"{text_column}_is_valid"]
    df_valid = df_processed[valid_mask].copy()
    
    texts = df_valid[f"{text_column}_processed"].tolist()
    reference_summaries = df_valid[summary_column].tolist() if summary_column in df_valid.columns else None
    
    # Initialize summarizer
    summarizer = HybridSummarizer()
    
    # Generate summaries
    logger.info("Generating summaries...")
    generated_summaries = []
    for text in texts:
        result = summarizer.summarize(text, method=method)
        generated_summaries.append(result['summary'])
        
    # Evaluate
    evaluator = ModelEvaluator(output_dir)
    results = evaluator.evaluate_summarization(
        generated_summaries, texts, reference_summaries, f"{method}_summarization"
    )
    
    # Save results
    results_path = evaluator.save_results(results)
    
    logger.info(f"Summarization evaluation completed")
    logger.info(f"Average compression ratio: {results['metrics']['avg_compression_ratio']:.4f}")
    logger.info(f"Average content overlap: {results['metrics']['avg_content_overlap']:.4f}")
    logger.info(f"Results saved to: {results_path}")
    
    return results


def compare_models(results_dir, model_type):
    """Compare multiple model results."""
    logger.info(f"Comparing {model_type} models")
    
    # Find result files
    results_dir = Path(results_dir)
    pattern = f"{model_type}_*.json"
    result_files = list(results_dir.glob(pattern))
    
    if not result_files:
        logger.warning(f"No result files found matching pattern: {pattern}")
        return
        
    logger.info(f"Found {len(result_files)} result files")
    
    # Load and compare results
    evaluator = ModelEvaluator(results_dir)
    comparison = evaluator.compare_models([str(f) for f in result_files])
    
    # Save comparison
    comparison_path = results_dir / f"{model_type}_comparison.json"
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
        
    # Print summary
    if comparison['best_model']:
        best = comparison['best_model']
        logger.info(f"Best model: {best['model_name']}")
        logger.info(f"F1 Score: {best.get('f1_score', 'N/A')}")
        logger.info(f"Accuracy: {best.get('accuracy', 'N/A')}")
        
    logger.info(f"Comparison saved to: {comparison_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    
    parser.add_argument("--model_type", type=str, 
                        choices=["sentiment", "emotion", "summarization", "all"],
                        default="all", help="Type of model to evaluate")
    parser.add_argument("--model_path", type=str, default="",
                        help="Path to trained model (optional)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to evaluation data")
    parser.add_argument("--text_column", type=str, default="comment",
                        help="Name of text column")
    parser.add_argument("--label_column", type=str, default="label",
                        help="Name of label column")
    parser.add_argument("--summary_column", type=str, default="summary",
                        help="Name of summary column (for summarization)")
    parser.add_argument("--output_dir", type=str, default="results/evaluation",
                        help="Output directory for results")
    parser.add_argument("--summarization_method", type=str, default="extractive",
                        choices=["extractive", "abstractive", "hybrid"],
                        help="Summarization method to evaluate")
    parser.add_argument("--compare", action="store_true",
                        help="Compare multiple models")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting model evaluation")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.compare:
            # Compare existing results
            if args.model_type != "all":
                compare_models(output_dir, args.model_type)
            else:
                for model_type in ["sentiment", "emotion"]:
                    try:
                        compare_models(output_dir, model_type)
                    except Exception as e:
                        logger.warning(f"Could not compare {model_type} models: {e}")
        else:
            # Evaluate models
            if args.model_type == "sentiment" or args.model_type == "all":
                try:
                    evaluate_sentiment_model(
                        args.model_path, args.data_path, args.text_column, 
                        args.label_column, str(output_dir)
                    )
                except Exception as e:
                    logger.error(f"Sentiment evaluation failed: {e}")
                    
            if args.model_type == "emotion" or args.model_type == "all":
                try:
                    evaluate_emotion_model(
                        args.model_path, args.data_path, args.text_column,
                        args.label_column, str(output_dir)
                    )
                except Exception as e:
                    logger.error(f"Emotion evaluation failed: {e}")
                    
            if args.model_type == "summarization" or args.model_type == "all":
                try:
                    evaluate_summarization_model(
                        args.data_path, args.text_column, args.summary_column,
                        str(output_dir), args.summarization_method
                    )
                except Exception as e:
                    logger.error(f"Summarization evaluation failed: {e}")
                    
        logger.info("Model evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()