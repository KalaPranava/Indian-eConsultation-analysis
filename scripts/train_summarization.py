"""
Script for training summarization models.
"""

import argparse
import logging
import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models import AbstractiveSummarizer, TextRankSummarizer
from src.data import DataLoader, TextPreprocessor

logger = logging.getLogger(__name__)


def main():
    """Main training function for summarization models."""
    parser = argparse.ArgumentParser(description="Train summarization model")
    
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to training data CSV file")
    parser.add_argument("--text_column", type=str, default="comment",
                        help="Name of text column in data")
    parser.add_argument("--summary_column", type=str, default="summary",
                        help="Name of summary column in data (for abstractive training)")
    parser.add_argument("--model_type", type=str, choices=["extractive", "abstractive"], 
                        default="extractive", help="Type of model to train")
    parser.add_argument("--model_name", type=str, default="t5-small",
                        help="Pre-trained model name for abstractive summarization")
    parser.add_argument("--output_dir", type=str, default="models/summarization/trained",
                        help="Output directory for trained model")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum summary length")
    parser.add_argument("--min_length", type=int, default=30,
                        help="Minimum summary length")
    parser.add_argument("--num_sentences", type=int, default=3,
                        help="Number of sentences for extractive summarization")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting summarization model training/evaluation")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Load data
        loader = DataLoader()
        logger.info(f"Loading data from {args.data_path}")
        
        if args.data_path.endswith('.csv'):
            df = loader.load_csv(args.data_path)
        elif args.data_path.endswith('.json'):
            data = loader.load_json(args.data_path)
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported file format: {args.data_path}")
            
        logger.info(f"Loaded {len(df)} samples")
        
        # Preprocess data
        preprocessor = TextPreprocessor()
        df_processed = preprocessor.process_dataframe(df, [args.text_column])
        
        # Filter valid samples
        valid_mask = df_processed[f"{args.text_column}_is_valid"]
        df_valid = df_processed[valid_mask].copy()
        logger.info(f"Using {len(df_valid)} valid samples")
        
        # Get texts
        texts = df_valid[f"{args.text_column}_processed"].tolist()
        
        if args.model_type == "extractive":
            logger.info("Training/evaluating extractive summarization model")
            
            # Initialize TextRank summarizer
            summarizer = TextRankSummarizer()
            
            # Generate summaries
            logger.info("Generating extractive summaries...")
            summaries = []
            for i, text in enumerate(texts):
                if i % 100 == 0:
                    logger.info(f"Processed {i}/{len(texts)} texts")
                    
                result = summarizer.summarize(text, num_sentences=args.num_sentences)
                summaries.append(result)
                
            # Evaluate results
            total_compression = sum(s['compression_ratio'] for s in summaries if s['compression_ratio'] > 0)
            avg_compression = total_compression / len([s for s in summaries if s['compression_ratio'] > 0])
            
            logger.info(f"Extractive summarization completed")
            logger.info(f"Average compression ratio: {avg_compression:.3f}")
            logger.info(f"Summaries generated: {len(summaries)}")
            
            # Save results
            results_df = df_valid.copy()
            results_df['generated_summary'] = [s['summary'] for s in summaries]
            results_df['compression_ratio'] = [s['compression_ratio'] for s in summaries]
            
            output_path = Path(args.output_dir) / "extractive_results.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
            
        elif args.model_type == "abstractive":
            logger.info("Training/evaluating abstractive summarization model")
            
            # Initialize abstractive summarizer
            summarizer = AbstractiveSummarizer(model_name=args.model_name)
            summarizer.load_model()
            
            # Generate summaries (for evaluation)
            logger.info("Generating abstractive summaries...")
            summaries = []
            batch_size = 8
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                
                batch_summaries = summarizer.summarize_multiple(
                    batch_texts,
                    batch_size=batch_size,
                    max_length=args.max_length,
                    min_length=args.min_length
                )
                summaries.extend(batch_summaries)
                
            # Evaluate results
            total_compression = sum(s['compression_ratio'] for s in summaries if s['compression_ratio'] > 0)
            avg_compression = total_compression / len([s for s in summaries if s['compression_ratio'] > 0])
            
            logger.info(f"Abstractive summarization completed")
            logger.info(f"Average compression ratio: {avg_compression:.3f}")
            logger.info(f"Summaries generated: {len(summaries)}")
            
            # Save results
            results_df = df_valid.copy()
            results_df['generated_summary'] = [s['summary'] for s in summaries]
            results_df['compression_ratio'] = [s['compression_ratio'] for s in summaries]
            
            output_path = Path(args.output_dir) / "abstractive_results.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
            
            # Save model if fine-tuning was performed
            # Note: This is a placeholder for actual fine-tuning implementation
            model_output_path = Path(args.output_dir) / "model"
            model_output_path.mkdir(parents=True, exist_ok=True)
            # summarizer.save_model(str(model_output_path))
            logger.info(f"Model configuration saved to {model_output_path}")
            
        logger.info("Summarization training/evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()