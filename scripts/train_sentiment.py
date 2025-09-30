"""
Script for training sentiment analysis models.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training import ModelTrainer
from src.data import create_sample_data

logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train sentiment analysis model")
    
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to training data CSV file")
    parser.add_argument("--text_column", type=str, default="comment",
                        help="Name of text column in data")
    parser.add_argument("--label_column", type=str, default="sentiment",
                        help="Name of label column in data")
    parser.add_argument("--model_name", type=str, default="ai4bharat/indic-bert",
                        help="Pre-trained model name")
    parser.add_argument("--output_dir", type=str, default="models/sentiment/trained",
                        help="Output directory for trained model")
    parser.add_argument("--config_path", type=str, default="config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--validation_split", type=float, default=0.2,
                        help="Validation split ratio")
    parser.add_argument("--create_sample", action="store_true",
                        help="Create sample data if data file doesn't exist")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting sentiment analysis model training")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Create sample data if requested and file doesn't exist
        if args.create_sample and not Path(args.data_path).exists():
            logger.info("Creating sample data...")
            create_sample_data()
            logger.info("Sample data created")
        
        # Initialize trainer
        trainer = ModelTrainer(config_path=args.config_path)
        
        # Update configuration with command line arguments
        sentiment_config = trainer.config.setdefault('models', {}).setdefault('sentiment', {})
        sentiment_config.update({
            'name': args.model_name,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate
        })
        
        training_config = trainer.config.setdefault('training', {})
        training_config['validation_split'] = args.validation_split
        
        # Train model
        logger.info("Starting model training...")
        results = trainer.train_sentiment_model(
            data_path=args.data_path,
            text_column=args.text_column,
            label_column=args.label_column,
            model_name=args.model_name,
            output_dir=args.output_dir
        )
        
        # Print results
        logger.info("Training completed successfully!")
        logger.info(f"Training results: {results['training_results']}")
        
        if 'test_results' in results and results['test_results']:
            test_results = results['test_results']
            logger.info(f"Test accuracy: {test_results.get('accuracy', 'N/A'):.4f}")
            logger.info(f"Test F1 score: {test_results.get('f1_weighted', 'N/A'):.4f}")
            
        logger.info(f"Model saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()