"""
Data loading utilities for the e-consultation sentiment analysis project.
"""

import pandas as pd
import json
import os
from typing import List, Dict, Any, Optional, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DataLoader:
    """
    A utility class for loading data from various file formats.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the DataLoader.
        
        Args:
            data_dir (str): Base directory for data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            **kwargs: Additional arguments for pandas.read_csv
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            full_path = self.data_dir / file_path if not os.path.isabs(file_path) else file_path
            logger.info(f"Loading CSV data from {full_path}")
            
            default_kwargs = {
                'encoding': 'utf-8',
                'on_bad_lines': 'skip'
            }
            default_kwargs.update(kwargs)
            
            data = pd.read_csv(full_path, **default_kwargs)
            logger.info(f"Successfully loaded {len(data)} rows from CSV")
            return data
            
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {str(e)}")
            raise
            
    def load_json(self, file_path: str) -> Union[Dict, List]:
        """
        Load data from a JSON file.
        
        Args:
            file_path (str): Path to the JSON file
            
        Returns:
            Union[Dict, List]: Loaded JSON data
        """
        try:
            full_path = self.data_dir / file_path if not os.path.isabs(file_path) else file_path
            logger.info(f"Loading JSON data from {full_path}")
            
            with open(full_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            logger.info(f"Successfully loaded JSON data")
            return data
            
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {str(e)}")
            raise
            
    def load_jsonl(self, file_path: str) -> List[Dict]:
        """
        Load data from a JSONL (JSON Lines) file.
        
        Args:
            file_path (str): Path to the JSONL file
            
        Returns:
            List[Dict]: List of JSON objects
        """
        try:
            full_path = self.data_dir / file_path if not os.path.isabs(file_path) else file_path
            logger.info(f"Loading JSONL data from {full_path}")
            
            data = []
            with open(full_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data.append(json.loads(line.strip()))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                        continue
                        
            logger.info(f"Successfully loaded {len(data)} records from JSONL")
            return data
            
        except Exception as e:
            logger.error(f"Error loading JSONL file {file_path}: {str(e)}")
            raise
            
    def load_text(self, file_path: str, split_lines: bool = True) -> Union[str, List[str]]:
        """
        Load data from a plain text file.
        
        Args:
            file_path (str): Path to the text file
            split_lines (bool): Whether to split text into lines
            
        Returns:
            Union[str, List[str]]: Text content or list of lines
        """
        try:
            full_path = self.data_dir / file_path if not os.path.isabs(file_path) else file_path
            logger.info(f"Loading text data from {full_path}")
            
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if split_lines:
                content = [line.strip() for line in content.split('\n') if line.strip()]
                logger.info(f"Successfully loaded {len(content)} lines from text file")
            else:
                logger.info(f"Successfully loaded text file ({len(content)} characters)")
                
            return content
            
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {str(e)}")
            raise
            
    def save_csv(self, data: pd.DataFrame, file_path: str, **kwargs) -> None:
        """
        Save data to a CSV file.
        
        Args:
            data (pd.DataFrame): Data to save
            file_path (str): Output file path
            **kwargs: Additional arguments for pandas.to_csv
        """
        try:
            full_path = self.data_dir / file_path if not os.path.isabs(file_path) else file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            default_kwargs = {
                'index': False,
                'encoding': 'utf-8'
            }
            default_kwargs.update(kwargs)
            
            data.to_csv(full_path, **default_kwargs)
            logger.info(f"Successfully saved {len(data)} rows to {full_path}")
            
        except Exception as e:
            logger.error(f"Error saving CSV file {file_path}: {str(e)}")
            raise
            
    def save_json(self, data: Union[Dict, List], file_path: str, indent: int = 2) -> None:
        """
        Save data to a JSON file.
        
        Args:
            data (Union[Dict, List]): Data to save
            file_path (str): Output file path
            indent (int): JSON indentation
        """
        try:
            full_path = self.data_dir / file_path if not os.path.isabs(file_path) else file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=indent)
                
            logger.info(f"Successfully saved JSON data to {full_path}")
            
        except Exception as e:
            logger.error(f"Error saving JSON file {file_path}: {str(e)}")
            raise
            
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get information about a data file.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            Dict[str, Any]: File information
        """
        try:
            full_path = self.data_dir / file_path if not os.path.isabs(file_path) else file_path
            
            if not full_path.exists():
                return {"exists": False}
                
            stat = full_path.stat()
            
            info = {
                "exists": True,
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified_time": stat.st_mtime,
                "extension": full_path.suffix
            }
            
            # Try to get row count for CSV files
            if full_path.suffix.lower() == '.csv':
                try:
                    df = pd.read_csv(full_path, nrows=0)
                    info["columns"] = list(df.columns)
                    info["num_columns"] = len(df.columns)
                    
                    # Count rows efficiently
                    with open(full_path, 'r', encoding='utf-8') as f:
                        info["num_rows"] = sum(1 for _ in f) - 1  # Subtract header
                        
                except Exception:
                    pass
                    
            return info
            
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {str(e)}")
            return {"exists": False, "error": str(e)}


def create_sample_data():
    """
    Create sample data files for testing and demonstration.
    """
    loader = DataLoader()
    
    # Sample comments data
    sample_comments = [
        {
            "id": 1,
            "comment": "यह सेवा बहुत अच्छी है। डॉक्टर ने बहुत अच्छी सलाह दी।",
            "language": "hi",
            "sentiment": "positive"
        },
        {
            "id": 2,
            "comment": "The online consultation was very helpful. Thank you!",
            "language": "en",
            "sentiment": "positive"
        },
        {
            "id": 3,
            "comment": "Service अच्छी है but wait time बहुत ज्यादा है।",
            "language": "mixed",
            "sentiment": "neutral"
        },
        {
            "id": 4,
            "comment": "I am not satisfied with the consultation. बहुत disappointing था।",
            "language": "mixed",
            "sentiment": "negative"
        },
        {
            "id": 5,
            "comment": "डॉक्टर बहुत रूखे थे और properly examine नहीं किया।",
            "language": "mixed",
            "sentiment": "negative"
        }
    ]
    
    # Save as CSV
    df = pd.DataFrame(sample_comments)
    loader.save_csv(df, "sample/comments.csv")
    
    # Save as JSON
    loader.save_json(sample_comments, "sample/comments.json")
    
    logger.info("Sample data files created successfully")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_sample_data()