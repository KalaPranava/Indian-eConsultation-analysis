"""
Project initialization and setup script.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


def create_directory_structure():
    """Create the required directory structure."""
    directories = [
        "data/raw",
        "data/processed", 
        "data/sample",
        "models/sentiment",
        "models/summarization", 
        "models/emotion",
        "logs",
        "results/evaluation",
        "results/training",
        "tests",
        "docs",
        ".github"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def install_dependencies():
    """Install Python dependencies."""
    logger.info("Installing Python dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False
    return True


def download_nltk_data():
    """Download required NLTK data."""
    logger.info("Downloading NLTK data...")
    
    import nltk
    
    datasets = [
        'punkt', 'stopwords', 'vader_lexicon',
        'averaged_perceptron_tagger', 'wordnet'
    ]
    
    for dataset in datasets:
        try:
            nltk.download(dataset, quiet=True)
            logger.info(f"Downloaded NLTK dataset: {dataset}")
        except Exception as e:
            logger.warning(f"Could not download {dataset}: {e}")


def check_spacy_model():
    """Check and install spaCy English model if needed."""
    try:
        import spacy
        spacy.load("en_core_web_sm")
        logger.info("spaCy English model is available")
    except OSError:
        logger.info("Installing spaCy English model...")
        try:
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            logger.info("spaCy English model installed successfully")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Could not install spaCy model: {e}")


def create_sample_data():
    """Create sample data for testing."""
    logger.info("Creating sample data...")
    
    try:
        from scripts.create_sample_data import save_sample_data
        save_sample_data()
        logger.info("Sample data created successfully")
    except Exception as e:
        logger.error(f"Failed to create sample data: {e}")


def validate_configuration():
    """Validate the configuration file."""
    config_path = "config.yaml"
    
    if not Path(config_path).exists():
        logger.warning(f"Configuration file {config_path} not found")
        return False
        
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Check required sections
        required_sections = ['models', 'training', 'preprocessing']
        for section in required_sections:
            if section not in config:
                logger.warning(f"Missing configuration section: {section}")
                return False
                
        logger.info("Configuration file validated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error validating configuration: {e}")
        return False


def test_imports():
    """Test if all required packages can be imported."""
    logger.info("Testing package imports...")
    
    packages = [
        'torch', 'transformers', 'datasets', 'numpy', 'pandas',
        'sklearn', 'matplotlib', 'seaborn', 'plotly', 'wordcloud',
        'fastapi', 'uvicorn', 'pydantic', 'spacy', 'nltk',
        'networkx', 'langdetect', 'streamlit', 'requests',
        'beautifulsoup4', 'tqdm', 'pytest'
    ]
    
    failed_imports = []
    
    for package in packages:
        try:
            __import__(package)
            logger.debug(f"Successfully imported: {package}")
        except ImportError as e:
            failed_imports.append(package)
            logger.warning(f"Failed to import {package}: {e}")
    
    if failed_imports:
        logger.error(f"Failed to import packages: {failed_imports}")
        return False
    else:
        logger.info("All required packages imported successfully")
        return True


def create_env_file():
    """Create a sample .env file."""
    env_content = """# Environment variables for Indian E-Consultation Analysis

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1
LOG_LEVEL=info

# Model Configuration
CONFIG_PATH=config.yaml

# Database Configuration (optional)
DATABASE_URL=postgresql://admin:password@localhost:5432/econsultation_db
REDIS_URL=redis://localhost:6379

# Hugging Face Hub (optional, for model downloads)
# HUGGINGFACE_HUB_TOKEN=your_token_here

# CUDA Configuration (if using GPU)
# CUDA_VISIBLE_DEVICES=0
"""
    
    env_path = ".env.example"
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    logger.info(f"Created example environment file: {env_path}")


def create_gitignore():
    """Create a comprehensive .gitignore file."""
    gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Virtual environments
.env
.venv/
env/
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Project specific
logs/
*.log
models/*/
!models/README.md
data/raw/
data/processed/
!data/sample/
results/
.pytest_cache/
.mypy_cache/

# Jupyter Notebook
.ipynb_checkpoints

# Environments
.env
.env.local
.env.development.local
.env.test.local
.env.production.local

# SSL certificates
ssl/
*.pem
*.crt
*.key

# Docker
.dockerignore

# Temporary files
tmp/
temp/
*.tmp
*.temp
"""
    
    with open(".gitignore", 'w') as f:
        f.write(gitignore_content)
    
    logger.info("Created .gitignore file")


def main():
    """Main setup function."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting project setup...")
    
    # Create directory structure
    create_directory_structure()
    
    # Create supporting files
    create_env_file()
    create_gitignore()
    
    # Install dependencies
    if not install_dependencies():
        logger.error("Failed to install dependencies. Please check requirements.txt")
        return
    
    # Download NLTK data
    download_nltk_data()
    
    # Check spaCy model
    check_spacy_model()
    
    # Test imports
    if not test_imports():
        logger.warning("Some packages failed to import. Please check your installation.")
    
    # Validate configuration
    validate_configuration()
    
    # Create sample data
    create_sample_data()
    
    logger.info("Project setup completed successfully!")
    logger.info("""
Next steps:
1. Review and update config.yaml as needed
2. Copy .env.example to .env and update values
3. Run sample training: python scripts/train_sentiment.py --data_path data/sample/sentiment_data.csv --create_sample
4. Start the API server: python scripts/serve_api.py
5. Visit http://localhost:8000/docs for API documentation

For more information, see README.md
""")


if __name__ == "__main__":
    main()