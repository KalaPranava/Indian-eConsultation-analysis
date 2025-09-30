# Indian E-Consultation Analysis Platform

AI-powered sentiment analysis and emotion detection platform for Indian e-consultation feedback with multilingual support (Hindi, English, Code-mixed).

## 🌟 Features

- **Multi-language Support**: Hindi, English, and code-mixed text analysis
- **Comprehensive Analysis**: Sentiment analysis, emotion detection, and text summarization
- **Interactive Dashboards**: Modern React frontend with advanced data visualizations
- **Word Cloud Analytics**: Interactive word filtering and comment exploration
- **Real-time Processing**: Batch processing with progress tracking
- **Export Capabilities**: Download results in CSV format
- **Dark Mode Support**: Professional UI with light/dark themes

## 🚀 Live Demo

- **Frontend**: https://your-project.vercel.app
- **API Documentation**: https://your-service.onrender.com/docs

## 🏗️ Architecture

### Frontend (HackHive Platform)
- **Framework**: Next.js 14 with TypeScript
- **Styling**: Tailwind CSS v4 with Shadcn/ui components
- **State Management**: React hooks
- **Visualizations**: Custom SVG charts and interactive components

### Backend (FastAPI)
- **Framework**: FastAPI with Python 3.10+
- **ML Models**: 
  - DistilBERT for sentiment analysis
  - XLM-RoBERTa for multilingual support
  - Emotion classification models
  - TF-IDF and extractive summarization
- **Processing**: Batch processing with async operations

## 📊 Analysis Capabilities

1. **Sentiment Analysis**: Positive, Negative, Neutral classification
2. **Emotion Detection**: Joy, Sadness, Anger, Fear, Surprise, Disgust, Neutral
3. **Language Detection**: Automatic identification of Hindi, English, Code-mixed
4. **Text Summarization**: Extractive and abstractive summarization
5. **Confidence Scoring**: ML model confidence for all predictions

## 🛠️ Local Development

### Prerequisites
- Python 3.10+
- Node.js 18+
- Git

### Backend Setup
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/indian-econsultation-analysis.git
cd indian-econsultation-analysis

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Start API server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend Setup
```bash
# Navigate to frontend directory
cd hackhive-platform

# Install dependencies
npm install

# Start development server
npm run dev
```

Access the application at `http://localhost:3000`

## 📁 Project Structure

```
├── api/                    # FastAPI backend
│   ├── main.py            # Main API application
│   └── schemas.py         # Pydantic models
├── hackhive-platform/     # Next.js frontend
│   ├── components/        # React components
│   ├── app/              # Next.js 14 app directory
│   └── lib/              # Utility functions
├── src/                   # Core ML modules
│   ├── inference/        # ML inference pipeline
│   ├── models/           # Model configurations
│   └── training/         # Training scripts
├── scripts/              # Utility scripts
└── docs/                 # Documentation
```

## 🌐 Deployment

### Production Deployment
1. **Backend**: Deploy on Render.com or similar platform
2. **Frontend**: Deploy on Vercel with environment variables
3. **Environment Variables**: 
   - `NEXT_PUBLIC_API_BASE`: Backend API URL

### Docker Deployment
```bash
# Backend
docker build -f Dockerfile.backend -t econsult-api .
docker run -p 8000:8000 econsult-api

# Frontend
docker build -f Dockerfile.frontend -t econsult-frontend .
docker run -p 3000:3000 econsult-frontend
```

## 🔧 Configuration

### Environment Variables
- `NEXT_PUBLIC_API_BASE`: Backend API base URL
- `LOG_LEVEL`: Logging level (info, debug, error)
- `MODEL_CACHE_DIR`: Directory for caching ML models

## 📈 Performance

- **Processing Speed**: ~10-50 comments per batch
- **Model Loading**: First request may take 2-3 minutes (model download)
- **Memory Requirements**: 1-2GB RAM for ML models
- **Supported File Formats**: CSV, JSON, TXT

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hugging Face for transformer models
- Next.js and React community
- FastAPI framework
- Tailwind CSS and Shadcn/ui components

## 📞 Support

For support, please open an issue on GitHub or contact [your-email@example.com]

---

Built with ❤️ for improving healthcare feedback analysis in India

A comprehensive modular Python project for analyzing sentiment and generating summaries from Indian e-consultation comments, with support for Hindi, English, and code-mixed content.

## Features

- **Multi-language Support**: Process Hindi, English, and code-mixed comments
- **Sentiment Analysis**: Using IndicBERT/mBERT for accurate sentiment classification
- **Text Summarization**: Both extractive (TextRank) and abstractive (T5/BART) approaches
- **Emotion Detection**: Secondary classification for detailed emotion analysis
- **Interactive Visualization**: Dashboards with filtering and word clouds
- **API Deployment**: FastAPI backend with Docker containerization
- **Database Integration**: PostgreSQL/MongoDB support for data persistence

## Project Structure

```
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py          # Data loading utilities
│   │   └── preprocessor.py    # Text preprocessing and cleaning
│   ├── models/
│   │   ├── __init__.py
│   │   ├── sentiment.py       # Sentiment analysis models
│   │   ├── summarization.py   # Summarization models
│   │   └── emotion.py         # Emotion detection models
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py         # Model training utilities
│   │   └── evaluator.py       # Model evaluation metrics
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predictor.py       # Inference pipeline
│   └── visualization/
│       ├── __init__.py
│       ├── dashboard.py       # Interactive dashboard
│       └── charts.py          # Visualization utilities
├── api/
│   ├── __init__.py
│   ├── main.py               # FastAPI application
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── sentiment.py      # Sentiment analysis endpoints
│   │   └── summarization.py  # Summarization endpoints
│   └── schemas.py            # Pydantic models
├── scripts/
│   ├── train_sentiment.py    # Training script for sentiment analysis
│   ├── train_summarization.py # Training script for summarization
│   ├── evaluate_models.py    # Model evaluation script
│   └── serve_api.py          # API server script
├── data/
│   ├── raw/                  # Raw data files
│   ├── processed/            # Processed data
│   └── sample/               # Sample datasets
├── models/
│   ├── sentiment/            # Trained sentiment models
│   ├── summarization/        # Trained summarization models
│   └── emotion/              # Trained emotion models
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_models.py
│   └── test_api.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
├── setup.py
└── config.yaml
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd indian-econsultation-analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required models:
```bash
python scripts/download_models.py
```

## Quick Start

### 1. Data Preparation
```python
from src.data.loader import DataLoader
from src.data.preprocessor import TextPreprocessor

# Load data
loader = DataLoader()
data = loader.load_csv("data/sample/comments.csv")

# Preprocess
preprocessor = TextPreprocessor()
processed_data = preprocessor.process(data)
```

### 2. Sentiment Analysis
```python
from src.models.sentiment import SentimentAnalyzer

analyzer = SentimentAnalyzer(model_name="ai4bharat/indic-bert")
results = analyzer.predict(processed_data)
```

### 3. Text Summarization
```python
from src.models.summarization import Summarizer

summarizer = Summarizer(method="textrank")
summaries = summarizer.generate_summary(processed_data)
```

### 4. Run API Server
```bash
python scripts/serve_api.py
```

## Training Custom Models

### Sentiment Analysis
```bash
python scripts/train_sentiment.py --data_path data/processed/sentiment_data.csv --model_name indic-bert
```

### Summarization
```bash
python scripts/train_summarization.py --data_path data/processed/summary_data.csv --model_name t5-small
```

## API Usage

The FastAPI server provides REST endpoints for sentiment analysis and summarization:

- `POST /sentiment/analyze` - Analyze sentiment of comments
- `POST /summarization/generate` - Generate summaries
- `GET /health` - Health check endpoint

Example request:
```python
import requests

response = requests.post(
    "http://localhost:8000/sentiment/analyze",
    json={"comments": ["यह बहुत अच्छी सेवा है", "This service needs improvement"]}
)
```

## Docker Deployment

```bash
# Build the image
docker build -f docker/Dockerfile -t econsultation-analyzer .

# Run with docker-compose
docker-compose -f docker/docker-compose.yml up
```

## Configuration

Edit `config.yaml` to customize:
- Model configurations
- Database settings
- API parameters
- Visualization options

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.