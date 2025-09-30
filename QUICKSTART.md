# Quick Start Guide

## 🚀 Getting Started (Windows)

### Option 1: Automated Setup
```bash
# Run the initialization script
init_project.bat
```

### Option 2: Manual Setup
```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run project setup
python scripts\setup_project.py

# 4. Create sample data
python scripts\create_sample_data.py

# 5. Start the API server
python scripts\serve_api.py
```

## 📝 API Testing

Once the server is running, visit:
- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

### Sample API Requests

**Sentiment Analysis:**
```bash
curl -X POST "http://localhost:8000/analyze/sentiment" \
     -H "Content-Type: application/json" \
     -d '{"text": "मुझे डॉक्टर से बहुत अच्छी सलाह मिली"}'
```

**Text Summarization:**
```bash
curl -X POST "http://localhost:8000/analyze/summarize" \
     -H "Content-Type: application/json" \
     -d '{"text": "Your long text here...", "method": "textrank", "max_length": 100}'
```

**Emotion Detection:**
```bash
curl -X POST "http://localhost:8000/analyze/emotion" \
     -H "Content-Type: application/json" \
     -d '{"text": "I am very happy with the consultation"}'
```

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# API will be available at http://localhost:8000
```

## 🧪 Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest --cov=src tests/
```

## 📊 Training Models

```bash
# Train sentiment model
python scripts\train_sentiment.py --data_path data\sample\sentiment_data.csv

# Train emotion model  
python scripts\train_emotion.py --data_path data\sample\emotion_data.csv

# Evaluate models
python scripts\evaluate_models.py
```

## 🛠️ Development

### Project Structure
```
├── src/                 # Source code
│   ├── data/           # Data loading and preprocessing
│   ├── models/         # ML models
│   ├── training/       # Training utilities
│   └── inference/      # Inference pipeline
├── api/                # FastAPI application
├── scripts/            # Utility scripts
├── docker/             # Docker configuration
└── data/               # Data storage
```

### Configuration

Edit `config.yaml` to customize:
- Model parameters
- Training settings
- API configuration
- Preprocessing options

### Environment Variables

Copy `.env.example` to `.env` and configure:
```env
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=info
```

## 📈 Monitoring

Check logs in the `logs/` directory:
- `api.log`: API server logs
- `training.log`: Training logs  
- `inference.log`: Inference logs

## 🔧 Troubleshooting

**Common Issues:**

1. **Import Errors**: Run `python scripts\setup_project.py` to install dependencies
2. **Model Download Issues**: Check internet connection and HuggingFace access
3. **Port Already in Use**: Change API_PORT in `.env` file
4. **CUDA Issues**: Set `CUDA_VISIBLE_DEVICES=-1` for CPU-only mode

**Getting Help:**

1. Check logs in `logs/` directory
2. Review API documentation at `/docs` endpoint
3. Validate configuration with `python -c "import yaml; print(yaml.safe_load(open('config.yaml')))"`

## 🎯 Next Steps

1. **Data Collection**: Replace sample data with real e-consultation comments
2. **Model Fine-tuning**: Train models on your specific domain data
3. **Deployment**: Deploy to cloud platforms (AWS, GCP, Azure)
4. **Monitoring**: Set up monitoring and logging in production
5. **Scaling**: Use Redis for caching and PostgreSQL for data storage