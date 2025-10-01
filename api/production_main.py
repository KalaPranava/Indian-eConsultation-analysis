"""
Production API - Automatically switches between heavy and lightweight models based on environment
Uses lightweight models in production (Render 512MB) and full models in development
"""
import os
import sys
import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is on sys.path so 'scripts' and other top-level modules import correctly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Detect environment, memory constraints and explicit model profile
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
MEMORY_LIMIT_MB = int(os.getenv("MEMORY_LIMIT_MB", "1024"))
MODEL_PROFILE = os.getenv("MODEL_PROFILE")  # 'lightweight' | 'heavy' | None

if MODEL_PROFILE:
    USE_LIGHTWEIGHT = MODEL_PROFILE.lower() == "lightweight"
else:
    # Auto mode: pick lightweight only if memory budget is small (<=512MB)
    USE_LIGHTWEIGHT = MEMORY_LIMIT_MB <= 512

logger.info(f"🌍 Environment: {ENVIRONMENT}")
logger.info(f"💾 Memory limit: {MEMORY_LIMIT_MB}MB")
logger.info(f"🧩 Model profile (requested): {MODEL_PROFILE or 'auto'}")
logger.info(f"⚡ Using lightweight models: {USE_LIGHTWEIGHT}")

if USE_LIGHTWEIGHT:
    logger.info("🪶 Loading LIGHTWEIGHT model stack (VADER / TextBlob / Lexicons)...")
    from scripts.lightweight_ml_api import *  # noqa: F401,F403
else:
    logger.info("🚀 Loading FULL model stack (XLM-R / DistilBERT / Emotion)...")
    from scripts.working_ml_api import *  # noqa: F401,F403

# Override CORS for production
if ENVIRONMENT == "production":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://hackhive-platform.vercel.app",
            "https://*.vercel.app",
            "https://your-frontend-domain.com"  # Replace with your actual domain
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    logger.info("✅ Production CORS configured")

# Add environment info to root endpoint
@app.get("/info")
async def environment_info():
    return {
        "environment": ENVIRONMENT,
        "memory_limit_mb": MEMORY_LIMIT_MB,
        "model_profile": MODEL_PROFILE or "auto",
        "using_lightweight": USE_LIGHTWEIGHT,
        "model_type": "lightweight" if USE_LIGHTWEIGHT else "full_ml",
        "estimated_memory_usage": (f"~{estimate_memory_usage()}MB" if USE_LIGHTWEIGHT else "~800-1200MB"),
        "selection_mode": "forced" if MODEL_PROFILE else "auto"
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info" if ENVIRONMENT == "production" else "debug"
    )