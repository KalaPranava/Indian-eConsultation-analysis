"""
Routes package initialization.
"""

from .sentiment import router as sentiment_router
from .summarization import router as summarization_router

__all__ = ['sentiment_router', 'summarization_router']