"""
Configuration settings for the Gen AI Interview Project
"""

import os
from pathlib import Path
from typing import Optional

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"
SRC_DIR = PROJECT_ROOT / "src"

# API Keys (set these in your environment or .env file)
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")

# RAG System Configuration
RAG_CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "collection_name": "genai_interview_docs",
    "persist_directory": str(DATA_DIR / "chroma_db"),
    "openai_api_key": OPENAI_API_KEY,
    "anthropic_api_key": ANTHROPIC_API_KEY,
}

# Model configurations for different providers
MODEL_CONFIGS = {
    "openai": {
        "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
        "default": "gpt-3.5-turbo",
        "temperature": 0.1,
        "max_tokens": 1000,
    },
    "anthropic": {
        "models": ["claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"],
        "default": "claude-3-sonnet-20240229",
        "temperature": 0.1,
        "max_tokens": 1000,
    }
}

# Evaluation settings
EVALUATION_CONFIG = {
    "test_queries": [
        "What is machine learning?",
        "Explain the difference between supervised and unsupervised learning",
        "How does RAG work?",
        "What are the main components of a neural network?",
    ],
    "metrics": ["answer_relevance", "context_relevance", "factual_accuracy"],
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": str(DATA_DIR / "logs" / "genai_interview.log"),
}