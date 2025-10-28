"""Configuration management for Legal AI Chat Assistant"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration"""
    
    # Base paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    DOCUMENTS_DIR = DATA_DIR / "documents"
    VECTOR_STORE_DIR = DATA_DIR / "vector_store"
    
    # Model configurations (all free and open-source)
    LLM_MODEL = os.getenv("LLM_MODEL", "google/flan-t5-large")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    SUMMARIZATION_MODEL = os.getenv("SUMMARIZATION_MODEL", "facebook/bart-large-cnn")
    
    # RAG settings
    MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", 5))
    
    # Voice settings
    WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large
    TTS_LANG = "en"
    
    # LLM generation settings
    MAX_NEW_TOKENS = 512
    TEMPERATURE = 0.3  # Lower for more focused legal responses
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.DOCUMENTS_DIR.mkdir(exist_ok=True)
        cls.VECTOR_STORE_DIR.mkdir(exist_ok=True)
        
    @classmethod
    def get_vector_store_path(cls):
        """Get the vector store path as string"""
        return str(cls.VECTOR_STORE_DIR)
    
    @classmethod
    def get_documents_path(cls):
        """Get the documents path as string"""
        return str(cls.DOCUMENTS_DIR)
