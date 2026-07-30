import os
from pathlib import Path
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# Project Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = BASE_DIR / os.getenv("VECTOR_STORE_DIR", "vector_store")

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)

# Local Offline Model Configuration
# Qwen/Qwen2.5-0.5B-Instruct is a 950MB state-of-the-art small model that runs fast on CPU.
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")

# RAG Splitting Configurations
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

# Embedding Configurations
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Retrieval Configurations
RETRIEVER_K = int(os.getenv("RETRIEVER_K", 4))  # Number of chunks to retrieve

# Local LLM Generation Configurations
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 512))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", 1.15))

# Embedding batch size (larger batches speed up ingestion of big PDFs on CPU)
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 32))
