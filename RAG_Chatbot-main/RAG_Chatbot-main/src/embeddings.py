from langchain_community.embeddings import HuggingFaceEmbeddings
import config
from src.utils import get_logger

logger = get_logger("embeddings")

class EmbeddingModel:
    """
    Embedding model factory responsible for initializing the text embedding model.
    Runs locally using sentence-transformers/all-MiniLM-L6-v2.
    """
    def __init__(self, model_name: str = config.EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.embeddings = None

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Initializes and returns the local HuggingFaceEmbeddings instance.
        Returns:
            HuggingFaceEmbeddings: The initialized embedding model object.
        """
        if self.embeddings is None:
            logger.info(f"Initializing local HuggingFace embeddings model: {self.model_name}")
            try:
                # Runs locally on CPU by default. No API key needed.
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={
                        'normalize_embeddings': True,
                        'batch_size': config.EMBEDDING_BATCH_SIZE,
                    }
                )
                logger.info("Local embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {str(e)}")
                raise RuntimeError(f"Failed to load embedding model: {str(e)}") from e
        
        return self.embeddings

# --- Manual Test Execution Guide ---
# To test this file:
# 1. Run in terminal: python -c "from src.embeddings import EmbeddingModel; model = EmbeddingModel().get_embeddings(); vector = model.embed_query('Hello world'); print('Embedding vector dimensions:', len(vector)); print('First 5 dimensions:', vector[:5])"
