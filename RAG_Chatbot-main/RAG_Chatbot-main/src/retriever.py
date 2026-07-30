from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
import config
from src.utils import get_logger

logger = get_logger("retriever")

class ContextRetriever:
    """
    ContextRetriever interfaces with the vector database to search for
    document chunks that match the query semantically.
    """
    def __init__(self, vector_store: FAISS, k: int = config.RETRIEVER_K):
        """
        Initializes retriever with a loaded vector store instance and number of results k.
        """
        self.vector_store = vector_store
        self.k = k
        # Instantiate retriever interface with specified parameters
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k}
        )

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieves matching document chunks for a query.
        Args:
            query (str): The search phrase.
        Returns:
            List[Document]: Top k relevant document chunks.
        """
        logger.info(f"Retrieving top {self.k} contexts for query: '{query}'")
        try:
            results = self.retriever.invoke(query)
            logger.info(f"Retrieved {len(results)} context chunks")
            return results
        except Exception as e:
            logger.error(f"Failed to retrieve contexts: {str(e)}")
            raise RuntimeError(f"Retrieval failed: {str(e)}") from e

# --- Manual Test Execution Guide ---
# To test this file:
# 1. Run in terminal: python -c "from src.embeddings import EmbeddingModel; from src.vectorstore import FAISSVectorStore; from src.retriever import ContextRetriever; embeddings = EmbeddingModel().get_embeddings(); store = FAISSVectorStore(embeddings); db = store.load_store(); retriever = ContextRetriever(db); docs = retriever.retrieve('RAG'); print('Retrieved:', [d.page_content for d in docs])"
