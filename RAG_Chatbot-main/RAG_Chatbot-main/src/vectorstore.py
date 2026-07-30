import os
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import config
from src.utils import get_logger

logger = get_logger("vectorstore")

class FAISSVectorStore:
    """
    FAISSVectorStore manages the local vector database lifecycle, including
    saving, loading, creating, and verifying indexes.
    """
    def __init__(self, embeddings: HuggingFaceEmbeddings, store_dir: Path = config.VECTOR_STORE_DIR):
        """
        Initializes vector store manager with configured embeddings and folder path.
        """
        self.embeddings = embeddings
        self.store_dir = Path(store_dir)
        self.index_name = "index"

    def is_store_present(self) -> bool:
        """
        Checks if the FAISS index files already exist in local directory.
        Returns:
            bool: True if index files are present, False otherwise.
        """
        faiss_file = self.store_dir / f"{self.index_name}.faiss"
        pkl_file = self.store_dir / f"{self.index_name}.pkl"
        return faiss_file.exists() and pkl_file.exists()

    def create_from_documents(self, documents: List[Document]) -> FAISS:
        """
        Creates a new FAISS index from the given documents and saves it locally.
        Args:
            documents (List[Document]): Text chunks to embed and store.
        Returns:
            FAISS: The initialized FAISS vector store object.
        """
        logger.info(f"Creating new FAISS vector store with {len(documents)} document chunks")
        try:
            vector_store = FAISS.from_documents(documents, self.embeddings)
            self.save_store(vector_store)
            return vector_store
        except Exception as e:
            logger.error(f"Failed to create FAISS vector store: {str(e)}")
            raise RuntimeError(f"FAISS creation failed: {str(e)}") from e

    def save_store(self, vector_store: FAISS) -> None:
        """
        Saves the FAISS index to the local file system.
        """
        logger.info(f"Saving FAISS index locally to: {self.store_dir}")
        try:
            self.store_dir.mkdir(exist_ok=True, parents=True)
            vector_store.save_local(str(self.store_dir), index_name=self.index_name)
            logger.info("Vector store index saved successfully")
        except Exception as e:
            logger.error(f"Failed to save FAISS store: {str(e)}")
            raise IOError(f"FAISS saving failed: {str(e)}") from e

    def load_store(self) -> FAISS:
        """
        Loads the FAISS index from the local file system.
        Returns:
            FAISS: The loaded FAISS vector store object.
        """
        logger.info(f"Loading local FAISS index from: {self.store_dir}")
        if not self.is_store_present():
            error_msg = f"FAISS files not found at: {self.store_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        try:
            # allow_dangerous_deserialization is required to load FAISS pickle files.
            # Since the file is created locally on the user's system, this is safe here.
            vector_store = FAISS.load_local(
                folder_path=str(self.store_dir),
                embeddings=self.embeddings,
                index_name=self.index_name,
                allow_dangerous_deserialization=True
            )
            logger.info("Local vector store loaded successfully")
            return vector_store
        except Exception as e:
            logger.error(f"Failed to load FAISS store: {str(e)}")
            raise RuntimeError(f"FAISS loading failed: {str(e)}") from e

# --- Manual Test Execution Guide ---
# To test this file:
# 1. Run in terminal: python -c "from langchain_core.documents import Document; from src.embeddings import EmbeddingModel; from src.vectorstore import FAISSVectorStore; embeddings = EmbeddingModel().get_embeddings(); store = FAISSVectorStore(embeddings); docs = [Document(page_content='This is a document about RAG.')]; db = store.create_from_documents(docs); print('Is present after save?', store.is_store_present()); loaded_db = store.load_store(); results = loaded_db.similarity_search('RAG'); print('Search Result:', results[0].page_content)"
