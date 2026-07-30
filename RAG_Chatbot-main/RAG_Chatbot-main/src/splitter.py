from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config
from src.utils import get_logger

logger = get_logger("splitter")

class DocumentSplitter:
    """
    Document splitter class responsible for breaking text documents down
    into smaller semantic chunks to feed into the vector store.
    """
    def __init__(self, chunk_size: int = config.CHUNK_SIZE, chunk_overlap: int = config.CHUNK_OVERLAP):
        """
        Initializes splitter with size and overlap configuration.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Recursive splitter tries to split by paragraphs, sentences, and then words
        # to preserve structural coherence and context boundaries.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits a list of documents into chunks.
        Args:
            documents (List[Document]): Raw loaded page documents.
        Returns:
            List[Document]: Chunked documents.
        """
        logger.info(f"Splitting {len(documents)} source pages with chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")
        
        try:
            chunks = self.splitter.split_documents(documents)
            logger.info(f"Successfully split into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Failed to split documents: {str(e)}")
            raise RuntimeError(f"Failed to split documents: {str(e)}") from e

# --- Manual Test Execution Guide ---
# To test this file:
# 1. Run in terminal: python -c "from langchain_core.documents import Document; from src.splitter import DocumentSplitter; doc = Document(page_content='This is a long sentence for testing the splitter component. ' * 50); chunks = DocumentSplitter(chunk_size=100, chunk_overlap=10).split_documents([doc]); print(f'Split into {len(chunks)} chunks. First chunk length: {len(chunks[0].page_content)}'); print('First chunk content:', chunks[0].page_content)"
