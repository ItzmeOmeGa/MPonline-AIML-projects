from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from src.utils import get_logger

logger = get_logger("loader")

class PDFDocumentLoader:
    """
    Document loader class responsible for parsing PDF documents and extracting text.
    Uses LangChain's PyPDFLoader under the hood.
    """
    def __init__(self, file_path: str):
        """
        Initializes the loader with the path of the PDF.
        """
        self.file_path = Path(file_path)

    def load(self) -> List[Document]:
        """
        Loads the PDF file and extracts text page by page.
        Returns:
            List[Document]: List of LangChain Document objects containing page text and metadata.
        """
        logger.info(f"Attempting to load PDF from path: {self.file_path}")
        
        if not self.file_path.exists():
            error_msg = f"File not found at: {self.file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        try:
            # PyPDFLoader parses the PDF and creates a document for each page
            loader = PyPDFLoader(str(self.file_path))
            documents = loader.load()
            
            logger.info(f"Successfully loaded {len(documents)} pages from {self.file_path.name}")
            return documents
            
        except Exception as e:
            logger.error(f"Error occurred while parsing PDF {self.file_path.name}: {str(e)}")
            raise RuntimeError(f"Failed to parse PDF file: {str(e)}") from e

# --- Manual Test Execution Guide ---
# To test this file individually:
# 1. Place a PDF in d:/A/RAG_Chatbot/data/sample.pdf
# 2. Run in terminal: python -c "from src.loader import PDFDocumentLoader; loader = PDFDocumentLoader('data/sample.pdf'); docs = loader.load(); print(f'Loaded {len(docs)} pages'); print('First page content sample:', docs[0].page_content[:200] if docs else 'No pages')"
