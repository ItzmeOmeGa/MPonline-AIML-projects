import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
import config
from src.utils import get_logger
from src.loader import PDFDocumentLoader
from src.splitter import DocumentSplitter
from src.embeddings import EmbeddingModel
from src.vectorstore import FAISSVectorStore
from src.retriever import ContextRetriever
from src.prompt import PromptBuilder
from src.llm import LocalLLM

logger = get_logger("chatbot")

class RAGChatbot:
    """
    RAGChatbot is the primary coordinator orchestrating the local offline RAG pipeline.
    """
    def __init__(self):
        """
        Initializes local models and configurations.
        """
        logger.info("Initializing RAG Chatbot coordinator")
        
        # Load local embedding model (CPU)
        self.embedding_factory = EmbeddingModel()
        self.embeddings = self.embedding_factory.get_embeddings()
        
        # Setup local vector store manager
        self.vector_store_manager = FAISSVectorStore(self.embeddings)
        self.vector_store = None
        self.retriever = None
        
        # Initialize local LLM factory (CPU)
        self.llm_factory = LocalLLM()
        self.llm = self.llm_factory.get_llm()
        
        # Load prompt template
        self.prompt_template = PromptBuilder.get_prompt_template()
        
        # Auto-load existing index if present
        if self.vector_store_manager.is_store_present():
            logger.info("Existing FAISS database detected on startup. Loading...")
            try:
                self.vector_store = self.vector_store_manager.load_store()
                self.retriever = ContextRetriever(self.vector_store)
                logger.info("FAISS database loaded successfully on startup")
            except Exception as e:
                logger.error(f"Failed to auto-load vector database: {str(e)}")

    def is_ready(self) -> bool:
        """
        Checks if the chatbot is ready to answer queries.
        """
        return self.retriever is not None

    def ingest_pdf(self, file_path: str) -> int:
        """
        Ingests a PDF file: loads it, splits it, embeds chunks, saves FAISS index.
        """
        logger.info(f"Ingestion started for file: {file_path}")
        
        # Step 1: Extract Text
        loader = PDFDocumentLoader(file_path)
        documents = loader.load()
        
        # Step 2: Split text into small chunks
        splitter = DocumentSplitter()
        chunks = splitter.split_documents(documents)
        
        # Step 3: Embed chunks and save in FAISS Vector Store
        self.vector_store = self.vector_store_manager.create_from_documents(chunks)
        
        # Step 4: Initialize Retriever
        self.retriever = ContextRetriever(self.vector_store)
        
        logger.info(f"Ingestion complete. {len(chunks)} chunks saved.")
        return len(chunks)

    def ask(self, query: str) -> Dict[str, Any]:
        """
        Runs RAG pipeline query: retrieves context chunks, builds prompt, calls Local LLM.
        """
        if not self.is_ready():
            return {
                "answer": "No documents uploaded. Please upload a PDF document first.",
                "source_documents": []
            }
            
        logger.info(f"Querying chatbot: '{query}'")
        
        try:
            # Step 1: Retrieve context
            retrieved_docs = self.retriever.retrieve(query)
            
            # Step 2: Stringify chunks (list + join avoids repeated string re-allocation
            # that a naive "+=" loop causes as the number of chunks grows)
            chunk_blocks = [
                f"Document Chunk #{idx + 1} [Source: Page {doc.metadata.get('page', 0) + 1}]:\n{doc.page_content}"
                for idx, doc in enumerate(retrieved_docs)
            ]
            context_text = "\n\n".join(chunk_blocks)
            
            # Step 3: Format prompt
            formatted_prompt = self.prompt_template.format(
                context=context_text,
                question=query
            )
            
            # Step 4: Generate response locally
            logger.info("Invoking Local LLM for response generation")
            response = self.llm.invoke(formatted_prompt)
            
            # The pipeline is configured with return_full_text=False, so the prompt is not
            # echoed back. We still defensively strip the legacy marker in case a different
            # model/config is swapped in that doesn't honor that setting.
            answer = response
            helpful_indicator = "HELPFUL ANSWER:"
            if helpful_indicator in answer:
                answer = answer.split(helpful_indicator)[-1]
            
            return {
                "answer": answer.strip(),
                "source_documents": retrieved_docs
            }
            
        except Exception as e:
            logger.error(f"Error answering query: {str(e)}")
            return {
                "answer": f"An error occurred while answering your question: {str(e)}",
                "source_documents": []
            }
