import os
import sys
from pathlib import Path
import config

def main():
    print("=== Modular RAG Chatbot: Local Ingestion and Retrieval Test ===")
    
    # Verify configs
    print(f"Local LLM Model: {config.LOCAL_MODEL_NAME}")
    print(f"Embedding Model: {config.EMBEDDING_MODEL_NAME}")
    print(f"Chunk Size / Overlap: {config.CHUNK_SIZE} / {config.CHUNK_OVERLAP}")

    # Check for PDF files in data directory
    pdf_files = list(config.DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print("\n❌ No PDF files found in the 'data/' directory.")
        print("To run a test query:")
        print("1. Place a PDF document (e.g. sample.pdf) inside the 'data/' folder.")
        print("2. Run this script again.")
        sys.exit(0)
        
    target_pdf = pdf_files[0]
    print(f"\n📂 Found PDF file for testing: {target_pdf.name}")
    
    try:
        # Import core modules inside venv context
        print("🤖 Loading embedding models and initializing components...")
        from src.chatbot import RAGChatbot
        
        bot = RAGChatbot()
        
        # Step 1: Ingest
        print(f"⏳ Ingesting {target_pdf.name} (Extracting, splitting, and embedding)...")
        num_chunks = bot.ingest_pdf(str(target_pdf))
        print(f"✅ Ingestion successful! Created {num_chunks} vector chunks in FAISS store.")
        
        # Step 2: Query
        query = "What is the main topic of this document?"
        print(f"\n💬 Querying: '{query}'")
        print("⏳ Fetching context and generating answer...")
        
        result = bot.ask(query)
        
        print("\n=== Chatbot Response ===")
        print(result["answer"])
        print("========================")
        
        print(f"\n📚 Sources retrieved: {len(result['source_documents'])} chunks used.")
        for idx, doc in enumerate(result['source_documents']):
            page = doc.metadata.get("page", 0) + 1
            print(f"  - Chunk #{idx+1}: Page {page} (Length: {len(doc.page_content)} characters)")
            
    except Exception as e:
        print(f"\n❌ Test pipeline failed: {str(e)}")
        print("Please check that your virtual environment is active and dependencies are installed.")

if __name__ == "__main__":
    main()
