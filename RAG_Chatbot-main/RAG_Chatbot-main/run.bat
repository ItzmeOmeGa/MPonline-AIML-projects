@echo off
echo ===================================================
echo 📚 Starting Modular RAG Chatbot (DocuQuest AI) ...
echo ===================================================

:: Check if virtual environment exists
if not exist venv\Scripts\activate.bat (
    echo ❌ Python virtual environment [venv] not found in this folder.
    echo Please make sure you have run: python -m venv venv
    pause
    exit /b
)

:: Activate local environment
call venv\Scripts\activate.bat

:: Launch Streamlit web app
echo 🚀 Launching Streamlit interface...
streamlit run app.py
