from langchain_core.prompts import ChatPromptTemplate

# System prompt forcing grounded answers and prohibiting hallucination.
SYSTEM_PROMPT_TEMPLATE = """You are a precise, factual Retrieval-Augmented Generation (RAG) assistant.
Your goal is to answer the user's question using ONLY the provided text Context.

Strict Instructions:
1. Answer the question using ONLY the information provided in the Context below.
2. Do NOT extrapolate, invent, or use any prior knowledge outside of the provided Context.
3. If the answer to the query cannot be found within the provided Context, you must reply exactly:
   "I cannot find the answer to this question in the uploaded document."
4. Maintain a professional, direct, and concise tone. Do not provide unnecessary preambles or chatty responses.

---
CONTEXT:
{context}
---

QUESTION:
{question}

HELPFUL ANSWER:"""

class PromptBuilder:
    """
    PromptBuilder structures the prompt template using LangChain's ChatPromptTemplate.
    """
    @staticmethod
    def get_prompt_template() -> ChatPromptTemplate:
        """
        Creates and returns the chat prompt template format.
        """
        return ChatPromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE)

# --- Manual Test Execution Guide ---
# To test this file:
# 1. Run in terminal: python -c "from src.prompt import PromptBuilder; template = PromptBuilder.get_prompt_template(); formatted = template.format(context='The capital of France is Paris.', question='What is the capital of France?'); print(formatted)"
