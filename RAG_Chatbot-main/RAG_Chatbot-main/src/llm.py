from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import config
from src.utils import get_logger

logger = get_logger("llm")

class LocalLLM:
    """
    LocalLLM manages the initialization of our 100% offline, local language model.
    Downloads the model automatically from Hugging Face on first run and runs on CPU.
    """
    def __init__(self, model_name: str = config.LOCAL_MODEL_NAME):
        """
        Initializes local model wrapper.
        """
        self.model_name = model_name
        self.llm = None

    def get_llm(self) -> HuggingFacePipeline:
        """
        Initializes and returns the HuggingFacePipeline local instance.
        """
        if self.llm is None:
            logger.info(f"Initializing local HuggingFace LLM: {self.model_name}")
            logger.info("NOTE: First run will download model weights (~950MB) to your machine.")
            try:
                # Load tokenizer and model weights from local cache or download them.
                # low_cpu_mem_usage trims peak RAM during weight loading on CPU-only machines.
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    low_cpu_mem_usage=True,
                )

                # Some causal LMs (Qwen included) ship without a pad token, which throws
                # noisy warnings during generation. Fall back to EOS.
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id

                # Setup HuggingFace pipeline configured for CPU execution
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=config.MAX_NEW_TOKENS,
                    temperature=config.TEMPERATURE,
                    repetition_penalty=config.REPETITION_PENALTY,
                    do_sample=True,
                    return_full_text=False,  # Only return the newly generated answer, not the echoed prompt
                    pad_token_id=tokenizer.pad_token_id,
                    device=-1,  # Force CPU execution (safe for all standard laptops)
                )

                # Wrap the pipeline inside LangChain interface
                self.llm = HuggingFacePipeline(pipeline=pipe)
                logger.info("Local LLM model initialized successfully")

            except Exception as e:
                logger.error(f"Failed to load local model: {str(e)}")
                raise RuntimeError(f"Failed to load local model: {str(e)}") from e

        return self.llm

# --- Manual Test Execution Guide ---
# To test this file:
# 1. Run in terminal: python -c "from src.llm import LocalLLM; llm = LocalLLM().get_llm(); print('Response:', llm.invoke('Say Hello'))"
