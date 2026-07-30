import logging
import sys
from pathlib import Path

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger with the specified name.
    """
    return logging.getLogger(name)

def clean_directory(directory_path: Path) -> None:
    """
    Deletes all files within the given directory (excludes hidden files like .gitkeep).
    """
    logger = get_logger("utils")
    if not directory_path.exists():
        return
    
    for file_path in directory_path.iterdir():
        if file_path.is_file() and not file_path.name.startswith('.'):
            try:
                file_path.unlink()
                logger.info(f"Deleted file: {file_path.name}")
            except Exception as e:
                logger.error(f"Failed to delete {file_path.name}: {e}")
