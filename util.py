import sys
import logging
from loguru import logger
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from tqdm import tqdm
import time
import random

# Remove default handlers to avoid duplicate logs
logger.remove()

# --- Configuration ---
LOG_LEVEL = "INFO"  # Set the desired log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

# --- Jupyter/Colab Check ---
def is_in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # Check if not in Jupyter
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True

IN_NOTEBOOK = is_in_notebook()

# --- Loguru + Rich Handler Setup ---
if IN_NOTEBOOK:
    # For Jupyter/Colab, use the default RichHandler
    handler = RichHandler(
        rich_tracebacks=True,  # Show rich tracebacks for exceptions
        markup=True  # Enable rich markup
    )
else:
    # For terminal, customize RichHandler for better formatting
    handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_time=False,  # We have time in the log format
        show_level=False,  # We have level in the format
        show_path=False,  # We have path in the format
    )

logger.add(handler, format=LOG_FORMAT, level=LOG_LEVEL, backtrace=True, diagnose=True, catch=True)