import os
import logging
from datetime import datetime

def setup_logging(log_file_path, level=logging.INFO, console_output=False):
    """
    Configures logging by saving logs to file and optionally to the console.
    """

    logger = logging.getLogger()
    logger.setLevel(level)

    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
 
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)


def create_output_directory():
    """Create an output folder with a timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('output', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir
