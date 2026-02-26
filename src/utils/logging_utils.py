import logging
import os

def get_logger(save_dir):
    os.makedirs(save_dir, exist_ok=True)

    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(os.path.join(save_dir, "train.log"))
    file_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    return logger
