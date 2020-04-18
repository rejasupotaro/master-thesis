import os
from pathlib import Path
from logging import getLogger, Formatter, FileHandler, StreamHandler, INFO, DEBUG

project_dir = Path(__file__).resolve().parents[2]
logs_dir = os.path.join(project_dir, 'logs')


def create_logger(version=1):
    log_file = os.path.join(logs_dir, f'{version}.log')

    logger = getLogger(str(version))
    logger.setLevel(DEBUG)
    formatter = Formatter("[%(levelname)s] %(asctime)s >> %(message)s")

    file_handler = FileHandler(log_file)
    file_handler.setLevel(DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = StreamHandler()
    stream_handler.setLevel(DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


def get_logger(version=1):
    return getLogger(str(version))
