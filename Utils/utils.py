from typing import List
from tqdm import tqdm
import multiprocessing as mp
import logging
from datetime import datetime
import sys 

class tqdm_print():
    def __init__(self) -> None:
        self.progress_bars: List[tqdm] = list()

    def print(self, str: str):
        lines = str.split("\n")
        for line in lines:
            pbar = tqdm(desc=line, bar_format='{desc}', leave=False)
            self.progress_bars.append(pbar)

    def close(self):
        for pbar in self.progress_bars:
            pbar.close()

    def __call__(self, str: str):
        self.print(str)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def __enter__(self):
        return self


def create_logger(path):
    logger = mp.get_logger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '[%(asctime)s| %(levelname)s| %(processName)s] %(message)s')
    names = [handler.get_name() for handler in logger.handlers]
    for name, handler in zip(["file handler", "console handler"], [logging.FileHandler(path), logging.StreamHandler(sys.stdout)]):
        if name not in names:
            handler.setFormatter(formatter)
            handler.set_name(name)
            logger.addHandler(handler)
    return logger
