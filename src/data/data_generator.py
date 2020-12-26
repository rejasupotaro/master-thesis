import math
from typing import Tuple, Dict, List

from pandas import DataFrame
from tensorflow import keras

from src.data.cookpad.preprocessors import DataProcessor


class DataGenerator(keras.utils.Sequence):
    def __init__(self, df: DataFrame, processor: DataProcessor, batch_size: int = 256):
        self.df = df
        self.processor = processor
        self.batch_size = batch_size

    def __len__(self) -> int:
        return math.ceil(len(self.df) / self.batch_size)

    def __getitem__(self, idx: int) -> Tuple[Dict, List[int]]:
        batch = self.df[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self.processor.process_batch(batch)
