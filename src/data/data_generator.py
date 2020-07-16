import math

import pandas as pd
from tensorflow import keras
from src.data.preprocessors import DataProcessor


class DataGenerator(keras.utils.Sequence):
    def __init__(self, df: pd.DataFrame, processor: DataProcessor, batch_size: int = 128):
        self.df: pd.DataFrame = df
        self.processor: DataProcessor = processor
        self.batch_size: int = batch_size

    def __len__(self) -> int:
        return math.ceil(len(self.df) / self.batch_size)

    def __getitem__(self, idx: int):
        batch = self.df[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self.processor.process_batch(batch)
