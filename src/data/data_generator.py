import math

from tensorflow import keras


class DataGenerator(keras.utils.Sequence):
    def __init__(self, df, processor, batch_size=128):
        self.df = df
        self.processor = processor
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.df) / self.batch_size)

    def __getitem__(self, idx):
        batch = self.df[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self.processor.process_batch(batch)
