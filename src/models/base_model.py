import abc

import numpy as np
from tensorflow.keras import layers

from src.data.preprocessors import DataProcessor


class BaseModel(abc.ABC):
    def __init__(self, data_processor: DataProcessor, embedding_dim: int = 32):
        self.total_words = data_processor.total_words
        self.total_authors = data_processor.total_authors
        self.total_countries = data_processor.total_countries
        self.embedding_dim = embedding_dim

    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError('Calling an abstract method.')

    def ngram_block(self, n_gram, total_words):
        def wrapped(inputs):
            layer = layers.Conv1D(1, n_gram, use_bias=False, trainable=False)
            x = layers.Reshape((-1, 1))(inputs)
            x = layer(x)
            kernel = np.power(total_words, range(0, n_gram))
            layer.set_weights([kernel.reshape(n_gram, 1, 1)])
            return layers.Reshape((-1,))(x)

        return wrapped

    @abc.abstractmethod
    def build(self):
        raise NotImplementedError('Calling an abstract method.')
