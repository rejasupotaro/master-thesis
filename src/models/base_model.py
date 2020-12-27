import abc

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.data.cookpad.preprocessors import DataProcessor


class BaseModel(abc.ABC):
    def __init__(self, data_processor: DataProcessor, embedding_dim: int = 32):
        self.doc_id_encoder = data_processor.doc_id_encoder
        self.total_words = data_processor.total_words
        if hasattr(data_processor, 'total_authors'):
            self.total_authors = data_processor.total_authors
        if hasattr(data_processor, 'total_countries'):
            self.total_countries = data_processor.total_countries
        self.embedding_dim = embedding_dim

    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError('Calling an abstract method.')

    def new_doc_id_input(self):
        return tf.keras.Input(shape=(1,), name='doc_id')

    def new_query_input(self, size=6):
        return tf.keras.Input(shape=(size,), name='query')

    def new_title_input(self, size=20):
        return tf.keras.Input(shape=(size,), name='title')

    def new_ingredients_input(self, size=300):
        return tf.keras.Input(shape=(size,), name='ingredients')

    def new_description_input(self, size=100):
        return tf.keras.Input(shape=(size,), name='description')

    def new_author_input(self, size=1):
        return tf.keras.Input(shape=(size,), name='author')

    def new_country_input(self, size=1):
        return tf.keras.Input(shape=(size,), name='country')

    def new_url_input(self, size=20):
        return tf.keras.Input(shape=(size,), name='url')

    def new_body_input(self, size=12000):
        return tf.keras.Input(shape=(size,), name='body')

    def load_pretrained_embedding(self, embedding_filepath: str, embedding_dim: int, name: str) -> layers.Embedding:
        df = pd.read_pickle(embedding_filepath)
        df.index = df.index.astype(np.int64)
        classes = self.doc_id_encoder.classes_
        doc_ids = [doc_id if doc_id in classes else -1 for doc_id in df.index]
        doc_ids = self.doc_id_encoder.transform(doc_ids)
        input_dim = len(classes)
        output_dim = embedding_dim
        embedding_matrix = np.zeros((input_dim, output_dim))
        for doc_id, values in zip(doc_ids, df.values):
            embedding_matrix[doc_id] = values

        embedding = layers.Embedding(
            input_dim,
            output_dim,
            embeddings_initializer=keras.initializers.Constant(embedding_matrix),
            trainable=False,
            name=name
        )
        return embedding

    def ngram_block(self, n_gram: int, total_words: int):
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
