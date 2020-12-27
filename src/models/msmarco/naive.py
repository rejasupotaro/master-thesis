import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.base_model import BaseModel


class Naive(BaseModel):
    @property
    def name(self) -> str:
        return 'naive'

    def build(self):
        query_input = self.new_query_input(size=20)
        url_input = self.new_url_input()
        title_input = self.new_title_input()
        body_input = self.new_body_input()
        inputs = [query_input, url_input, title_input, body_input]

        word_embedding = layers.Embedding(self.total_words, self.embedding_dim)
        query = layers.GlobalMaxPooling1D()(word_embedding(query_input))
        url = layers.GlobalMaxPooling1D()(word_embedding(url_input))
        title = layers.GlobalMaxPooling1D()(word_embedding(title_input))
        body = layers.GlobalMaxPooling1D()(word_embedding(body_input))

        x = layers.concatenate([query, url, title, body])
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dense(8, activation='relu')(x)
        output = layers.Dense(1, activation='sigmoid', name='label')(x)

        return keras.Model(inputs=inputs, outputs=output, name=self.name)
