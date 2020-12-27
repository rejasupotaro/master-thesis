import itertools

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.base_model import BaseModel


class NRMFSimpleQuery(BaseModel):
    @property
    def name(self):
        return 'nrmf_simple_query'

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
        fields = [url, title, body]

        interactions = []
        for field in fields:
            interactions.append(tf.multiply(query, field))

        x = layers.concatenate(interactions)
        x = layers.Dense(32, activation='relu')(x)
        output = layers.Dense(1, activation='sigmoid', name='label')(x)

        return keras.Model(inputs=inputs, outputs=output, name=self.name)


class NRMFSimpleAll(BaseModel):
    @property
    def name(self):
        return 'nrmf_simple_all'

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
        features = [query, url, title, body]

        interactions = []
        for feature1, feature2 in itertools.combinations(features, 2):
            interactions.append(tf.multiply(feature1, feature2))

        x = layers.concatenate(interactions)
        x = layers.Dense(32, activation='relu')(x)
        output = layers.Dense(1, activation='sigmoid', name='label')(x)

        return keras.Model(inputs=inputs, outputs=output, name=self.name)
