import itertools

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.base_model import BaseModel


class FM(BaseModel):
    @property
    def name(self) -> str:
        return 'FM'

    def build(self):
        query_len = 6
        title_len = 20
        ingredients_len = 300
        description_len = 100

        text_input_size = {
            'query': 6,
            'title': 20,
            'ingredients': 300,
            'description': 100,
        }
        categorical_input_size = {
            'author': self.total_authors,
            'country': self.total_countries,
        }

        query_input = keras.Input(shape=(query_len,), name='query_word_ids')
        title_input = keras.Input(shape=(title_len,), name='title_word_ids')
        ingredients_input = keras.Input(shape=(ingredients_len,), name='ingredients_word_ids')
        description_input = keras.Input(shape=(description_len,), name='description_word_ids')
        text_inputs = [query_input, title_input, ingredients_input, description_input]

        author_input = keras.Input(shape=(1,), name='author')
        country_input = keras.Input(shape=(1,), name='country')
        categorical_inputs = [author_input, country_input]

        inputs = text_inputs + categorical_inputs

        embedding = layers.Embedding(self.total_words, self.embedding_dim)
        text_features = [embedding(text_input) for text_input in text_inputs]
        text_features = [layers.GlobalMaxPooling1D()(feature) for feature in text_features]

        categorical_features = []
        for name, categorical_input in zip(categorical_input_size, categorical_inputs):
            embedding = layers.Embedding(categorical_input_size[name], self.embedding_dim)
            feature = embedding(categorical_input)
            feature = tf.reshape(feature, shape=(-1, self.embedding_dim,))
            categorical_features.append(feature)

        features = text_features + categorical_features

        interactions = []
        for feature1, feature2 in itertools.combinations(features, 2):
            interactions.append(layers.Dot(axes=1)([feature1, feature2]))
        interactions = layers.Add()(interactions)

        biases = []
        for name, feature in zip(text_input_size, text_inputs):
            feature = layers.Embedding(self.total_words, 1)(feature)
            feature = layers.GlobalMaxPooling1D()(feature)
            biases.append(feature)
        for name, feature in zip(categorical_input_size, categorical_inputs):
            feature = layers.Embedding(categorical_input_size[name], 1)(feature)
            feature = tf.reshape(feature, shape=(-1, 1))
            biases.append(feature)
        biases = layers.Add()(biases)

        output = layers.Activation('sigmoid', name='label')(interactions + biases)
        return tf.keras.Model(inputs=inputs, outputs=output)
