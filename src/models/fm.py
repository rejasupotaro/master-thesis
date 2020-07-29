import itertools

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.layers.bias import AddBias0
from src.models.base_model import BaseModel


class FMQuery(BaseModel):
    @property
    def name(self) -> str:
        return 'fm_query'

    def build(self):
        query_len = 6
        title_len = 20
        ingredients_len = 300
        description_len = 100

        categorical_input_size = {
            'author': self.total_authors,
            'country': self.total_countries,
        }

        query_input = keras.Input(shape=(query_len,), name='query_word_ids')
        title_input = keras.Input(shape=(title_len,), name='title_word_ids')
        ingredients_input = keras.Input(shape=(ingredients_len,), name='ingredients_word_ids')
        description_input = keras.Input(shape=(description_len,), name='description_word_ids')
        author_input = keras.Input(shape=(1,), name='author')
        country_input = keras.Input(shape=(1,), name='country')
        inputs = [query_input, title_input, ingredients_input, description_input, author_input, country_input]

        embedding = layers.Embedding(self.total_words, self.embedding_dim)
        query = layers.GlobalMaxPooling1D()(embedding(query_input))
        title = layers.GlobalMaxPooling1D()(embedding(title_input))
        ingredients = layers.GlobalMaxPooling1D()(embedding(ingredients_input))
        description = layers.GlobalMaxPooling1D()(embedding(description_input))

        embedding = layers.Embedding(categorical_input_size['author'], self.embedding_dim)
        author = embedding(author_input)
        author = tf.reshape(author, shape=(-1, self.embedding_dim,))
        embedding = layers.Embedding(categorical_input_size['country'], self.embedding_dim)
        country = embedding(country_input)
        country = tf.reshape(country, shape=(-1, self.embedding_dim,))

        query_title = layers.Dot(axes=1)([query, title])
        query_ingredients = layers.Dot(axes=1)([query, ingredients])
        query_description = layers.Dot(axes=1)([query, description])
        query_author = layers.Dot(axes=1)([query, author])
        query_country = layers.Dot(axes=1)([query, country])
        interactions = layers.Add()([query_title, query_ingredients, query_description, query_author, query_country])

        embedding = layers.Embedding(self.total_words, 1)
        query = layers.GlobalMaxPooling1D()(embedding(query_input))
        title = layers.GlobalMaxPooling1D()(embedding(title_input))
        ingredients = layers.GlobalMaxPooling1D()(embedding(ingredients_input))
        description = layers.GlobalMaxPooling1D()(embedding(description_input))

        embedding = layers.Embedding(categorical_input_size['author'], 1)
        author = embedding(author_input)
        author = tf.reshape(author, shape=(-1, 1))
        embedding = layers.Embedding(categorical_input_size['country'], 1)
        country = embedding(country_input)
        country = tf.reshape(country, shape=(-1, 1))
        biases = layers.Add()([query, title, ingredients, description, author, country])
        biases = AddBias0()(biases)

        output = layers.Activation('sigmoid', name='label')(biases + interactions)
        return tf.keras.Model(inputs=inputs, outputs=output, name=self.name)


class FMAll(BaseModel):
    @property
    def name(self) -> str:
        return 'fm_all'

    def build(self):
        query_len = 6
        title_len = 20
        ingredients_len = 300
        description_len = 100

        categorical_input_size = {
            'author': self.total_authors,
            'country': self.total_countries,
        }

        text_inputs = [
            keras.Input(shape=(query_len,), name='query_word_ids'),
            keras.Input(shape=(title_len,), name='title_word_ids'),
            keras.Input(shape=(ingredients_len,), name='ingredients_word_ids'),
            keras.Input(shape=(description_len,), name='description_word_ids'),
        ]

        categorical_inputs = [
            keras.Input(shape=(1,), name='author'),
            keras.Input(shape=(1,), name='country'),
        ]

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
        for feature in text_inputs:
            feature = layers.Embedding(self.total_words, 1)(feature)
            feature = layers.GlobalMaxPooling1D()(feature)
            biases.append(feature)
        for name, feature in zip(categorical_input_size, categorical_inputs):
            feature = layers.Embedding(categorical_input_size[name], 1)(feature)
            feature = tf.reshape(feature, shape=(-1, 1))
            biases.append(feature)
        biases = layers.Add()(biases)
        biases = AddBias0()(biases)

        output = layers.Activation('sigmoid', name='label')(biases + interactions)
        return tf.keras.Model(inputs=inputs, outputs=output, name=self.name)
