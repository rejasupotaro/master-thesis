import itertools

import tensorflow as tf
from tensorflow.keras import layers

from src.layers.bias import AddBias0
from src.layers.interaction import WeightedInteraction
from src.models.base_model import BaseModel


class FMQuery(BaseModel):
    @property
    def name(self) -> str:
        return 'fm_query'

    def build(self):
        categorical_input_size = {
            'country': self.total_countries,
        }

        query_input = self.new_query_input()
        title_input = self.new_title_input()
        ingredients_input = self.new_ingredients_input()
        description_input = self.new_description_input()
        country_input = self.new_country_input()
        inputs = [query_input, title_input, ingredients_input, description_input, country_input]

        embedding = layers.Embedding(self.total_words, self.embedding_dim)
        query = layers.GlobalMaxPooling1D()(embedding(query_input))
        title = layers.GlobalMaxPooling1D()(embedding(title_input))
        ingredients = layers.GlobalMaxPooling1D()(embedding(ingredients_input))
        description = layers.GlobalMaxPooling1D()(embedding(description_input))

        embedding = layers.Embedding(categorical_input_size['country'], self.embedding_dim)
        country = embedding(country_input)
        country = tf.reshape(country, shape=(-1, self.embedding_dim,))

        query_title = layers.Dot(axes=1)([query, title])
        query_ingredients = layers.Dot(axes=1)([query, ingredients])
        query_description = layers.Dot(axes=1)([query, description])
        query_country = layers.Dot(axes=1)([query, country])
        interactions = layers.Add()([query_title, query_ingredients, query_description, query_country])

        embedding = layers.Embedding(self.total_words, 1)
        query = layers.GlobalMaxPooling1D()(embedding(query_input))
        title = layers.GlobalMaxPooling1D()(embedding(title_input))
        ingredients = layers.GlobalMaxPooling1D()(embedding(ingredients_input))
        description = layers.GlobalMaxPooling1D()(embedding(description_input))

        embedding = layers.Embedding(categorical_input_size['country'], 1)
        country = embedding(country_input)
        country = tf.reshape(country, shape=(-1, 1))
        biases = layers.Add()([query, title, ingredients, description, country])
        biases = AddBias0()(biases)

        output = layers.Activation('sigmoid', name='label')(biases + interactions)
        return tf.keras.Model(inputs=inputs, outputs=output, name=self.name)


class FMAll(BaseModel):
    @property
    def name(self) -> str:
        return 'fm_all'

    def build(self):
        categorical_input_size = {
            'country': self.total_countries,
        }

        text_inputs = [
            self.new_query_input(),
            self.new_title_input(),
            self.new_ingredients_input(),
            self.new_description_input(),
        ]
        categorical_inputs = [
            self.new_country_input(),
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


class FwFM(BaseModel):
    @property
    def name(self) -> str:
        return 'fwfm'

    def build(self):
        categorical_input_size = {
            'country': self.total_countries,
        }

        text_inputs = [
            self.new_query_input(),
            self.new_title_input(),
            self.new_ingredients_input(),
            self.new_description_input(),
        ]
        categorical_inputs = [
            self.new_country_input(),
        ]

        inputs = text_inputs + categorical_inputs

        embedding = layers.Embedding(self.total_words, self.embedding_dim, name='text_embedding')
        text_features = [embedding(text_input) for text_input in text_inputs]
        text_features = [layers.GlobalMaxPooling1D()(feature) for feature in text_features]

        categorical_features = []
        for name, categorical_input in zip(categorical_input_size, categorical_inputs):
            embedding = layers.Embedding(categorical_input_size[name], self.embedding_dim)
            feature = embedding(categorical_input)
            feature = tf.reshape(feature, shape=(-1, self.embedding_dim,))
            categorical_features.append(feature)

        features = text_features + categorical_features

        num_fields = len(features)
        features = tf.concat(features, axis=1)
        x = WeightedInteraction(num_fields, name='field_weights')(features)
        output = layers.Activation('sigmoid', name='label')(x)
        return tf.keras.Model(inputs=inputs, outputs=output, name=self.name)
