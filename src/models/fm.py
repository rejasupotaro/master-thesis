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
        query_input = self.new_query_input()
        title_input = self.new_title_input()
        ingredients_input = self.new_ingredients_input()
        description_input = self.new_description_input()
        country_input = self.new_country_input()
        inputs = [query_input, title_input, ingredients_input, description_input, country_input]

        word_embedding = layers.Embedding(self.total_words, self.embedding_dim)
        query = layers.GlobalMaxPooling1D()(word_embedding(query_input))
        title = layers.GlobalMaxPooling1D()(word_embedding(title_input))
        ingredients = layers.GlobalMaxPooling1D()(word_embedding(ingredients_input))
        description = layers.GlobalMaxPooling1D()(word_embedding(description_input))
        country_embedding = layers.Embedding(self.total_countries, self.embedding_dim)
        country = country_embedding(country_input)
        country = tf.reshape(country, shape=(-1, self.embedding_dim,))
        input_features = [query, title, ingredients, description, country]

        query_title = layers.Dot(axes=1)([query, title])
        query_ingredients = layers.Dot(axes=1)([query, ingredients])
        query_description = layers.Dot(axes=1)([query, description])
        query_country = layers.Dot(axes=1)([query, country])
        interactions = layers.Add()([query_title, query_ingredients, query_description, query_country])

        features = []
        for feature in input_features:
            feature = layers.Dense(1, activation='relu')(feature)
            features.append(feature)
        features = layers.Add()(features)
        features = AddBias0()(features)

        output = layers.Activation('sigmoid', name='label')(features + interactions)
        return tf.keras.Model(inputs=inputs, outputs=output, name=self.name)


class FMAll(BaseModel):
    @property
    def name(self) -> str:
        return 'fm_all'

    def build(self):
        text_inputs = [
            self.new_query_input(),
            self.new_title_input(),
            self.new_ingredients_input(),
            self.new_description_input(),
        ]
        country_input = self.new_country_input()
        inputs = text_inputs + [country_input]

        word_embedding = layers.Embedding(self.total_words, self.embedding_dim)
        text_features = [word_embedding(text_input) for text_input in text_inputs]
        text_features = [layers.GlobalMaxPooling1D()(feature) for feature in text_features]
        country_embedding = layers.Embedding(self.total_countries, self.embedding_dim)
        country = country_embedding(country_input)
        country = tf.reshape(country, shape=(-1, self.embedding_dim,))
        input_features = text_features + [country]

        interactions = []
        for feature1, feature2 in itertools.combinations(input_features, 2):
            interactions.append(layers.Dot(axes=1)([feature1, feature2]))
        interactions = layers.Add()(interactions)

        features = []
        for feature in input_features:
            feature = layers.Dense(1, activation='relu')(feature)
            features.append(feature)
        features = layers.Add()(features)
        features = AddBias0()(features)

        output = layers.Activation('sigmoid', name='label')(features + interactions)
        return tf.keras.Model(inputs=inputs, outputs=output, name=self.name)


class FwFMQuery(BaseModel):
    @property
    def name(self) -> str:
        return 'fwfm_query'

    def build(self):
        query_input = self.new_query_input()
        title_input = self.new_title_input()
        ingredients_input = self.new_ingredients_input()
        description_input = self.new_description_input()
        country_input = self.new_country_input()
        inputs = [query_input, title_input, ingredients_input, description_input, country_input]

        word_embedding = layers.Embedding(self.total_words, self.embedding_dim)
        query = layers.GlobalMaxPooling1D()(word_embedding(query_input))
        title = layers.GlobalMaxPooling1D()(word_embedding(title_input))
        ingredients = layers.GlobalMaxPooling1D()(word_embedding(ingredients_input))
        description = layers.GlobalMaxPooling1D()(word_embedding(description_input))
        country_embedding = layers.Embedding(self.total_countries, self.embedding_dim)
        country = country_embedding(country_input)
        country = tf.reshape(country, shape=(-1, self.embedding_dim,))
        input_features = [query, title, ingredients, description, country]

        num_fields = len(input_features)
        features = tf.concat(input_features, axis=1)
        interactions = WeightedInteraction(num_fields, name='field_weights')(features)

        features = []
        for feature in input_features:
            feature = layers.Dense(1, activation='relu')(feature)
            features.append(feature)
        features = layers.Add()(features)
        features = AddBias0()(features)

        output = layers.Activation('sigmoid', name='label')(features + interactions)
        return tf.keras.Model(inputs=inputs, outputs=output, name=self.name)


class FwFMAll(BaseModel):
    @property
    def name(self) -> str:
        return 'fwfm_all'

    def build(self):
        text_inputs = [
            self.new_query_input(),
            self.new_title_input(),
            self.new_ingredients_input(),
            self.new_description_input(),
        ]
        country_input = self.new_country_input()
        inputs = text_inputs + [country_input]

        word_embedding = layers.Embedding(self.total_words, self.embedding_dim, name='text_embedding')
        text_features = [word_embedding(text_input) for text_input in text_inputs]
        text_features = [layers.GlobalMaxPooling1D()(feature) for feature in text_features]
        country_embedding = layers.Embedding(self.total_countries, self.embedding_dim)
        country = country_embedding(country_input)
        country = tf.reshape(country, shape=(-1, self.embedding_dim,))
        input_features = text_features + [country]

        num_fields = len(input_features)
        features = tf.concat(input_features, axis=1)
        interactions = WeightedInteraction(num_fields, name='field_weights')(features)

        features = []
        for feature in input_features:
            feature = layers.Dense(1, activation='relu')(feature)
            features.append(feature)
        features = layers.Add()(features)
        features = AddBias0()(features)

        output = layers.Activation('sigmoid', name='label')(features + interactions)
        return tf.keras.Model(inputs=inputs, outputs=output, name=self.name)
