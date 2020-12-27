import itertools

import tensorflow as tf
from tensorflow.keras import layers

from src.layers.bias import AddBias0
from src.layers.interaction import WeightedQueryFieldInteraction, WeightedFeatureInteraction
from src.models.base_model import BaseModel


class FMQuery(BaseModel):
    @property
    def name(self) -> str:
        return 'fm_query'

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
        input_features = [query, url, title, body]

        query_url = layers.Dot(axes=1)([query, url])
        query_title = layers.Dot(axes=1)([query, title])
        query_body = layers.Dot(axes=1)([query, body])
        interactions = layers.Add()([query_url, query_title, query_body])

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
            self.new_query_input(size=20),
            self.new_url_input(),
            self.new_title_input(),
            self.new_body_input(),
        ]
        inputs = text_inputs

        word_embedding = layers.Embedding(self.total_words, self.embedding_dim)
        text_features = [word_embedding(text_input) for text_input in text_inputs]
        text_features = [layers.GlobalMaxPooling1D()(feature) for feature in text_features]
        input_features = text_features

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
        input_features = [query, url, title, body]

        num_fields = len(input_features)
        features = tf.concat(input_features, axis=1)
        interactions = WeightedQueryFieldInteraction(num_fields, name='field_weights')(features)

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
            self.new_query_input(size=20),
            self.new_url_input(),
            self.new_title_input(),
            self.new_body_input(),
        ]
        inputs = text_inputs

        word_embedding = layers.Embedding(self.total_words, self.embedding_dim, name='text_embedding')
        text_features = [word_embedding(text_input) for text_input in text_inputs]
        text_features = [layers.GlobalMaxPooling1D()(feature) for feature in text_features]
        input_features = text_features

        num_fields = len(input_features)
        features = tf.concat(input_features, axis=1)
        interactions = WeightedFeatureInteraction(num_fields, name='field_weights')(features)

        features = []
        for feature in input_features:
            feature = layers.Dense(1, activation='relu')(feature)
            features.append(feature)
        features = layers.Add()(features)
        features = AddBias0()(features)

        output = layers.Activation('sigmoid', name='label')(features + interactions)
        return tf.keras.Model(inputs=inputs, outputs=output, name=self.name)
