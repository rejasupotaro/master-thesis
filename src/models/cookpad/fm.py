import itertools
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers

from src.layers.bias import AddBias0
from src.layers.interaction import WeightedQueryFieldInteraction, WeightedFeatureInteraction, \
    WeightedSelectedFeatureInteraction
from src.models.base_model import BaseModel

project_dir = Path(__file__).resolve().parents[3]


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
        doc_id_input = self.new_doc_id_input()
        inputs = [query_input, title_input, ingredients_input, description_input, country_input, doc_id_input]

        word_embedding = layers.Embedding(self.total_words, self.embedding_dim, name='word_embedding')
        query = layers.GlobalMaxPooling1D()(word_embedding(query_input))
        title = layers.GlobalMaxPooling1D()(word_embedding(title_input))
        ingredients = layers.GlobalMaxPooling1D()(word_embedding(ingredients_input))
        description = layers.GlobalMaxPooling1D()(word_embedding(description_input))
        country_embedding = layers.Embedding(self.total_countries, self.embedding_dim)
        country = country_embedding(country_input)
        country = tf.reshape(country, shape=(-1, self.embedding_dim,))
        image_embedding = self.load_pretrained_embedding(
            embedding_filepath=f'{project_dir}/data/raw/en_2020-03-16T00_04_34_recipe_image_tagspace5000_300.pkl',
            embedding_dim=300,
            name='image_embedding'
        )
        image = image_embedding(doc_id_input)
        image = tf.reshape(image, shape=(-1, 300,))
        image = layers.Dropout(.2)(image)
        image = layers.Dense(self.embedding_dim)(image)
        input_features = [query, title, ingredients, description, country, image]

        query_title = layers.Dot(axes=1)([query, title])
        query_ingredients = layers.Dot(axes=1)([query, ingredients])
        query_description = layers.Dot(axes=1)([query, description])
        query_country = layers.Dot(axes=1)([query, country])
        query_image = layers.Dot(axes=1)([query, image])
        interactions = layers.Add()([query_title, query_ingredients, query_description, query_country, query_image])

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
        doc_id_input = self.new_doc_id_input()
        inputs = text_inputs + [country_input, doc_id_input]

        word_embedding = layers.Embedding(self.total_words, self.embedding_dim, name='word_embedding')
        texts = [layers.GlobalMaxPooling1D()(word_embedding(text_input)) for text_input in text_inputs]
        country_embedding = layers.Embedding(self.total_countries, self.embedding_dim)
        country = country_embedding(country_input)
        country = tf.reshape(country, shape=(-1, self.embedding_dim,))
        image_embedding = self.load_pretrained_embedding(
            embedding_filepath=f'{project_dir}/data/raw/en_2020-03-16T00_04_34_recipe_image_tagspace5000_300.pkl',
            embedding_dim=300,
            name='image_embedding'
        )
        image = image_embedding(doc_id_input)
        image = tf.reshape(image, shape=(-1, 300,))
        image = layers.Dropout(.2)(image)
        image = layers.Dense(self.embedding_dim)(image)
        input_features = texts + [country, image]

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
        doc_id_input = self.new_doc_id_input()
        inputs = [query_input, title_input, ingredients_input, description_input, country_input, doc_id_input]

        word_embedding = layers.Embedding(self.total_words, self.embedding_dim, name='word_embedding')
        query = layers.GlobalMaxPooling1D()(word_embedding(query_input))
        title = layers.GlobalMaxPooling1D()(word_embedding(title_input))
        ingredients = layers.GlobalMaxPooling1D()(word_embedding(ingredients_input))
        description = layers.GlobalMaxPooling1D()(word_embedding(description_input))
        country_embedding = layers.Embedding(self.total_countries, self.embedding_dim)
        country = country_embedding(country_input)
        country = tf.reshape(country, shape=(-1, self.embedding_dim,))
        image_embedding = self.load_pretrained_embedding(
            embedding_filepath=f'{project_dir}/data/raw/en_2020-03-16T00_04_34_recipe_image_tagspace5000_300.pkl',
            embedding_dim=300,
            name='image_embedding'
        )
        image = image_embedding(doc_id_input)
        image = tf.reshape(image, shape=(-1, 300,))
        image = layers.Dropout(.2)(image)
        image = layers.Dense(self.embedding_dim)(image)
        input_features = [query, title, ingredients, description, country, image]

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
            self.new_query_input(),
            self.new_title_input(),
            self.new_ingredients_input(),
            self.new_description_input(),
        ]
        country_input = self.new_country_input()
        doc_id_input = self.new_doc_id_input()
        inputs = text_inputs + [country_input, doc_id_input]

        word_embedding = layers.Embedding(self.total_words, self.embedding_dim, name='word_embedding')
        text_features = [word_embedding(text_input) for text_input in text_inputs]
        text_features = [layers.GlobalMaxPooling1D()(feature) for feature in text_features]
        country_embedding = layers.Embedding(self.total_countries, self.embedding_dim)
        country = country_embedding(country_input)
        country = tf.reshape(country, shape=(-1, self.embedding_dim,))
        image_embedding = self.load_pretrained_embedding(
            embedding_filepath=f'{project_dir}/data/raw/en_2020-03-16T00_04_34_recipe_image_tagspace5000_300.pkl',
            embedding_dim=300,
            name='image_embedding'
        )
        image = image_embedding(doc_id_input)
        image = tf.reshape(image, shape=(-1, 300,))
        image = layers.Dropout(.2)(image)
        image = layers.Dense(self.embedding_dim)(image)
        input_features = text_features + [country, image]

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


class FwFMSelected(BaseModel):
    @property
    def name(self) -> str:
        return 'fwfm_selected'

    def build(self):
        query_input = self.new_query_input()
        title_input = self.new_title_input()
        ingredients_input = self.new_ingredients_input()
        description_input = self.new_description_input()
        country_input = self.new_country_input()
        doc_id_input = self.new_doc_id_input()
        inputs = [query_input, title_input, ingredients_input, description_input, country_input, doc_id_input]

        word_embedding = layers.Embedding(self.total_words, self.embedding_dim, name='word_embedding')
        query = layers.GlobalMaxPooling1D()(word_embedding(query_input))
        title = layers.GlobalMaxPooling1D()(word_embedding(title_input))
        ingredients = layers.GlobalMaxPooling1D()(word_embedding(ingredients_input))
        description = layers.GlobalMaxPooling1D()(word_embedding(description_input))
        country_embedding = layers.Embedding(self.total_countries, self.embedding_dim)
        country = country_embedding(country_input)
        country = tf.reshape(country, shape=(-1, self.embedding_dim,))
        image_embedding = self.load_pretrained_embedding(
            embedding_filepath=f'{project_dir}/data/raw/en_2020-03-16T00_04_34_recipe_image_tagspace5000_300.pkl',
            embedding_dim=300,
            name='image_embedding'
        )
        image = image_embedding(doc_id_input)
        image = tf.reshape(image, shape=(-1, 300,))
        image = layers.Dropout(.2)(image)
        image = layers.Dense(self.embedding_dim)(image)
        input_features = [query, title, ingredients, description, country, image]

        num_fields = len(input_features)
        features = tf.concat(input_features, axis=1)
        interactions = WeightedSelectedFeatureInteraction(num_fields, name='field_weights')(features)

        features = []
        for feature in input_features:
            feature = layers.Dense(1, activation='relu')(feature)
            features.append(feature)
        features = layers.Add()(features)
        features = AddBias0()(features)

        output = layers.Activation('sigmoid', name='label')(features + interactions)
        return tf.keras.Model(inputs=inputs, outputs=output, name=self.name)


class FwFMAllWithout1st(BaseModel):
    @property
    def name(self) -> str:
        return 'fwfm_all_without_1st'

    def build(self):
        text_inputs = [
            self.new_query_input(),
            self.new_title_input(),
            self.new_ingredients_input(),
            self.new_description_input(),
        ]
        country_input = self.new_country_input()
        doc_id_input = self.new_doc_id_input()
        inputs = text_inputs + [country_input, doc_id_input]

        word_embedding = layers.Embedding(self.total_words, self.embedding_dim, name='word_embedding')
        text_features = [word_embedding(text_input) for text_input in text_inputs]
        text_features = [layers.GlobalMaxPooling1D()(feature) for feature in text_features]
        country_embedding = layers.Embedding(self.total_countries, self.embedding_dim)
        country = country_embedding(country_input)
        country = tf.reshape(country, shape=(-1, self.embedding_dim,))
        image_embedding = self.load_pretrained_embedding(
            embedding_filepath=f'{project_dir}/data/raw/en_2020-03-16T00_04_34_recipe_image_tagspace5000_300.pkl',
            embedding_dim=300,
            name='image_embedding'
        )
        image = image_embedding(doc_id_input)
        image = tf.reshape(image, shape=(-1, 300,))
        image = layers.Dropout(.2)(image)
        image = layers.Dense(self.embedding_dim)(image)
        input_features = text_features + [country, image]

        num_fields = len(input_features)
        features = tf.concat(input_features, axis=1)
        interactions = WeightedFeatureInteraction(num_fields, name='field_weights')(features)

        output = layers.Activation('sigmoid', name='label')(interactions)
        return tf.keras.Model(inputs=inputs, outputs=output, name=self.name)
