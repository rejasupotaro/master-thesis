import itertools
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.base_model import BaseModel

project_dir = Path(__file__).resolve().parents[3]

class NRMF(BaseModel):
    @property
    def name(self):
        return 'nrmf'

    def build(self):
        ingredient_len = 20
        n_ingredients = 30
        n_gram = 3
        embedding_dim = 128

        query_input = self.new_query_input()
        title_input = self.new_title_input()
        ingredients_input = keras.Input(shape=(n_ingredients, ingredient_len,), name='ingredients')
        description_input = self.new_description_input()
        country_input = self.new_country_input()
        inputs = [query_input, title_input, ingredients_input, description_input, country_input]

        embedding = layers.Embedding(self.total_words, embedding_dim, mask_zero=True)
        query = embedding(query_input)
        title = embedding(title_input)
        ingredients = embedding(ingredients_input)
        description = embedding(description_input)
        country = layers.Embedding(self.total_countries, embedding_dim)(country_input)

        query = layers.Conv1D(embedding_dim, n_gram, activation='relu')(query)
        query = layers.GlobalMaxPooling1D()(query)
        title = layers.Conv1D(embedding_dim, n_gram, activation='relu')(title)
        title = layers.GlobalMaxPooling1D()(title)
        ingredients = tf.reshape(ingredients, [-1, ingredient_len, embedding_dim])
        ingredients = layers.Conv1D(embedding_dim, n_gram, activation='relu')(ingredients)
        ingredients = layers.GlobalMaxPooling1D()(ingredients)
        ingredients = tf.reshape(ingredients, [-1, n_ingredients, embedding_dim])
        ingredients = layers.GlobalMaxPooling1D()(ingredients)
        description = layers.Conv1D(embedding_dim, n_gram, activation='relu')(description)
        description = layers.GlobalMaxPooling1D()(description)
        country = tf.reshape(country, shape=(-1, embedding_dim,))

        query_title = tf.multiply(query, title)
        query_ingredients = tf.multiply(query, ingredients)
        query_description = tf.multiply(query, description)
        query_country = tf.multiply(query, country)

        x = layers.concatenate([
            query_title,
            query_ingredients,
            query_description,
            query_country,
        ])
        x = layers.Dense(32, activation='relu')(x)
        output = layers.Dense(1, activation='sigmoid', name='label')(x)

        return keras.Model(inputs=inputs, outputs=output, name=self.name)


class NRMFSimpleQuery(BaseModel):
    @property
    def name(self):
        return 'nrmf_simple_query'

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
        country = layers.Embedding(self.total_countries, self.embedding_dim)(country_input)
        country = tf.reshape(country, shape=(-1, self.embedding_dim))
        image_embedding = self.load_pretrained_embedding(
            embedding_filepath=f'{project_dir}/data/raw/en_2020-03-16T00_04_34_recipe_image_tagspace5000_300.pkl',
            embedding_dim=300,
            name='image_embedding'
        )
        image = image_embedding(doc_id_input)
        image = tf.reshape(image, shape=(-1, 300,))
        image = layers.Dropout(.2)(image)
        image = layers.Dense(self.embedding_dim)(image)
        fields = [title, ingredients, description, country, image]

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
        country = layers.Embedding(self.total_countries, self.embedding_dim)(country_input)
        country = tf.reshape(country, shape=(-1, self.embedding_dim))
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

        interactions = []
        for feature1, feature2 in itertools.combinations(input_features, 2):
            interactions.append(tf.multiply(feature1, feature2))

        x = layers.concatenate(interactions)
        x = layers.Dense(32, activation='relu')(x)
        output = layers.Dense(1, activation='sigmoid', name='label')(x)

        return keras.Model(inputs=inputs, outputs=output, name=self.name)


class NRMFSimpleQueryWith1st(BaseModel):
    @property
    def name(self):
        return 'nrmf_simple_query_with_1st'

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
        country = layers.Embedding(self.total_countries, self.embedding_dim)(country_input)
        country = tf.reshape(country, shape=(-1, self.embedding_dim))
        image_embedding = self.load_pretrained_embedding(
            embedding_filepath=f'{project_dir}/data/raw/en_2020-03-16T00_04_34_recipe_image_tagspace5000_300.pkl',
            embedding_dim=300,
            name='image_embedding'
        )
        image = image_embedding(doc_id_input)
        image = tf.reshape(image, shape=(-1, 300,))
        image = layers.Dropout(.2)(image)
        image = layers.Dense(self.embedding_dim)(image)
        fields = [title, ingredients, description, country, image]

        features = []
        for field in fields:
            features.append(tf.multiply(query, field))
        for feature in [query] + fields:
            feature = layers.Dense(1, activation='relu')(feature)
            features.append(feature)

        x = layers.concatenate(features)
        x = layers.Dense(32, activation='relu')(x)
        output = layers.Dense(1, activation='sigmoid', name='label')(x)

        return keras.Model(inputs=inputs, outputs=output, name=self.name)
