import itertools

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.base_model import BaseModel


class NRMF(BaseModel):
    @property
    def name(self):
        return 'nrmf'

    def build(self):
        query_len = 6
        title_len = 20
        ingredient_len = 20
        n_ingredients = 30
        description_len = 100
        n_gram = 3
        embedding_dim = 128

        query_input = keras.Input(shape=(query_len,), name='query_word_ids')
        title_input = keras.Input(shape=(title_len,), name='title_word_ids')
        ingredients_input = keras.Input(shape=(n_ingredients, ingredient_len,), name='ingredients_word_ids')
        description_input = keras.Input(shape=(description_len,), name='description_word_ids')
        author_input = keras.Input(shape=(1,), name='author')
        country_input = keras.Input(shape=(1,), name='country')
        inputs = [query_input, title_input, ingredients_input, description_input, author_input, country_input]

        embedding = layers.Embedding(self.total_words, embedding_dim, mask_zero=True)
        query = embedding(query_input)
        title = embedding(title_input)
        ingredients = embedding(ingredients_input)
        description = embedding(description_input)
        author = layers.Embedding(self.total_authors, embedding_dim)(author_input)
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
        author = tf.reshape(author, shape=(-1, embedding_dim,))
        country = tf.reshape(country, shape=(-1, embedding_dim,))

        query_title = tf.multiply(query, title)
        query_ingredients = tf.multiply(query, ingredients)
        query_description = tf.multiply(query, description)
        query_author = tf.multiply(query, author)
        query_country = tf.multiply(query, country)

        x = layers.concatenate([
            query_title,
            query_ingredients,
            query_description,
            query_author,
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
        query_len = 6
        title_len = 20
        ingredients_len = 300
        description_len = 100

        query_input = keras.Input(shape=(query_len,), name='query_word_ids')
        title_input = keras.Input(shape=(title_len,), name='title_word_ids')
        ingredients_input = keras.Input(shape=(ingredients_len,), name='ingredients_word_ids')
        description_input = keras.Input(shape=(description_len,), name='description_word_ids')
        author_input = keras.Input(shape=(1,), name='author')
        country_input = keras.Input(shape=(1,), name='country')
        inputs = [query_input, title_input, ingredients_input, description_input, author_input, country_input]

        embedding = layers.Embedding(self.total_words, self.embedding_dim)
        query = embedding(query_input)
        title = embedding(title_input)
        ingredients = embedding(ingredients_input)
        description = embedding(description_input)
        author = layers.Embedding(self.total_authors, self.embedding_dim)(author_input)
        country = layers.Embedding(self.total_countries, self.embedding_dim)(country_input)

        query = layers.GlobalMaxPooling1D()(query)
        title = layers.GlobalMaxPooling1D()(title)
        ingredients = layers.GlobalMaxPooling1D()(ingredients)
        description = layers.GlobalMaxPooling1D()(description)
        author = tf.reshape(author, shape=(-1, self.embedding_dim))
        country = tf.reshape(country, shape=(-1, self.embedding_dim))
        fields = [title, ingredients, description, author, country]

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
        query_len = 6
        title_len = 20
        ingredients_len = 300
        description_len = 100

        query_input = keras.Input(shape=(query_len,), name='query_word_ids')
        title_input = keras.Input(shape=(title_len,), name='title_word_ids')
        ingredients_input = keras.Input(shape=(ingredients_len,), name='ingredients_word_ids')
        description_input = keras.Input(shape=(description_len,), name='description_word_ids')
        author_input = keras.Input(shape=(1,), name='author')
        country_input = keras.Input(shape=(1,), name='country')
        inputs = [query_input, title_input, ingredients_input, description_input, author_input, country_input]

        embedding = layers.Embedding(self.total_words, self.embedding_dim)
        query = embedding(query_input)
        title = embedding(title_input)
        ingredients = embedding(ingredients_input)
        description = embedding(description_input)
        author = layers.Embedding(self.total_authors, self.embedding_dim)(author_input)
        country = layers.Embedding(self.total_countries, self.embedding_dim)(country_input)

        query = layers.GlobalMaxPooling1D()(query)
        title = layers.GlobalMaxPooling1D()(title)
        ingredients = layers.GlobalMaxPooling1D()(ingredients)
        description = layers.GlobalMaxPooling1D()(description)
        author = tf.reshape(author, shape=(-1, self.embedding_dim))
        country = tf.reshape(country, shape=(-1, self.embedding_dim))
        features = [query, title, ingredients, description, author, country]

        interactions = []
        for feature1, feature2 in itertools.combinations(features, 2):
            interactions.append(tf.multiply(feature1, feature2))

        x = layers.concatenate(interactions)
        x = layers.Dense(32, activation='relu')(x)
        output = layers.Dense(1, activation='sigmoid', name='label')(x)

        return keras.Model(inputs=inputs, outputs=output, name=self.name)
