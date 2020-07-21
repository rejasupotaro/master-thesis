import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.base_model import BaseModel


class NRMFConcat(BaseModel):
    @property
    def name(self) -> str:
        return 'NRM-F-Concat'

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

        embedding = layers.Embedding(self.total_words, self.embedding_dim)
        query_features = embedding(query_input)
        title_features = embedding(title_input)
        ingredients_features = embedding(ingredients_input)
        description_features = embedding(description_input)
        author_features = layers.Embedding(self.total_authors, self.embedding_dim)(author_input)
        country_features = layers.Embedding(self.total_countries, self.embedding_dim)(country_input)

        query_features = layers.GlobalMaxPooling1D()(query_features)
        title_features = layers.GlobalMaxPooling1D()(title_features)
        ingredients_features = layers.GlobalMaxPooling1D()(ingredients_features)
        description_features = layers.GlobalMaxPooling1D()(description_features)
        author_features = tf.reshape(author_features, shape=(-1, self.embedding_dim,))
        country_features = tf.reshape(country_features, shape=(-1, self.embedding_dim,))

        query_title_features = tf.multiply(query_features, title_features)
        query_ingredients_features = tf.multiply(query_features, ingredients_features)
        query_description_features = tf.multiply(query_features, description_features)
        query_author_features = tf.multiply(query_features, author_features)
        query_country_features = tf.multiply(query_features, country_features)

        x = layers.concatenate([
            query_title_features,
            query_ingredients_features,
            query_description_features,
            query_author_features,
            query_country_features
        ])
        x = layers.Dense(32, activation='relu')(x)
        output = layers.Dense(1, activation='sigmoid', name='label')(x)

        return keras.Model(
            inputs=[query_input, title_input, ingredients_input, description_input, author_input, country_input],
            outputs=[output],
            name=self.name
        )
