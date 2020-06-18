import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.base_model import BaseModel


class Naive(BaseModel):
    @property
    def name(self) -> str:
        return 'Naive'

    def build(self):
        query_len = 6
        title_len = 20
        ingredients_len = 300
        description_len = 100
        embedding_dim = 128

        query_input = keras.Input(shape=(query_len,), name='query_word_ids')
        title_input = keras.Input(shape=(title_len,), name='title_word_ids')
        ingredients_input = keras.Input(shape=(ingredients_len,), name='ingredients_word_ids')
        description_input = keras.Input(shape=(description_len,), name='description_word_ids')
        author_input = keras.Input(shape=(1,), name='author')
        country_input = keras.Input(shape=(1,), name='country')

        embedding = layers.Embedding(self.total_words, embedding_dim)
        query_features = embedding(query_input)
        title_features = embedding(title_input)
        ingredients_features = embedding(ingredients_input)
        description_features = embedding(description_input)
        author_features = layers.Embedding(self.total_authors, embedding_dim)(author_input)
        country_features = layers.Embedding(self.total_countries, embedding_dim)(country_input)

        query_features = layers.GlobalMaxPooling1D()(query_features)
        title_features = layers.GlobalMaxPooling1D()(title_features)
        ingredients_features = layers.GlobalMaxPooling1D()(ingredients_features)
        description_features = layers.GlobalMaxPooling1D()(description_features)
        author_features = tf.reshape(author_features, shape=(-1, embedding_dim,))
        country_features = tf.reshape(country_features, shape=(-1, embedding_dim,))

        x = layers.concatenate([
            query_features,
            title_features,
            ingredients_features,
            description_features,
            author_features,
            country_features
        ])
        x = layers.Dense(32, activation='relu')(x)
        output = layers.Dense(1, activation='sigmoid', name='label')(x)

        return keras.Model(
            inputs=[query_input, title_input, ingredients_input, description_input, author_input, country_input],
            outputs=[output],
            name=self.name
        )

