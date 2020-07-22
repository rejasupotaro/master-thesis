import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.base_model import BaseModel


class Naive(BaseModel):
    @property
    def name(self) -> str:
        return 'naive'

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
        author = tf.reshape(author, shape=(-1, self.embedding_dim,))
        country = tf.reshape(country, shape=(-1, self.embedding_dim,))

        x = layers.concatenate([query, title, ingredients, description, author, country])
        x = layers.Dense(32, activation='relu')(x)
        output = layers.Dense(1, activation='sigmoid', name='label')(x)

        return keras.Model(inputs=inputs, outputs=output, name=self.name)
