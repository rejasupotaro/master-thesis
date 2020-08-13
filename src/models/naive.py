import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.base_model import BaseModel


class Naive(BaseModel):
    @property
    def name(self) -> str:
        return 'naive'

    def build(self):
        query_input = self.new_query_input()
        title_input = self.new_title_input()
        ingredients_input = self.new_ingredients_input()
        description_input = self.new_description_input()
        country_input = self.new_country_input()
        inputs = [query_input, title_input, ingredients_input, description_input, country_input]

        embedding = layers.Embedding(self.total_words, self.embedding_dim)
        query = embedding(query_input)
        title = embedding(title_input)
        ingredients = embedding(ingredients_input)
        description = embedding(description_input)
        country = layers.Embedding(self.total_countries, self.embedding_dim)(country_input)

        query = layers.GlobalMaxPooling1D()(query)
        title = layers.GlobalMaxPooling1D()(title)
        ingredients = layers.GlobalMaxPooling1D()(ingredients)
        description = layers.GlobalMaxPooling1D()(description)
        country = tf.reshape(country, shape=(-1, self.embedding_dim,))

        x = layers.concatenate([query, title, ingredients, description, country])
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dense(8, activation='relu')(x)
        output = layers.Dense(1, activation='sigmoid', name='label')(x)

        return keras.Model(inputs=inputs, outputs=output, name=self.name)
