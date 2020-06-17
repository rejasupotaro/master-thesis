import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.base_model import BaseModel


class Naive(BaseModel):
    @property
    def name(self) -> str:
        return 'Naive'

    def build(self):
        query_input = keras.Input(shape=(6,), name='query_word_ids')
        title_input = keras.Input(shape=(20,), name='title_word_ids')
        ingredients_input = keras.Input(shape=(300,), name='ingredients_word_ids')
        description_input = keras.Input(shape=(100,), name='description_word_ids')
        country_input = keras.Input(shape=(1,), name='country')

        embedding = layers.Embedding(self.total_words, 64)
        query_features = embedding(query_input)
        title_features = embedding(title_input)
        ingredients_features = embedding(ingredients_input)
        description_features = embedding(description_input)
        country_features = layers.Embedding(self.total_countries, 64)(country_input)

        query_features = layers.GlobalMaxPooling1D()(query_features)
        title_features = layers.GlobalMaxPooling1D()(title_features)
        ingredients_features = layers.GlobalMaxPooling1D()(ingredients_features)
        description_features = layers.GlobalMaxPooling1D()(description_features)
        country_features = tf.reshape(country_features, shape=(-1, 64,))

        x = layers.concatenate([
            query_features,
            title_features,
            ingredients_features,
            description_features,
            country_features
        ])
        x = layers.Dense(32, activation='relu')(x)
        output = layers.Dense(1, activation='sigmoid', name='label')(x)

        return keras.Model(
            inputs=[query_input, title_input, ingredients_input, description_input, country_input],
            outputs=[output],
            name=self.name
        )
