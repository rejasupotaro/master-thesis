import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.base_model import BaseModel


class AutoInt(BaseModel):
    @property
    def name(self) -> str:
        return 'autoint'

    def build(self):
        query_len = 6
        title_len = 20
        ingredients_len = 300
        description_len = 100

        categorical_input_size = {
            'author': self.total_authors,
            'country': self.total_countries,
        }

        query_input = keras.Input(shape=(query_len,), name='query_word_ids')
        title_input = keras.Input(shape=(title_len,), name='title_word_ids')
        ingredients_input = keras.Input(shape=(ingredients_len,), name='ingredients_word_ids')
        description_input = keras.Input(shape=(description_len,), name='description_word_ids')
        text_inputs = [query_input, title_input, ingredients_input, description_input]

        author_input = keras.Input(shape=(1,), name='author')
        country_input = keras.Input(shape=(1,), name='country')
        categorical_inputs = [author_input, country_input]

        inputs = text_inputs + categorical_inputs

        embedding = layers.Embedding(self.total_words, self.embedding_dim)
        text_features = [embedding(text_input) for text_input in text_inputs]
        text_features = [layers.GlobalMaxPooling1D()(feature) for feature in text_features]

        categorical_features = []
        for name, categorical_input in zip(categorical_input_size, categorical_inputs):
            embedding = layers.Embedding(categorical_input_size[name], self.embedding_dim)
            feature = embedding(categorical_input)
            feature = tf.reshape(feature, shape=(-1, self.embedding_dim,))
            categorical_features.append(feature)

        features = text_features + categorical_features

        features = layers.concatenate(features)
        x = layers.Attention()([features, features])

        output = layers.Dense(1, activation='sigmoid', name='label')(x)
        return tf.keras.Model(inputs=inputs, outputs=output, name=self.name)
