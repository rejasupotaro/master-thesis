import tensorflow as tf
from tensorflow.keras import layers

from src.models.base_model import BaseModel


class Attention(BaseModel):
    @property
    def name(self) -> str:
        return 'attention'

    def build(self):
        categorical_input_size = {
            'country': self.total_countries,
        }

        text_inputs = [
            self.new_query_input(),
            self.new_title_input(),
            self.new_ingredients_input(),
            self.new_description_input(),
        ]
        categorical_inputs = [
            self.new_country_input(),
        ]
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
