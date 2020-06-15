import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(total_words, total_countries):
    query_input = keras.Input(shape=(6,), name='query_word_ids')
    title_input = keras.Input(shape=(20,), name='title_word_ids')
    ingredients_input = keras.Input(shape=(300,), name='ingredients_word_ids')
    country_input = keras.Input(shape=(1,), name='country')

    embedding = layers.Embedding(total_words, 64)
    query_features = embedding(query_input)
    title_features = embedding(title_input)
    ingredients_features = embedding(ingredients_input)
    country_features = layers.Embedding(total_countries, 64)(country_input)

    query_features = layers.GlobalAveragePooling1D()(query_features)
    title_features = layers.GlobalAveragePooling1D()(title_features)
    ingredients_features = layers.GlobalAveragePooling1D()(ingredients_features)
    country_features = tf.reshape(country_features, shape=(-1, 64,))

    x = layers.concatenate([query_features, title_features, ingredients_features, country_features])
    x = layers.Dense(32, activation='relu')(x)
    output = layers.Dense(1, activation='sigmoid', name='label')(x)

    return keras.Model(
        inputs=[query_input, title_input, ingredients_input, country_input],
        outputs=[output],
        name='Naive'
    )
