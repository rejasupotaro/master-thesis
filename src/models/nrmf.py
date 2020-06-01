import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def ngram_block(n, total_words):
    def wrapped(inputs):
        layer = layers.Conv1D(1, n, use_bias=False, trainable=False)
        x = layers.Reshape((-1, 1))(inputs)
        x = layer(x)
        kernel = np.power(total_words, range(0, n))
        layer.set_weights([kernel.reshape(n, 1, 1)])
        return layers.Reshape((-1,))(x)

    return wrapped


def build_model(total_words, total_countries):
    query_input = keras.Input(shape=(6,), name='query_word_ids')
    title_input = keras.Input(shape=(120,), name='title_word_ids')
    ingredients_input = keras.Input(shape=(1500,), name='ingredients_word_ids')
    desc_input = keras.Input(shape=(3000,), name='desc_word_ids')
    country_input = keras.Input(shape=(1,), name='country')

    ngram = 2
    query_features = ngram_block(ngram, total_words)(query_input)
    title_features = ngram_block(ngram, total_words)(title_input)
    ingredients_features = ngram_block(ngram, total_words)(ingredients_input)
    desc_features = ngram_block(ngram, total_words)(desc_input)

    embedding = layers.Embedding(pow(total_words, ngram), 64)
    query_features = embedding(query_features)
    title_features = embedding(title_features)
    ingredients_features = embedding(ingredients_features)
    desc_features = embedding(desc_features)
    country_features = layers.Embedding(total_countries, 64)(country_input)

    #     query_features = layers.GlobalAveragePooling1D()(query_features)
    #     title_features = layers.GlobalAveragePooling1D()(title_features)
    #     ingredients_features = layers.GlobalAveragePooling1D()(ingredients_features)
    #     desc_features = layers.GlobalAveragePooling1D()(desc_features)
    query_features = tf.reduce_mean(layers.Conv1D(64, 3, activation='relu')(query_features), axis=1)
    title_features = tf.reduce_mean(layers.Conv1D(64, 3, activation='relu')(title_features), axis=1)
    ingredients_features = tf.reduce_mean(layers.Conv1D(64, 3, activation='relu')(ingredients_features), axis=1)
    desc_features = tf.reduce_mean(layers.Conv1D(64, 3, activation='relu')(desc_features), axis=1)
    country_features = tf.reshape(country_features, shape=(-1, 64,))

    query_title_features = tf.multiply(query_features, title_features)
    query_ingredients_features = tf.multiply(query_features, ingredients_features)
    query_desc_features = tf.multiply(query_features, desc_features)
    query_country_features = tf.multiply(query_features, country_features)

    x = layers.concatenate(
        [query_title_features, query_ingredients_features, query_desc_features, query_country_features])
    x = layers.Dense(32, activation='relu')(x)
    output = layers.Dense(1, activation='sigmoid', name='relevance')(x)

    return keras.Model(
        inputs=[query_input, title_input, ingredients_input, desc_input, country_input],
        outputs=[output],
        name='NRM-F'
    )
