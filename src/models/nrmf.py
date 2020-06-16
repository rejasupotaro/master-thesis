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
    query_len = 6
    title_len = 20
    ingredient_len = 20
    n_ingredients = 30
    n_gram = 3
    embedding_dim = 64

    query_input = keras.Input(shape=(query_len,), name='query_word_ids')
    title_input = keras.Input(shape=(title_len,), name='title_word_ids')
    ingredients_input = keras.Input(shape=(n_ingredients, ingredient_len,), name='ingredients_word_ids')
    country_input = keras.Input(shape=(1,), name='country')

    embedding = layers.Embedding(total_words, embedding_dim, mask_zero=True)
    query_features = embedding(query_input)
    title_features = embedding(title_input)
    ingredients_features = embedding(ingredients_input)
    country_features = layers.Embedding(total_countries, embedding_dim)(country_input)

    query_features = layers.Conv1D(embedding_dim, n_gram, activation='relu')(query_features)
    query_features = layers.GlobalAveragePooling1D()(query_features)
    title_features = tf.reduce_mean(layers.Conv1D(embedding_dim, n_gram, activation='relu')(title_features), axis=1)
    ingredients_features = tf.reshape(ingredients_features, [-1, ingredient_len, 64])
    ingredients_features = layers.Conv1D(embedding_dim, n_gram, activation='relu')(ingredients_features)
    ingredients_features = layers.GlobalAveragePooling1D()(ingredients_features)
    ingredients_features = tf.reshape(ingredients_features, [-1, n_ingredients, 64])
    ingredients_features = layers.GlobalAveragePooling1D()(ingredients_features)
    country_features = tf.reshape(country_features, shape=(-1, embedding_dim,))

    query_title_features = tf.multiply(query_features, title_features)
    query_ingredients_features = tf.multiply(query_features, ingredients_features)
    query_country_features = tf.multiply(query_features, country_features)

    x = layers.concatenate(
        [query_title_features, query_ingredients_features, query_country_features])
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(16, activation='relu')(x)
    output = layers.Dense(1, activation='sigmoid', name='label')(x)

    return keras.Model(
        inputs=[query_input, title_input, ingredients_input, country_input],
        outputs=[output],
        name='NRM-F'
    )
