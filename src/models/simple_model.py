import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(total_words, total_countries):
    query_input = keras.Input(shape=(6,), dtype=tf.int32, name='query_word_ids')
    title_input = keras.Input(shape=(20,), dtype=tf.int32, name='title_word_ids')
    desc_input = keras.Input(shape=(500,), dtype=tf.int32, name='desc_word_ids')
    country_input = keras.Input(shape=(1,), dtype=tf.int32, name='country')

    embedding = layers.Embedding(total_words, 64)
    query_features = embedding(query_input)
    title_features = embedding(title_input)
    desc_features = embedding(desc_input)
    country_features = layers.Embedding(total_countries, 64)(country_input)

    query_features = layers.GlobalAveragePooling1D()(query_features)
    title_features = layers.GlobalAveragePooling1D()(title_features)
    desc_features = layers.GlobalAveragePooling1D()(desc_features)
    country_features = tf.reshape(country_features, shape=(-1, 64,))

    query_title_features = tf.multiply(query_features, title_features)
    query_desc_features = tf.multiply(query_features, desc_features)
    query_country_features = tf.multiply(query_features, country_features)

    x = layers.concatenate([query_title_features, query_desc_features, query_country_features])
    x = layers.Dense(32, activation='relu')(x)
    output = layers.Dense(1, activation='sigmoid', name='relevance')(x)

    return keras.Model(
        inputs=[query_input, title_input, desc_input, country_input],
        outputs=[output],
        name='NRM-F'
    )
