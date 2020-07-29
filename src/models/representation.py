import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.base_model import BaseModel


class EBR(BaseModel):
    @property
    def name(self) -> str:
        return 'ebr'

    def build_query_encoder(self, embedding):
        query_len = 6
        query_input = keras.Input(shape=(query_len,), name='query_word_ids')
        inputs = [query_input]
        query = embedding(query_input)
        encoded_query = layers.GlobalMaxPooling1D()(query)
        encoder = tf.keras.Model(inputs=inputs, outputs=encoded_query)
        return inputs, encoded_query, encoder

    def build_recipe_encoder(self, embedding):
        title_len = 20
        ingredients_len = 300
        description_len = 100
        title_input = keras.Input(shape=(title_len,), name='title_word_ids')
        ingredients_input = keras.Input(shape=(ingredients_len,), name='ingredients_word_ids')
        description_input = keras.Input(shape=(description_len,), name='description_word_ids')
        author_input = keras.Input(shape=(1,), name='author')
        country_input = keras.Input(shape=(1,), name='country')
        inputs = [title_input, ingredients_input, description_input, author_input, country_input]

        title = embedding(title_input)
        ingredients = embedding(ingredients_input)
        description = embedding(description_input)
        author = layers.Embedding(self.total_authors, self.embedding_dim)(author_input)
        country = layers.Embedding(self.total_countries, self.embedding_dim)(country_input)

        title = layers.GlobalMaxPooling1D()(title)
        ingredients = layers.GlobalMaxPooling1D()(ingredients)
        description = layers.GlobalMaxPooling1D()(description)
        author = tf.reshape(author, shape=(-1, self.embedding_dim,))
        country = tf.reshape(country, shape=(-1, self.embedding_dim,))

        encoded_recipe = layers.concatenate([
            title,
            ingredients,
            description,
            author,
            country
        ])
        encoded_recipe = layers.Dense(self.embedding_dim, activation='relu')(encoded_recipe)

        encoder = tf.keras.Model(inputs=inputs, outputs=encoded_recipe)
        return inputs, encoded_recipe, encoder

    def build(self):
        embedding = layers.Embedding(self.total_words, self.embedding_dim)
        query_inputs, encoded_query, query_encoder = self.build_query_encoder(embedding)
        recipe_inputs, encoded_recipe, recipe_encoder = self.build_recipe_encoder(embedding)
        inputs = query_inputs + recipe_inputs
        output = layers.Dot(axes=1, normalize=True, name='label')([encoded_query, encoded_recipe])
        model = tf.keras.Model(inputs=inputs, outputs=output, name=self.name)
        return model
