import tensorflow as tf
from tensorflow.keras import layers

from src.models.base_model import BaseModel


class EBR(BaseModel):
    @property
    def name(self) -> str:
        return 'ebr'

    def build_query_encoder(self, embedding):
        query_input = self.new_query_input()
        inputs = [query_input]
        query = embedding(query_input)
        encoded_query = layers.GlobalMaxPooling1D()(query)
        encoder = tf.keras.Model(inputs=inputs, outputs=encoded_query)
        return inputs, encoded_query, encoder

    def build_recipe_encoder(self, embedding):
        title_input = self.new_title_input()
        ingredients_input = self.new_ingredients_input()
        description_input = self.new_description_input()
        country_input = self.new_country_input()
        inputs = [title_input, ingredients_input, description_input, country_input]

        title = embedding(title_input)
        ingredients = embedding(ingredients_input)
        description = embedding(description_input)
        country = layers.Embedding(self.total_countries, self.embedding_dim)(country_input)

        title = layers.GlobalMaxPooling1D()(title)
        ingredients = layers.GlobalMaxPooling1D()(ingredients)
        description = layers.GlobalMaxPooling1D()(description)
        country = tf.reshape(country, shape=(-1, self.embedding_dim,))

        encoded_recipe = layers.concatenate([
            title,
            ingredients,
            description,
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
