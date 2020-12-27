import tensorflow as tf
from tensorflow.keras import layers

from src.models.base_model import BaseModel


class EBR(BaseModel):
    @property
    def name(self) -> str:
        return 'ebr'

    def build_query_encoder(self, word_embedding):
        query_input = self.new_query_input(size=20)
        inputs = [query_input]
        encoded_query = layers.GlobalMaxPooling1D()(word_embedding(query_input))
        encoder = tf.keras.Model(inputs=inputs, outputs=encoded_query)
        return inputs, encoded_query, encoder

    def build_recipe_encoder(self, word_embedding):
        url_input = self.new_url_input()
        title_input = self.new_title_input()
        body_input = self.new_body_input()
        inputs = [url_input, title_input, body_input]

        url = layers.GlobalMaxPooling1D()(word_embedding(url_input))
        title = layers.GlobalMaxPooling1D()(word_embedding(title_input))
        body = layers.GlobalMaxPooling1D()(word_embedding(body_input))

        encoded_recipe = layers.concatenate([url, title, body])
        encoded_recipe = layers.Dense(self.embedding_dim, activation='relu')(encoded_recipe)

        encoder = tf.keras.Model(inputs=inputs, outputs=encoded_recipe)
        return inputs, encoded_recipe, encoder

    def build(self):
        word_embedding = layers.Embedding(self.total_words, self.embedding_dim)
        query_inputs, encoded_query, query_encoder = self.build_query_encoder(word_embedding)
        recipe_inputs, encoded_recipe, recipe_encoder = self.build_recipe_encoder(word_embedding)
        inputs = query_inputs + recipe_inputs
        output = layers.Dot(axes=1, normalize=True, name='label')([encoded_query, encoded_recipe])
        model = tf.keras.Model(inputs=inputs, outputs=output, name=self.name)
        return model
