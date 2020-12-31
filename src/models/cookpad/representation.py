from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers

from src.models.base_model import BaseModel

project_dir = Path(__file__).resolve().parents[3]


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
        doc_id_input = self.new_doc_id_input()
        inputs = [title_input, ingredients_input, description_input, country_input, doc_id_input]

        title = layers.GlobalMaxPooling1D()(embedding(title_input))
        ingredients = layers.GlobalMaxPooling1D()(embedding(ingredients_input))
        description = layers.GlobalMaxPooling1D()(embedding(description_input))
        country_embedding = layers.Embedding(self.total_countries, self.embedding_dim)
        country = country_embedding(country_input)
        country = tf.reshape(country, shape=(-1, self.embedding_dim,))
        image_embedding = self.load_pretrained_embedding(
            embedding_filepath=f'{project_dir}/data/raw/en_2020-03-16T00_04_34_recipe_image_tagspace5000_300.pkl',
            embedding_dim=300,
            name='image_embedding'
        )
        image = image_embedding(doc_id_input)
        image = tf.reshape(image, shape=(-1, 300,))
        image = layers.Dropout(.2)(image)
        image = layers.Dense(self.embedding_dim)(image)

        encoded_recipe = layers.concatenate([
            title,
            ingredients,
            description,
            country,
            image,
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
