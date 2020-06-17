import abc

import os
import pickle
from pathlib import Path

import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm

from src.data.recipes import load_recipes


class DataProcessor(abc.ABC):
    def __init__(self, batch_size=128):
        self.recipes = load_recipes()
        self.tokenizer = None
        self.country_encoder = None
        self.batch_size = batch_size

    @property
    def total_words(self):
        return len(self.tokenizer.word_index) + 1

    @property
    def total_countries(self):
        return len(self.country_encoder.classes_)

    def listwise_to_df(self, listwise_filename: str, max_negatives: int = 10) -> pd.DataFrame:
        project_dir = Path(__file__).resolve().parents[2]
        with open(os.path.join(project_dir, 'data', 'processed', listwise_filename), 'rb') as file:
            dataset = pickle.load(file)

        rows = []
        for example in tqdm(dataset):
            query = example['query']
            positives = []
            negatives = []
            for doc in example['docs']:
                data = {
                    'query': query,
                    'doc_id': doc['doc_id'],
                    'label': doc['label']
                }
                if doc['label'] == 1:
                    positives.append(data)
                else:
                    negatives.append(data)

            for positive in positives:
                for negative in negatives[:max_negatives]:
                    rows.append(positive)
                    rows.append(negative)
        return pd.DataFrame(rows)

    @abc.abstractmethod
    def fit(self, df: pd.DataFrame):
        raise NotImplementedError('Calling an abstract method.')

    @abc.abstractmethod
    def transform(self, df: pd.DataFrame) -> tf.data.Dataset:
        raise NotImplementedError('Calling an abstract method.')

    def fit_transform(self, df: pd.DataFrame) -> tf.data.Dataset:
        self.fit(df)
        return self.transform(df)


class ConcatDataProcessor(DataProcessor):
    def fit(self, df: pd.DataFrame):
        df['query'] = df['query'].astype(str)
        df['title'] = df['doc_id'].apply(lambda doc_id: self.recipes[doc_id]['title']).astype(str)
        df['ingredients'] = df['doc_id'].apply(lambda doc_id: self.recipes[doc_id]['ingredients'])
        df['ingredients'] = df['ingredients'].apply(lambda x: ' '.join(x))
        df['description'] = df['doc_id'].apply(
            lambda doc_id: self.recipes[doc_id]['story_or_description'])
        df['description'].fillna('', inplace=True)
        df['country'] = df['doc_id'].apply(lambda doc_id: self.recipes[doc_id]['country'])

        oov_token = '<oov>'
        sentences = []
        sentences += df['query'].tolist()
        sentences += df['title'].tolist()
        sentences += df['ingredients'].tolist()
        sentences += df['description'].tolist()
        self.tokenizer = Tokenizer(
            oov_token=oov_token,
            char_level=False
        )
        self.tokenizer.fit_on_texts(sentences)

        self.country_encoder = LabelEncoder()
        self.country_encoder.fit(df['country'].tolist() + [''])

    def transform(self, df: pd.DataFrame) -> tf.data.Dataset:
        df['query'] = df['query'].astype(str)
        df['title'] = df['doc_id'].apply(lambda doc_id: self.recipes[doc_id]['title']).astype(str)
        df['ingredients'] = df['doc_id'].apply(lambda doc_id: self.recipes[doc_id]['ingredients'])
        df['ingredients'] = df['ingredients'].apply(lambda x: ' '.join(x))
        df['description'] = df['doc_id'].apply(
            lambda doc_id: self.recipes[doc_id]['story_or_description'])
        df['description'].fillna('', inplace=True)
        df['country'] = df['doc_id'].apply(lambda doc_id: self.recipes[doc_id]['country'])

        df['query_word_ids'] = self.tokenizer.texts_to_sequences(df['query'].tolist())
        df['title_word_ids'] = self.tokenizer.texts_to_sequences(df['title'].tolist())
        df['ingredients_word_ids'] = self.tokenizer.texts_to_sequences(df['ingredients'].tolist())
        df['description_word_ids'] = self.tokenizer.texts_to_sequences(df['description'].tolist())

        df['country'] = df['country'].apply(lambda c: c if c in self.country_encoder.classes_ else '')
        df['country'] = self.country_encoder.transform(df['country'])

        query_word_ids = df['query_word_ids'].tolist()
        query_word_ids = pad_sequences(query_word_ids,
                                       padding='post',
                                       truncating='post',
                                       maxlen=6)

        title_word_ids = df['title_word_ids'].tolist()
        title_word_ids = pad_sequences(title_word_ids,
                                       padding='post',
                                       truncating='post',
                                       maxlen=20)
        ingredients_word_ids = df['ingredients_word_ids'].tolist()
        ingredients_word_ids = pad_sequences(ingredients_word_ids,
                                             padding='post',
                                             truncating='post',
                                             maxlen=300)
        description_word_ids = df['description_word_ids'].tolist()
        description_word_ids = pad_sequences(description_word_ids,
                                             padding='post',
                                             truncating='post',
                                             maxlen=100)
        country = df['country'].tolist()
        label = df['label'].tolist()

        return tf.data.Dataset.from_tensor_slices((
            {
                'query_word_ids': query_word_ids,
                'title_word_ids': title_word_ids,
                'ingredients_word_ids': ingredients_word_ids,
                'description_word_ids': description_word_ids,
                'country': country
            },
            {'label': label}
        )).batch(self.batch_size)


class MultiInstanceDataProcessor(DataProcessor):
    def fit(self, df: pd.DataFrame):
        df['query'] = df['query'].astype(str)
        df['title'] = df['doc_id'].apply(lambda doc_id: self.recipes[doc_id]['title']).astype(str)
        df['ingredients'] = df['doc_id'].apply(lambda doc_id: self.recipes[doc_id]['ingredients'])
        df['description'] = df['doc_id'].apply(
            lambda doc_id: self.recipes[doc_id]['story_or_description'])
        df['description'].fillna('', inplace=True)
        df['country'] = df['doc_id'].apply(lambda doc_id: self.recipes[doc_id]['country'])

        oov_token = '<OOV>'
        sentences = []
        sentences += df['query'].tolist()
        sentences += df['title'].tolist()
        sentences += [ingredient for ingredients in df['ingredients'].tolist() for ingredient in ingredients]
        self.tokenizer = Tokenizer(
            oov_token=oov_token,
            char_level=False
        )
        self.tokenizer.fit_on_texts(sentences)

        self.country_encoder = LabelEncoder()
        self.country_encoder.fit(df['country'].tolist() + [''])

    def transform(self, df: pd.DataFrame) -> tf.data.Dataset:
        df['query'] = df['query'].astype(str)
        df['title'] = df['doc_id'].apply(lambda doc_id: self.recipes[doc_id]['title']).astype(str)
        df['ingredients'] = df['doc_id'].apply(lambda doc_id: self.recipes[doc_id]['ingredients'])
        df['description'] = df['doc_id'].apply(
            lambda doc_id: self.recipes[doc_id]['story_or_description'])
        df['description'].fillna('', inplace=True)
        df['country'] = df['doc_id'].apply(lambda doc_id: self.recipes[doc_id]['country'])

        df['query_word_ids'] = self.tokenizer.texts_to_sequences(df['query'].tolist())
        df['title_word_ids'] = self.tokenizer.texts_to_sequences(df['title'].tolist())
        df['ingredients_word_ids'] = [self.tokenizer.texts_to_sequences(ingredients) for ingredients in
                                      df['ingredients']]
        df['description_word_ids'] = self.tokenizer.texts_to_sequences(df['description'].tolist())

        df['country'] = df['country'].apply(lambda c: c if c in self.country_encoder.classes_ else '')
        df['country'] = self.country_encoder.transform(df['country'])

        query_word_ids = df['query_word_ids'].tolist()
        query_word_ids = pad_sequences(query_word_ids,
                                       padding='post',
                                       truncating='post',
                                       maxlen=6)

        title_word_ids = df['title_word_ids'].tolist()
        title_word_ids = pad_sequences(title_word_ids,
                                       padding='post',
                                       truncating='post',
                                       maxlen=20)
        ingredients_word_ids = df['ingredients_word_ids'].tolist()
        ingredients_word_ids = [pad_sequences(word_ids, padding='post', truncating='post', maxlen=20) for word_ids in
                                ingredients_word_ids]
        ingredients_word_ids = pad_sequences(ingredients_word_ids, padding='post', truncating='post', maxlen=30)
        description_word_ids = df['description_word_ids'].tolist()
        description_word_ids = pad_sequences(description_word_ids,
                                             padding='post',
                                             truncating='post',
                                             maxlen=100)
        country = df['country'].tolist()
        label = df['label'].tolist()

        return tf.data.Dataset.from_tensor_slices((
            {
                'query_word_ids': query_word_ids,
                'title_word_ids': title_word_ids,
                'ingredients_word_ids': ingredients_word_ids,
                'description_word_ids': description_word_ids,
                'country': country
            },
            {'label': label}
        )).batch(self.batch_size)
