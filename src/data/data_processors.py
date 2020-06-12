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
    def __init__(self):
        self.recipes = load_recipes()

    def listwise_to_dataset(self, listwise_filename, tokenizer=None, country_encoder=None):
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
                i, j = len(positives) - 1, len(negatives) - 1

                while i >= 0 and j >= 0:
                    rows.append(positives[i])
                    rows.append(negatives[j])
                    i -= 1
                    j -= 1
        df = pd.DataFrame(rows)
        return self.process(df, tokenizer, country_encoder)

    @abc.abstractmethod
    def process(self, df, tokenizer, country_encoder):
        raise NotImplementedError('Calling an abstract method.')


class ConcatDataProcessor(DataProcessor):
    def process(self, df, tokenizer, country_encoder):
        recipes = load_recipes()

        df['query'] = df['query'].astype(str)
        df['title'] = df['doc_id'].apply(lambda doc_id: recipes[doc_id]['title']).astype(str)
        df['ingredients'] = df['doc_id'].apply(lambda doc_id: recipes[doc_id]['ingredients'])
        df['ingredients'] = df['ingredients'].apply(lambda x: ' '.join(x))
        df['country'] = df['doc_id'].apply(lambda doc_id: recipes[doc_id]['country'])

        if not tokenizer:
            oov_token = '<OOV>'
            sentences = []
            for key in ['query', 'title', 'ingredients']:
                sentences += df[key].tolist()
            tokenizer = Tokenizer(
                oov_token=oov_token,
                char_level=False
            )
            tokenizer.fit_on_texts(sentences)

        for feature in ['query', 'title', 'ingredients']:
            df[f'{feature}_word_ids'] = tokenizer.texts_to_sequences(df[feature].tolist())

        if not country_encoder:
            country_encoder = LabelEncoder()
            country_encoder.fit(df['country'].tolist() + [''])
        else:
            df['country'] = df['country'].apply(lambda c: c if c in country_encoder.classes_ else '')
        df['country'] = country_encoder.transform(df['country'])

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
        country = df['country'].tolist()
        label = df['label'].tolist()

        dataset = tf.data.Dataset.from_tensor_slices((
            {
                'query_word_ids': query_word_ids,
                'title_word_ids': title_word_ids,
                'ingredients_word_ids': ingredients_word_ids,
                'country': country
            },
            {'label': label}
        )).batch(32)

        return dataset, tokenizer, country_encoder


class MultiInstanceDataProcessor(DataProcessor):
    def process(self, df, tokenizer, country_encoder):
        recipes = load_recipes()

        df['query'] = df['query'].astype(str)
        df['title'] = df['doc_id'].apply(lambda doc_id: recipes[doc_id]['title']).astype(str)
        df['ingredients'] = df['doc_id'].apply(lambda doc_id: recipes[doc_id]['ingredients'])
        df['country'] = df['doc_id'].apply(lambda doc_id: recipes[doc_id]['country'])

        if not tokenizer:
            oov_token = '<OOV>'
            sentences = []
            sentences += df['query'].tolist()
            sentences += df['title'].tolist()
            sentences += [ingredient for ingredients in df['ingredients'].tolist() for ingredient in ingredients]
            tokenizer = Tokenizer(
                oov_token=oov_token,
                char_level=False
            )
            tokenizer.fit_on_texts(sentences)

        df['query_word_ids'] = tokenizer.texts_to_sequences(df['query'].tolist())
        df['title_word_ids'] = tokenizer.texts_to_sequences(df['title'].tolist())
        df['ingredients_word_ids'] = [tokenizer.texts_to_sequences(ingredients) for ingredients in df['ingredients']]

        if not country_encoder:
            country_encoder = LabelEncoder()
            country_encoder.fit(df['country'].tolist() + [''])
        else:
            df['country'] = df['country'].apply(lambda c: c if c in country_encoder.classes_ else '')
        df['country'] = country_encoder.transform(df['country'])

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
        country = df['country'].tolist()
        label = df['label'].tolist()

        dataset = tf.data.Dataset.from_tensor_slices((
            {
                'query_word_ids': query_word_ids,
                'title_word_ids': title_word_ids,
                'ingredients_word_ids': ingredients_word_ids,
                'country': country
            },
            {'label': label}
        )).batch(32)

        return dataset, tokenizer, country_encoder
