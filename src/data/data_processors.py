import abc
import os
import pickle
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm

from src.data.recipes import load_recipes


class DataProcessor(abc.ABC):
    def __init__(self, dataset_size: str, num_words: int = 200000, max_negatives: int = 10, batch_size: int = 128):
        self.recipes = load_recipes(dataset_size)
        self.num_words: int = num_words
        self.tokenizer: Optional[Tokenizer] = None
        self.encoder: Dict[str, LabelEncoder] = {}
        self.max_negatives: int = max_negatives
        self.batch_size: int = batch_size

    @property
    def total_words(self) -> int:
        return len(self.tokenizer.word_index) + 1

    @property
    def total_authors(self) -> int:
        return len(self.encoder['author'].classes_)

    @property
    def total_countries(self) -> int:
        return len(self.encoder['country'].classes_)

    def listwise_to_df(self, listwise_filename: str) -> pd.DataFrame:
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
                for negative in negatives[:self.max_negatives]:
                    rows.append(positive)
                    rows.append(negative)
        return pd.DataFrame(rows)

    @abc.abstractmethod
    def process_df(self, df: pd.DataFrame):
        raise NotImplementedError('Calling an abstract method.')

    @abc.abstractmethod
    def fit(self, df: pd.DataFrame):
        raise NotImplementedError('Calling an abstract method.')

    @abc.abstractmethod
    def process_batch(self, df: pd.DataFrame):
        raise NotImplementedError('Calling an abstract method.')


class ConcatDataProcessor(DataProcessor):
    def process_df(self, df: pd.DataFrame):
        df['query'] = df['query'].astype(str)
        df['title'] = df['doc_id'].apply(lambda doc_id: self.recipes[doc_id]['title']).astype(str)
        df['ingredients'] = df['doc_id'].apply(lambda doc_id: self.recipes[doc_id]['ingredients'])
        df['ingredients'] = df['ingredients'].apply(lambda x: ' '.join(x))
        df['description'] = df['doc_id'].apply(
            lambda doc_id: self.recipes[doc_id]['story_or_description'])
        df['description'].fillna('', inplace=True)
        df['author'] = df['doc_id'].apply(lambda doc_id: self.recipes[doc_id]['user_id'])
        df['country'] = df['doc_id'].apply(lambda doc_id: self.recipes[doc_id]['country'])
        df['label'] = df['label'].astype(float)

    def fit(self, df: pd.DataFrame):
        self.process_df(df)

        sentences = set()
        sentences |= set(df['query'])
        sentences |= set(df['title'])
        sentences |= set(df['ingredients'])
        sentences |= set(df['description'])
        self.tokenizer = Tokenizer(
            oov_token='<OOV>',
            char_level=False,
            num_words=self.num_words,
        )
        self.tokenizer.fit_on_texts(sentences)
        del sentences

        self.encoder['author'] = LabelEncoder()
        self.encoder['author'].fit(list(set(df['author']) | {''}))

        self.encoder['country'] = LabelEncoder()
        self.encoder['country'].fit(list(set(df['country']) | {''}))

    def process_batch(self, df: pd.DataFrame):
        df = df.copy()
        self.process_df(df)

        df['query_word_ids'] = self.tokenizer.texts_to_sequences(df['query'].tolist())
        df['title_word_ids'] = self.tokenizer.texts_to_sequences(df['title'].tolist())
        df['ingredients_word_ids'] = self.tokenizer.texts_to_sequences(df['ingredients'].tolist())
        df['description_word_ids'] = self.tokenizer.texts_to_sequences(df['description'].tolist())

        df['author'] = df['author'].apply(lambda c: c if c in self.encoder['author'].classes_ else '')
        df['author'] = self.encoder['author'].transform(df['author'])

        df['country'] = df['country'].apply(lambda c: c if c in self.encoder['country'].classes_ else '')
        df['country'] = self.encoder['country'].transform(df['country'])

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
        author = df['author'].to_numpy()
        country = df['country'].to_numpy()
        label = df['label'].to_numpy()

        return {
                   'query_word_ids': query_word_ids,
                   'title_word_ids': title_word_ids,
                   'ingredients_word_ids': ingredients_word_ids,
                   'description_word_ids': description_word_ids,
                   'author': author,
                   'country': country
               }, label


class MultiInstanceDataProcessor(DataProcessor):
    def process_df(self, df: pd.DataFrame):
        df['query'] = df['query'].astype(str)
        df['title'] = df['doc_id'].apply(lambda doc_id: self.recipes[doc_id]['title']).astype(str)
        df['ingredients'] = df['doc_id'].apply(lambda doc_id: self.recipes[doc_id]['ingredients'])
        df['description'] = df['doc_id'].apply(
            lambda doc_id: self.recipes[doc_id]['story_or_description'])
        df['description'].fillna('', inplace=True)
        df['author'] = df['doc_id'].apply(lambda doc_id: self.recipes[doc_id]['user_id'])
        df['country'] = df['doc_id'].apply(lambda doc_id: self.recipes[doc_id]['country'])
        df['label'] = df['label'].astype(float)

    def fit(self, df: pd.DataFrame):
        self.process_df(df)

        sentences = set()
        sentences |= set(df['query'])
        sentences |= set(df['title'])
        sentences |= {ingredient for ingredients in df['ingredients'].tolist() for ingredient in ingredients}
        sentences |= set(df['description'])
        self.tokenizer = Tokenizer(
            oov_token='<OOV>',
            char_level=False,
            num_words=self.num_words,
        )
        self.tokenizer.fit_on_texts(sentences)
        del sentences

        self.encoder['author'] = LabelEncoder()
        self.encoder['author'].fit(list(set(df['author']) | {''}))

        self.encoder['country'] = LabelEncoder()
        self.encoder['country'].fit(list(set(df['country']) | {''}))

    def process_batch(self, df: pd.DataFrame):
        df = df.copy()
        self.process_df(df)

        df['query_word_ids'] = self.tokenizer.texts_to_sequences(df['query'].tolist())
        df['title_word_ids'] = self.tokenizer.texts_to_sequences(df['title'].tolist())
        df['ingredients_word_ids'] = [self.tokenizer.texts_to_sequences(ingredients) for ingredients in
                                      df['ingredients']]
        df['description_word_ids'] = self.tokenizer.texts_to_sequences(df['description'].tolist())

        df['author'] = df['author'].apply(lambda c: c if c in self.encoder['author'].classes_ else '')
        df['author'] = self.encoder['author'].transform(df['author'])

        df['country'] = df['country'].apply(lambda c: c if c in self.encoder['country'].classes_ else '')
        df['country'] = self.encoder['country'].transform(df['country'])

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
        author = df['author'].to_numpy()
        country = df['country'].to_numpy()
        label = df['label'].to_numpy()

        return {
                   'query_word_ids': query_word_ids,
                   'title_word_ids': title_word_ids,
                   'ingredients_word_ids': ingredients_word_ids,
                   'description_word_ids': description_word_ids,
                   'author': author,
                   'country': country
               }, label
