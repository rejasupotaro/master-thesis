import abc
import pickle
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm

from src.data.cookpad.recipes import load_recipes, load_raw_recipes

project_dir = Path(__file__).resolve().parents[3]


class DataProcessor(abc.ABC):
    def __init__(self, docs: Dict = None, dataset_size: str = None, num_words: int = 200000,
                 max_negatives: int = 10):
        if not docs:
            if dataset_size:
                self.docs = load_recipes(dataset_size)
            else:
                self.docs = load_raw_recipes()
        else:
            self.docs = docs
        self.num_words: int = num_words
        self.tokenizer: Optional[Tokenizer] = None
        self.encoder: Dict[str, LabelEncoder] = {}
        self.max_negatives: int = max_negatives

    @property
    def doc_id_encoder(self) -> LabelEncoder:
        return self.encoder['doc_id']

    @property
    def total_words(self) -> int:
        return len(self.tokenizer.word_index) + 1

    @property
    def total_authors(self) -> int:
        return len(self.encoder['author'].classes_)

    @property
    def total_countries(self) -> int:
        return len(self.encoder['country'].classes_)

    def listwise_to_pairs(self, listwise_filename: str) -> DataFrame:
        with open(f'{project_dir}/data/processed/{listwise_filename}', 'rb') as file:
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
        return DataFrame(rows)

    def listwise_to_random_pairs(self, listwise_filename: str) -> DataFrame:
        project_dir = Path(__file__).resolve().parents[2]
        with open(f'{project_dir}/data/processed/{listwise_filename}', 'rb') as file:
            dataset = pickle.load(file)

        positives = []
        negatives = []
        for example in tqdm(dataset):
            query = example['query']
            for doc in example['docs']:
                data = {
                    'query': query,
                    'doc_id': doc['doc_id'],
                    'label': doc['label']
                }
                if doc['label'] == 1:
                    positives.append(data)
                negatives.append(data)

        rows = []
        import random

        for positive in tqdm(positives):
            for _ in range(self.max_negatives):
                rows.append(positive)
                rows.append(random.choice(negatives))
        return DataFrame(rows)

    @abc.abstractmethod
    def process_df(self, df: DataFrame) -> None:
        raise NotImplementedError('Calling an abstract method.')

    @abc.abstractmethod
    def fit(self, df: DataFrame) -> None:
        raise NotImplementedError('Calling an abstract method.')

    @abc.abstractmethod
    def process_batch(self, df: DataFrame) -> Tuple[Dict, List[int]]:
        raise NotImplementedError('Calling an abstract method.')


class ConcatDataProcessor(DataProcessor):
    def process_df(self, df: DataFrame) -> None:
        df['doc_id'] = df['doc_id'].astype(np.int64)
        df['query'] = df['query'].astype(str)
        df['title'] = df['doc_id'].apply(lambda doc_id: self.docs[doc_id]['title']).astype(str)
        df['ingredients'] = df['doc_id'].apply(lambda doc_id: self.docs[doc_id]['ingredients'])
        df['ingredients'] = df['ingredients'].apply(lambda x: ' '.join(x))
        df['description'] = df['doc_id'].apply(
            lambda doc_id: self.docs[doc_id]['story_or_description'])
        df['description'].fillna('', inplace=True)
        df['author'] = df['doc_id'].apply(lambda doc_id: self.docs[doc_id]['user_id'])
        df['country'] = df['doc_id'].apply(lambda doc_id: self.docs[doc_id]['country'])
        df['label'] = df['label'].astype(float)

    def fit(self, df: DataFrame) -> None:
        self.process_df(df)

        self.encoder['doc_id'] = LabelEncoder()
        doc_ids = [doc_id for doc_id in self.docs] + [-1]
        self.encoder['doc_id'].fit(doc_ids)

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

    def process_batch(self, df: DataFrame) -> Tuple[Dict, List[int]]:
        df = df.copy()
        self.process_df(df)

        df['doc_id'] = df['doc_id'].apply(lambda c: c if c in self.encoder['doc_id'].classes_ else -1)
        df['doc_id'] = self.encoder['doc_id'].transform(df['doc_id'])
        doc_id = df['doc_id'].to_numpy()

        df['query'] = self.tokenizer.texts_to_sequences(df['query'].tolist())
        df['title'] = self.tokenizer.texts_to_sequences(df['title'].tolist())
        df['ingredients'] = self.tokenizer.texts_to_sequences(df['ingredients'].tolist())
        df['description'] = self.tokenizer.texts_to_sequences(df['description'].tolist())

        df['author'] = df['author'].apply(lambda c: c if c in self.encoder['author'].classes_ else '')
        df['author'] = self.encoder['author'].transform(df['author'])

        df['country'] = df['country'].apply(lambda c: c if c in self.encoder['country'].classes_ else '')
        df['country'] = self.encoder['country'].transform(df['country'])

        query = df['query'].tolist()
        query = pad_sequences(
            query,
            padding='post',
            truncating='post',
            maxlen=6
        )
        title = df['title'].tolist()
        title = pad_sequences(
            title,
            padding='post',
            truncating='post',
            maxlen=20
        )
        ingredients = df['ingredients'].tolist()
        ingredients = pad_sequences(
            ingredients,
            padding='post',
            truncating='post',
            maxlen=300
        )
        description = df['description'].tolist()
        description = pad_sequences(
            description,
            padding='post',
            truncating='post',
            maxlen=100
        )
        author = df['author'].to_numpy()
        country = df['country'].to_numpy()
        label = df['label'].to_numpy()

        return {
                   'doc_id': doc_id,
                   'query': query,
                   'title': title,
                   'ingredients': ingredients,
                   'description': description,
                   'author': author,
                   'country': country
               }, label


class MultiInstanceDataProcessor(DataProcessor):
    def process_df(self, df: DataFrame) -> None:
        df['doc_id'] = df['doc_id'].astype(np.int64)
        df['query'] = df['query'].astype(str)
        df['title'] = df['doc_id'].apply(lambda doc_id: self.docs[doc_id]['title']).astype(str)
        df['ingredients'] = df['doc_id'].apply(lambda doc_id: self.docs[doc_id]['ingredients'])
        df['description'] = df['doc_id'].apply(
            lambda doc_id: self.docs[doc_id]['story_or_description'])
        df['description'].fillna('', inplace=True)
        df['author'] = df['doc_id'].apply(lambda doc_id: self.docs[doc_id]['user_id'])
        df['country'] = df['doc_id'].apply(lambda doc_id: self.docs[doc_id]['country'])
        df['label'] = df['label'].astype(float)

    def fit(self, df: DataFrame) -> None:
        self.process_df(df)

        self.encoder['doc_id'] = LabelEncoder()
        doc_ids = [doc_id for doc_id in self.docs] + [-1]
        self.encoder['doc_id'].fit(doc_ids)

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

    def process_batch(self, df: DataFrame) -> Tuple[Dict, List[int]]:
        df = df.copy()
        self.process_df(df)

        df['doc_id'] = df['doc_id'].apply(lambda c: c if c in self.encoder['doc_id'].classes_ else -1)
        df['doc_id'] = self.encoder['doc_id'].transform(df['doc_id'])
        doc_id = df['doc_id'].to_numpy()

        df['query'] = self.tokenizer.texts_to_sequences(df['query'].tolist())
        df['title'] = self.tokenizer.texts_to_sequences(df['title'].tolist())
        df['ingredients'] = [self.tokenizer.texts_to_sequences(ingredients) for ingredients in
                             df['ingredients']]
        df['description'] = self.tokenizer.texts_to_sequences(df['description'].tolist())

        df['author'] = df['author'].apply(lambda c: c if c in self.encoder['author'].classes_ else '')
        df['author'] = self.encoder['author'].transform(df['author'])

        df['country'] = df['country'].apply(lambda c: c if c in self.encoder['country'].classes_ else '')
        df['country'] = self.encoder['country'].transform(df['country'])

        query = df['query'].tolist()
        query = pad_sequences(
            query,
            padding='post',
            truncating='post',
            maxlen=6
        )
        title = df['title'].tolist()
        title = pad_sequences(
            title,
            padding='post',
            truncating='post',
            maxlen=20
        )
        ingredients = df['ingredients'].tolist()
        ingredients = [
            pad_sequences(word_ids, padding='post', truncating='post', maxlen=20) for word_ids in ingredients
        ]
        ingredients = pad_sequences(ingredients, padding='post', truncating='post', maxlen=30)
        description = df['description'].tolist()
        description = pad_sequences(
            description,
            padding='post',
            truncating='post',
            maxlen=100
        )
        author = df['author'].to_numpy()
        country = df['country'].to_numpy()
        label = df['label'].to_numpy()

        return {
                   'doc_id': doc_id,
                   'query': query,
                   'title': title,
                   'ingredients': ingredients,
                   'description': description,
                   'author': author,
                   'country': country
               }, label
