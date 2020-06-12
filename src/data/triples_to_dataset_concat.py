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


def process(df, tokenizer=None, country_encoder=None):
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


def process_listwise(listwise_filename, tokenizer=None, country_encoder=None):
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
    return process(df, tokenizer, country_encoder)


def process_triples(triples_filename, tokenizer=None, country_encoder=None):
    project_dir = Path(__file__).resolve().parents[2]
    with open(os.path.join(project_dir, 'data', 'processed', triples_filename), 'rb') as file:
        triples = pickle.load(file)
    triples_df = pd.DataFrame(triples)

    rows = []
    for index, row in tqdm(triples_df.sample(frac=1).iterrows(), total=len(triples_df)):
        for sample in ['positive', 'negative']:
            rows.append({
                'query': row['query'],
                'doc_id': row[f'{sample}_doc_id'],
                'label': 1 if sample == 'positive' else 0
            })
    df = pd.DataFrame(rows)
    return process(df, tokenizer, country_encoder)


if __name__ == '__main__':
    process_triples('triples_100_100.train.pkl')
