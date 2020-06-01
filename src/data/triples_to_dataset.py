import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from src.data.recipes import load_recipes


def process(triples_filename, tokenizer=None, country_encoder=None):
    project_dir = Path(__file__).resolve().parents[2]
    with open(os.path.join(project_dir, 'data', 'processed', triples_filename), 'rb') as file:
        triples = pickle.load(file)
    recipes = load_recipes()
    triples_df = pd.DataFrame(triples)

    rows = []
    for index, row in triples_df.sample(frac=1).iterrows():
        for sample in ['positive', 'negative']:
            rows.append({
                'query': row['query'],
                'doc_id': row[f'{sample}_doc_id'],
                'label': 1 if sample == 'positive' else 0
            })
    df = pd.DataFrame(rows)

    df['title'] = df['doc_id'].apply(lambda doc_id: recipes[doc_id]['title'])
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
            num_words=30,
            char_level=True
        )
        tokenizer.fit_on_texts(sentences)

    for feature in ['query', 'title', 'ingredients']:
        df[f'{feature}_word_ids'] = tokenizer.texts_to_sequences(df[feature].tolist())

    if not country_encoder:
        country_encoder = LabelEncoder()
        country_encoder.fit(df['country'])

    df['country'] = country_encoder.transform(df['country'])

    query_word_ids = df['query_word_ids'].tolist()
    query_word_ids = pad_sequences(query_word_ids,
                                   padding='post',
                                   truncating='post',
                                   maxlen=50)

    title_word_ids = df['title_word_ids'].tolist()
    title_word_ids = pad_sequences(title_word_ids,
                                   padding='post',
                                   truncating='post',
                                   maxlen=120)
    ingredients_word_ids = df['ingredients_word_ids'].tolist()
    ingredients_word_ids = pad_sequences(title_word_ids,
                                   padding='post',
                                   truncating='post',
                                   maxlen=1500)
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


if __name__ == '__main__':
    process('triples_100_100.train.pkl')
