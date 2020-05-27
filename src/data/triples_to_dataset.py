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


def process(triples_filename):
    project_dir = Path(__file__).resolve().parents[2]
    with open(os.path.join(project_dir, 'data', 'processed', triples_filename), 'rb') as file:
        triples = pickle.load(file)
    recipes = load_recipes()
    df = pd.DataFrame(triples)

    df['positive_title'] = df['positive_doc_id'].apply(lambda doc_id: recipes[doc_id]['title'])
    df['negative_title'] = df['negative_doc_id'].apply(lambda doc_id: recipes[doc_id]['title'])
    df['positive_desc'] = df['positive_doc_id'].apply(lambda doc_id: recipes[doc_id]['story_or_description'])
    df['positive_desc'] = df['positive_desc'].fillna('')
    df['negative_desc'] = df['negative_doc_id'].apply(lambda doc_id: recipes[doc_id]['story_or_description'])
    df['negative_desc'] = df['negative_desc'].fillna('')
    df['positive_country'] = df['positive_doc_id'].apply(lambda doc_id: recipes[doc_id]['country'])
    df['negative_country'] = df['negative_doc_id'].apply(lambda doc_id: recipes[doc_id]['country'])

    oov_token = '<OOV>'
    sentences = []
    for key in ['query', 'positive_title', 'negative_title', 'positive_desc', 'negative_desc']:
        sentences += df[key].tolist()
    tokenizer = Tokenizer(oov_token=oov_token)
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    total_words = len(tokenizer.word_index) + 1

    df['query_word_ids'] = tokenizer.texts_to_sequences(df['query'].tolist())
    df['positive_title_word_ids'] = tokenizer.texts_to_sequences(df['positive_title'].tolist())
    df['negative_title_word_ids'] = tokenizer.texts_to_sequences(df['negative_title'].tolist())
    df['positive_desc_word_ids'] = tokenizer.texts_to_sequences(df['positive_desc'].tolist())
    df['negative_desc_word_ids'] = tokenizer.texts_to_sequences(df['negative_desc'].tolist())

    country_encoder = LabelEncoder()
    countries = pd.concat([df['positive_country'], df['negative_country']], axis=0)
    country_encoder.fit(countries)
    df['positive_country'] = country_encoder.transform(df['positive_country'])
    df['negative_country'] = country_encoder.transform(df['negative_country'])

    query_word_ids = df['query_word_ids'].tolist()
    query_word_ids = pad_sequences(query_word_ids,
                                   padding='post',
                                   truncating='post',
                                   maxlen=6)

    positive_title_word_ids = df['positive_title_word_ids'].tolist()
    positive_title_word_ids = pad_sequences(positive_title_word_ids,
                                            padding='post',
                                            truncating='post',
                                            maxlen=20)
    positive_desc_word_ids = df['positive_desc_word_ids'].tolist()
    positive_desc_word_ids = pad_sequences(positive_desc_word_ids,
                                            padding='post',
                                            truncating='post',
                                            maxlen=500)
    positive_countries = df['positive_country'].tolist()

    negative_title_word_ids = df['negative_title_word_ids'].tolist()
    negative_title_word_ids = pad_sequences(negative_title_word_ids,
                                            padding='post',
                                            truncating='post',
                                            maxlen=20)
    negative_desc_word_ids = df['negative_desc_word_ids'].tolist()
    negative_desc_word_ids = pad_sequences(negative_desc_word_ids,
                                            padding='post',
                                            truncating='post',
                                            maxlen=500)
    negative_countries = df['negative_country'].tolist()

    positive_relevance = np.array([1] * len(query_word_ids)).reshape(-1, 1)
    negative_relevance = np.array([0] * len(query_word_ids)).reshape(-1, 1)

    query_word_ids = np.concatenate((query_word_ids, query_word_ids), axis=0)
    title_word_ids = np.concatenate((positive_title_word_ids, negative_title_word_ids), axis=0)
    desc_word_ids = np.concatenate((positive_desc_word_ids, negative_desc_word_ids), axis=0)
    countries = np.concatenate((positive_countries, negative_countries), axis=0)
    relevance = np.concatenate((positive_relevance, negative_relevance), axis=0)

    train_dataset = tf.data.Dataset.from_tensor_slices((
        {
            'query_word_ids': query_word_ids,
            'title_word_ids': title_word_ids,
            'desc_word_ids': desc_word_ids,
            'country': countries
        },
        {'relevance': relevance}
    )).batch(32).shuffle(buffer_size=1000)

    test_dataset = tf.data.Dataset.from_tensor_slices((
        {
            'query_word_ids': query_word_ids,
            'title_word_ids': title_word_ids,
            'desc_word_ids': desc_word_ids,
            'country': countries
        },
        {'relevance': relevance}
    )).batch(32)

    return train_dataset, test_dataset, tokenizer


if __name__ == '__main__':
    with open('../data/processed/triples_100_100.train.pkl', 'rb') as file:
        triples = pickle.load(file)
    process(triples)
