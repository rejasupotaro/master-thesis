import json
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.utils.logger import create_logger, get_logger
from src.utils.seed import set_seed

project_dir = Path(__file__).resolve().parents[2]


def get_available_recipe_ids():
    recipe_ids = []
    with open(os.path.join(project_dir, 'data', 'raw', 'recipes.json')) as file:
        recipes = json.load(file)
        for recipe in recipes:
            recipe_ids.append(recipe['recipe_id'])
    return recipe_ids


def preprocess_query(query):
    return str(query).replace('"', '')


def generate_triples(n_queries=100, max_positives_per_query=100):
    get_logger().info('Load available recipe IDs')
    available_recipe_ids = get_available_recipe_ids()

    get_logger().info('Extract popular queries')
    interactions_df = pd.read_csv(os.path.join(project_dir, 'data', 'raw', 'interactions.csv'))
    interactions_df['processed_query'] = interactions_df['query'].apply(preprocess_query)
    queries_df = interactions_df[['processed_query']].groupby('processed_query').size().reset_index(name='count')
    queries_df = queries_df.sort_values('count', ascending=False)
    popular_queries = queries_df.head(n_queries)['processed_query'].tolist()

    get_logger().info('Generate qrels')
    df = interactions_df[['recipe_id', 'processed_query']]
    df = df[df['recipe_id'].isin(available_recipe_ids)]
    df = df[df['processed_query'].isin(popular_queries)]
    qrels = defaultdict(list)
    for index, row in tqdm(df.iterrows(), total=len(df)):
        query = row['processed_query']
        doc_id = row['recipe_id']
        qrels[query].append(doc_id)
    with open(os.path.join('data', 'processed', f'qrels_{n_queries}.pkl'), 'wb') as file:
        pickle.dump(qrels, file)

    get_logger().info('Generate triples')
    doc_ids = list({doc_id for sublist in list(qrels.values()) for doc_id in sublist})
    triples = []
    for query in tqdm(popular_queries):
        positive_doc_ids = qrels[query]
        random.shuffle(positive_doc_ids)
        for positive_doc_id in positive_doc_ids[:max_positives_per_query]:
            triple = {}
            triple['query'] = query
            triple['positive_doc_id'] = positive_doc_id
            while True:
                negative_doc_id = random.choice(doc_ids)
                if negative_doc_id not in positive_doc_ids:
                    triple['negative_doc_id'] = negative_doc_id
                    break
            triples.append(triple)
    get_logger().info(f'{len(triples)} triples generated')
    with open(os.path.join('data', 'processed', f'triples_{n_queries}_{max_positives_per_query}.pkl'),
              'wb') as file:
        pickle.dump(triples, file)
    get_logger().info('Done')


if __name__ == '__main__':
    create_logger()
    set_seed()
    generate_triples()
