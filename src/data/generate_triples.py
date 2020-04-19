import os
from pathlib import Path
from collections import defaultdict
import random

import numpy as np
import pandas as pd

from src.utils.logger import create_logger, get_logger

random.seed(42)
np.random.seed(42)

project_dir = Path(__file__).resolve().parents[2]
data_raw_dir = os.path.join(project_dir, 'data', 'raw')
data_processed_dir = os.path.join(project_dir, 'data', 'processed')


def load_queries(nrows):
    doctrain_queries_df = pd.read_csv(
        os.path.join(data_raw_dir, 'msmarco-doctrain-queries.tsv.gz'),
        delimiter='\t',
        names=['qid', 'query'],
        nrows=nrows)
    queries = {}
    for index, row in doctrain_queries_df.iterrows():
        queries[row['qid']] = row['query']
    return queries


def load_qrel(nrows):
    doctrain_qrels_df = pd.read_csv(
        os.path.join(data_raw_dir, 'msmarco-doctrain-qrels.tsv.gz'),
        delimiter=' ',
        names=['qid', '_', 'docid', 'rel'],
        nrows=nrows)
    qrel = defaultdict(list)
    for index, row in doctrain_qrels_df.iterrows():
        qrel[row['qid']].append(row['docid'])
    return qrel


def load_top100_df(nrows):
    doctrain_top100_df = pd.read_csv(
        os.path.join(data_raw_dir, 'msmarco-doctrain-top100.gz'),
        delimiter=' ',
        names=['qid', 'Q0', 'docid', 'rank', 'score', 'runstring'],
        nrows=nrows)
    return doctrain_top100_df


def load_docs(nrows):
    docs_df = pd.read_csv(
        os.path.join(data_raw_dir, 'msmarco-docs.tsv.gz'),
        delimiter='\t',
        header=None,
        names=['docid', 'url', 'title', 'body'],
        nrows=nrows)
    docs = defaultdict(list)
    for index, row in docs_df.iterrows():
        docid, url, title, body = row
        docs[docid].append({
            'docid': docid,
            'url': url,
            'title': title,
            'body': body
        })
    return docs


def generate_triples(triples_to_generate, nrows=10000):
    get_logger().info('Load queries...')
    queries = load_queries(nrows)
    get_logger().info('Load qrel...')
    qrel = load_qrel(nrows)
    get_logger().info('Load top100...')
    top100_df = load_top100_df(nrows)
    get_logger().info('Load docs...')
    docs = load_docs(nrows)

    get_logger().info('Generate triples...')
    stats = defaultdict(int)
    triples = []
    for index, row in top100_df.sample(frac=1).iterrows():
        qid, _, unjudged_docid, rank, _, _ = row

        if qid not in queries or qid not in qrel:
            stats['skipped (query not found)'] += 1
            continue

        positive_docid = random.choice(qrel[qid])
        if positive_docid not in docs or unjudged_docid not in docs:
            stats['skipped (doc not found)'] += 1
            continue

        if unjudged_docid in qrel[qid]:
            stats['docid collisions'] += 1
            continue

        stats['kept'] += 1

        triple = {
            'qid': qid,
            'query': queries[qid],
            'positive_docid': positive_docid,
            'negative_docid': unjudged_docid}
        triples.append(triple)

        triples_to_generate -= 1
        if triples_to_generate <= 0:
            break

    get_logger().info(dict(stats))
    get_logger().info(f'{len(triples)} triples were generated')

    triples_df = pd.DataFrame(triples)
    triples_df.to_csv(os.path.join(data_processed_dir, 'triples.csv'), index=False)
    get_logger().info('Done')


if __name__ == '__main__':
    create_logger()
    generate_triples(100000, nrows=None)
