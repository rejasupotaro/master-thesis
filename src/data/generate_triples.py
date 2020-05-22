import os
from pathlib import Path
from collections import defaultdict
import random
import gzip

import numpy as np
import pandas as pd

from src.utils.logger import create_logger, get_logger

random.seed(42)
np.random.seed(42)

project_dir = Path(__file__).resolve().parents[2]
data_raw_dir = os.path.join(project_dir, 'data', 'raw')
data_processed_dir = os.path.join(project_dir, 'data', 'processed')


def load_queries(nrows):
    queries_df = pd.read_csv(
        os.path.join(data_raw_dir, 'msmarco-doctrain-queries.tsv.gz'),
        delimiter='\t',
        names=['qid', 'query'],
        nrows=nrows)
    queries = {}
    for index, row in queries_df.iterrows():
        queries[row['qid']] = row['query']
    return queries


def load_qrel(nrows):
    qrels_df = pd.read_csv(
        os.path.join(data_raw_dir, 'msmarco-doctrain-qrels.tsv.gz'),
        delimiter=' ',
        names=['qid', '_', 'docid', 'rel'],
        nrows=nrows)
    qrels = defaultdict(list)
    for index, row in qrels_df.iterrows():
        qrels[row['qid']].append(row['docid'])
    return qrels


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
    qrels = load_qrel(nrows)
    get_logger().info('Load docs...')
    docs = load_docs(nrows)

    get_logger().info('Generate triples...')
    stats = defaultdict(int)
    with gzip.open(os.path.join(data_raw_dir, 'msmarco-doctrain-top100.gz'), 'rt', encoding='utf8') as top100, \
            open(os.path.join(data_processed_dir, 'triples.tsv'), 'w', encoding='utf8') as out:
        for line in top100:
            qid, _, unjudged_docid, rank, _, _ = line.split()
            qid = int(qid)

            if qid not in queries or qid not in qrels:
                stats['skipped (query not found)'] += 1
                continue

            positive_docid = random.choice(qrels[qid])
            if positive_docid not in docs or unjudged_docid not in docs:
                stats['skipped (doc not found)'] += 1
                continue

            if unjudged_docid in qrels[qid]:
                stats['docid collisions'] += 1
                continue

            stats['kept'] += 1

            out.write(f'{qid}\t{queries[qid]}\t{positive_docid}\t{unjudged_docid}\n')

            triples_to_generate -= 1
            if triples_to_generate <= 0:
                break

    get_logger().info(dict(stats))
    get_logger().info('Done')


if __name__ == '__main__':
    create_logger()
    generate_triples(1000, nrows=None)
