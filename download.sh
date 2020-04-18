#!/bin/bash
# See: https://github.com/microsoft/TREC-2019-Deep-Learning

# msmarco-docs.tsv
curl -OL https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz
mv msmarco-docs.tsv.gz data/raw/

# msmarco-doctrain-queries.tsv
curl -OL https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-queries.tsv.gz
mv msmarco-doctrain-queries.tsv data/raw/

# msmarco-doctrain-top100.gz
curl -OL https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-top100.gz
mv msmarco-doctrain-top100.gz data/raw/

# msmarco-doctrain-qrels.tsv.gz
curl -OL https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-qrels.tsv.gz
mv msmarco-doctrain-qrels.tsv.gz data/raw/
