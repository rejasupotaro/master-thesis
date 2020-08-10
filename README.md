# Study on Feature Interactions in Multiple Field Document Ranking (WIP)

This is a draft version of my thesis. It is planned to be completed by mid-September.

## Abstract

"Is a semi-structured document re-ranking task closer to a recommendation task rather than traditional document ranking?" I asked myself this question when I was working on a search system. Fortunately, I got an opportunity to examine it. In this thesis, I discuss how to design ranking models for modern search applications. I answer the following research questions through experiments.

- Is limiting feature interactions effective?
  - The literature of information retrieval says that it is important to capture query-field interactions, whereas recommendation models do not usually distinguish item features and context features.
  - They have the same goal: there are semi-structured complex items, given a context, find the optimal permutation. What makes them distinct?
- How important are text matching signals in document ranking?
  - If text matching signals are important, how critical it is?
  - There must be other important interactions besides query-field interactions. Are text matching signals still more important than those?
- Essentially, what learning to rank models learn?

My experiments show that ... (WIP)

The full version is located here: https://github.com/rejasupotaro/master-thesis/blob/master/thesis.pdf

## Training and Evaluation

Experiments are conducted using a proprietary dataset obtained from [Cookpad](https://cookpad.com). Since my research is aimed at practical search systems where documents consist of various types of fields, I had to train models using search logs from a real application.

### Local

```
$ poetry run python src/data/generate_listwise.py
$ poetry run python src/models/train_model.py
```

### Inside Container 

```
$ sudo docker build -t master-thesis .
$ sudo docker run --mount src="$(pwd)/data",dst=/workspace/data,type=bind -it master-thesis /bin/bash
```

### AI Platform

```
BUCKET_NAME = os.getenv('BUCKET_NAME')
REGION = os.getenv('REGION')
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
JOB_NAME = f'trainer_{timestamp}'
JOB_DIR = f'gs://{BUCKET_NAME}/job'
MODEL_NAME = 'naive'
DATASET_SIZE = 'small'
EPOCHS = 10

!gcloud ai-platform jobs submit training $JOB_NAME \
  --package-path ../src \
  --module-name src.task \
  --region $REGION \
  --python-version 3.7 \
  --runtime-version 2.1 \
  --job-dir $JOB_DIR \
  --stream-logs \
  --config ../ai_platform_config.yml \
  -- \
  --bucket-name $BUCKET_NAME \
  --model-name $MODEL_NAME \
  --dataset-size $DATASET_SIZE \
  --epochs $EPOCHS
```
