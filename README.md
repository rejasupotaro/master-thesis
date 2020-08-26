# Study on Feature Interactions in Multiple Field Document Ranking (WIP)

This is a draft version of my thesis. It is planned to be completed by mid-September.

## Abstract

Single-field document ranking has long been a central topic in information retrieval, and many ranking models have been proposed in the history. However, little is known about multi-field document ranking. In this thesis, I discuss how to design ranking models for modern search applications.
I hypothesized that semi-structured document ranking is located somewhere between single-field document ranking and recommendation. Therefore, I address the following research questions by comparing models in information retrieval and recommendation systems.

- **Is learning interaction important?** Interaction-based models are said to be better than representation-based models. To further discuss interaction strengths between query and fields, I assess the claim in a reliable way.
- **Is limiting feature interactions effective?** Capturing query-field interactions is said to be important in information retrieval, whereas recommendation models do not usually distinguish item features and context features. They have the same goal: there are semi-structured complex items, given a context, find the optimal permutation. What makes them distinct?
- **Does feeding first-order features contribute to effectiveness?** Feeding first-order features is said to increase the risk of overfitting, whereas recommendation models do not usually care about it.
- **How important are text matching signals in document re-ranking?** In re-ranking tasks, candidates are already filtered based on lexical features. Could considering lexical features in the re-ranking stage be redundant? There must be other important interactions besides query-field interactions.

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
