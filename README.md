# Study on Feature Interactions in Multiple Field Document Ranking

In this thesis, I discuss document ranking models for modern search applications from the relationship between the ranking task and the recommendation task. I hypothesized that the multi-field document ranking task is located somewhere between single-field document ranking and recommendation. If this hypothesis is correct, we can employ techniques to capture text matching signals from information retrieval and techniques to combine multiple evidence from recommender systems. The key contributions of this project are as follows:

- I reviewed the entire history of information retrieval from an academic and industrial perspective, and discussed the similarity between document ranking and item recommendation.
- My experiments has shown the possibility that recommendation models work for document ranking as well. It has also shown that incorporating the characteristics of document ranking into recommendation models further improves performance.
- I have published the implementation on GitHub for the validity of my experiments.

The full version is located here: https://github.com/rejasupotaro/master-thesis/blob/master/thesis.pdf

## Training and Evaluation

Experiments are conducted using a proprietary dataset obtained from [Cookpad](https://cookpad.com). Since my research is aimed at practical search systems where documents consist of various types of fields, I had to train models using search logs from a real application.

### Local

```
$ poetry run invoke generate-listwise-cookpad
```

### Inside Container 

```
$ sudo docker build -t master-thesis .
$ sudo docker run --mount src="$(pwd)/data",dst=/workspace/data,type=bind -it master-thesis /bin/bash
```

### AI Platform

```
ENV = 'cloud'
BUCKET_NAME = os.getenv('BUCKET_NAME')
REGION = os.getenv('REGION')
DATASET = 'cookpad'
MODEL_NAME = 'fwfm'
JOB_NAME = f'trainer_{DATASET}_{MODEL_NAME}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
JOB_DIR = f'gs://{BUCKET_NAME}/job'
DATASET_ID = '0-9'
EPOCHS = 8
BATCH_SIZE = 2048

!gcloud ai-platform jobs submit training $JOB_NAME \
  --package-path ../src \
  --module-name src.task \
  --region $REGION \
  --python-version 3.7 \
  --runtime-version 2.1 \
  --job-dir $JOB_DIR \
  --config ../ai_platform_config.yml \
  -- \
  --bucket-name $BUCKET_NAME \
  --env $ENV \
  --dataset $DATASET \
  --dataset-id $DATASET_ID \
  --model-name $MODEL_NAME \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE```
```
```
!gcloud beta ai-platform jobs stream-logs $JOB_NAME --polling-interval=180
```
