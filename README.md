# Relevance Estimation for Multiple-Field Documents

A short description of the project.

## Setup

### Local

```
$ poetry run python src/data/generate_listwise.py
$ poetry run python src/models/train_model.py
```

### Inside Container 

```
$ sudo docker built -t master-thesis .
$ sudo docker run --mount src="$(pwd)/data",dst=/workspace/data,type=bind -it master-thesis /bin/bash
```

### AI Platform

```
BUCKET_NAME = os.getenv('BUCKET_NAME')
REGION = os.getenv('REGION')
JOB_NAME = f'trainer_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
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

