import datetime
import pickle
from pathlib import Path
from typing import Dict

import tensorflow as tf
from loguru import logger
from tensorflow import keras

from src.config.base_configs import TrainConfig
from src.data.data_generator import DataGenerator
from src.losses import pairwise_losses

project_dir = Path(__file__).resolve().parents[1]


# https://www.tensorflow.org/api_docs/python/tf/keras/backend/set_floatx
# tf.keras.mixed_precision.experimental.set_policy('mixed_float16')


def train_ranking_model(config: TrainConfig, batch_size: int) -> Dict:
    """
    Returns JSON consisting of key: metric and value: list of values
    e.g. {'loss': [1], 'accuracy': [1], 'val_loss': [1], 'val_accuracy': [1]}
    """
    logger.info('Transform examples into dataset')
    data_processor = config.data_processor

    train_df = data_processor.listwise_to_pairs(f'{config.dataset}.train.pkl')
    val_df = data_processor.listwise_to_pairs(f'{config.dataset}.val.pkl')
    data_processor.fit(train_df)
    with open(f'{project_dir}/models/{config.data_processor_filename}.pkl', 'wb') as file:
        pickle.dump(data_processor, file)
    train_generator = DataGenerator(train_df, data_processor, batch_size=batch_size)
    val_generator = DataGenerator(val_df, data_processor, batch_size=batch_size)

    logger.info('Build model')
    model = config.model(data_processor).build()

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss={'label': pairwise_losses.cross_entropy_loss},
        metrics=['accuracy']
    )

    logger.info('Train model')
    log_dir = f'{project_dir}/logs/fit/{model.name}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    callbacks = [
        # tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    ]
    history = model.fit(
        train_generator,
        epochs=config.epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=config.verbose,
    )

    logger.info('Save model')
    # model.save(f'{project_dir}/models/{model.name}.h5')

    return model, history.history
