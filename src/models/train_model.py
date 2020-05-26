from tensorflow import keras

from src.data import triples_to_dataset
from src.utils.logger import create_logger, get_logger
from src.utils.seed import set_seed
from src.models import simple_model


def train(triples_filename, epochs):
    create_logger()
    set_seed()

    get_logger().info('Convert triples to dataset')
    train_dataset, test_dataset, tokenizer = triples_to_dataset.process(triples_filename)
    total_words = len(tokenizer.word_index) + 1

    get_logger().info('Build model')
    model = simple_model.build_model(total_words)
    model.summary()
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss={'relevance': keras.losses.BinaryCrossentropy(from_logits=True)},
        metrics=['accuracy']
    )

    get_logger().info('Train model')
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=test_dataset
    )
    get_logger().info('Done')


if __name__ == '__main__':
    train('triples_100_100.pkl', 1)
