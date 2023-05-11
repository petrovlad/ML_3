import os
import cv2
import random
import logging
import datetime
import pandas
import tensorflow as tf
from matplotlib import pyplot as plt

USE_LARGE_DATASET = False
if USE_LARGE_DATASET:
    DATASET_DIR = r'./dataset/notMNIST_large/'
    TRAIN_SIZE = 100000
    VALIDATION_SIZE = 10000
    TEST_SIZE = 10000
else:
    DATASET_DIR = r'./dataset/notMNIST_small/'
    TRAIN_SIZE = 13000
    VALIDATION_SIZE = 3000
    TEST_SIZE = 3000

LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
LABELS_COUNT = len(LABELS)

DATA_COLUMN_NAME = 'data'
LABELS_COLUMN_NAME = 'labels'
HASHED_DATA_COLUMN_NAME = 'data_bytes'

BALANCE_PERCENT_BORDER = 0.85

BATCH_SIZE = 128
INITIAL_LEARNING_RATE = 0.01
MIN_LEARNING_RATE = 1e-6
DECAY_STEPS = 12000
DECAY_RATE = 0.8
EPOCHS = 50
EPOCHS_RANGE = range(EPOCHS)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def output_random_images():
    for label in LABELS:
        image_folder = os.path.join(DATASET_DIR, label)
        chosen_one = random.choice(os.listdir(image_folder))
        image_path = os.path.join(image_folder, chosen_one)
        img = cv2.imread(image_path)
        plt.imshow(img)
        plt.show()


def get_class_data(folder_path):
    result_data = list()
    files = os.listdir(folder_path)
    for file in files:
        image_path = os.path.join(folder_path, file)
        img = cv2.imread(image_path)
        if img is not None:
            result_data.append(img)

    return result_data


def create_data_frame():
    data = list()
    labels = list()
    for label in LABELS:
        class_dirpath = os.path.join(DATASET_DIR, label)
        class_data = get_class_data(class_dirpath)

        data.extend(class_data)
        labels.extend([LABELS.index(label) for _ in range(len(class_data))])

    data_frame = pandas.DataFrame({'data': data, 'label': labels})
    logging.info("Data frame is created")

    # removing dups
    data_bytes = [item.tobytes() for item in data_frame['data']]
    data_frame['hash'] = data_bytes
    data_frame.sort_values('hash', inplace=True)
    cnt_before = len(data_frame)
    data_frame.drop_duplicates(subset='hash', keep='first', inplace=True)
    cnt_after = len(data_frame)
    data_frame.pop('hash')
    logging.info(f"{cnt_before - cnt_after} duplicates removed")

    return data_frame


def verify_balance(data_frame):
    classes_images_counts = list()
    for class_index in range(LABELS_COUNT):
        labels = data_frame[LABELS_COLUMN_NAME]
        class_rows = data_frame[labels == class_index]
        class_count = len(class_rows)

        classes_images_counts.append(class_count)
        logging.info(f"Class {LABELS[class_index]} contains {class_count} images")

    max_images_count = max(classes_images_counts)
    avg_images_count = sum(classes_images_counts) / len(classes_images_counts)
    balance_percent = avg_images_count / max_images_count

    plt.figure()
    plt.bar(LABELS, classes_images_counts)
    plt.show()
    logging.info("Histogram shown")
    logging.info(f"Balance: {balance_percent:.3f}")

    if balance_percent > BALANCE_PERCENT_BORDER:
        logging.info("Classes are balanced")
    else:
        logging.info("Classes are not balanced")


def shuffle_data(data):
    # todo: do i need random state?
    return data.sample(frac=1, random_state=1337)


def split_dataset(data_frame):
    data = list(data_frame[DATA_COLUMN_NAME].values)
    labels = list(data_frame[LABELS_COLUMN_NAME].values)

    data_dataset = tf.data.Dataset.from_tensor_slices(data)
    labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((data_dataset, labels_dataset))

    train_dataset = dataset.take(TRAIN_SIZE).batch(BATCH_SIZE)
    validation_dataset = dataset.skip(TRAIN_SIZE).take(VALIDATION_SIZE).batch(BATCH_SIZE)
    test_dataset = dataset.skip(TRAIN_SIZE + VALIDATION_SIZE).take(TEST_SIZE).batch(BATCH_SIZE)
    logging.info("Data split")

    return train_dataset, validation_dataset, test_dataset


def get_stats(model, train_dataset, validation_dataset, test_dataset, with_optimization=False):
    train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # todo test sigmoid & relu and user the best option
    convolutional_model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.Conv2D(32, (3, 3), activation='sigmoid', input_shape=(28, 28, 1), kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(64, (3, 3), activation='sigmoid', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(LABELS_COUNT, activation='softmax')
    ])
    pooling_model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.Conv2D(32, (3, 3), activation='sigmoid', input_shape=(28, 28, 1), kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='sigmoid', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(LABELS_COUNT, activation='softmax')
    ])
    lenet_model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.Conv2D(6, (5, 5), activation='sigmoid', input_shape=(28, 28, 1), kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(16, (5, 5), activation='sigmoid', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Dense(84, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Dense(LABELS_COUNT, activation='softmax')
    ])
    # todo stats & analyze


def get_single_stats(train_dataset, validation_dataset, test_dataset):
    # todo implement network
    pass


def show_result_plot(losses, accuracies, validation_losses, validation_accuracies):
    # todo plots
    pass


def main():
    start_time = datetime.datetime.now()

    output_random_images()

    data_frame = create_data_frame()
    verify_balance(data_frame)
    data_frame = shuffle_data(data_frame)

    train_dataset, validation_dataset, test_dataset = split_dataset(data_frame)

    losses, accuracies, validation_losses, validation_accuracies = get_single_stats(
        train_dataset, validation_dataset, test_dataset
    )
    show_result_plot(losses, accuracies, validation_losses, validation_accuracies)

    end_time = datetime.datetime.now()
    logging.info(end_time - start_time)


if __name__ == '__main__':
    main()
