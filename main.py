import os
import cv2
import random
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

BATCH_SIZE = 128
DECAY_STEPS = 12000
DECAY_RATE = 0.8
EPOCHS = 50
EPOCHS_RANGE = range(EPOCHS)


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

    # removing dups
    data_bytes = [item.tobytes() for item in data_frame['data']]
    data_frame['hash'] = data_bytes
    data_frame.sort_values('hash', inplace=True)
    cnt_before = len(data_frame)
    data_frame.drop_duplicates(subset='hash', keep='first', inplace=True)
    cnt_after = len(data_frame)
    data_frame.pop('hash')
    print(f"{cnt_before - cnt_after} duplicates removed")

    return data_frame


def verify_balance(data_frame):
    classes_images_counts = list()
    for label in range(LABELS_COUNT):
        labels = data_frame['label']
        class_rows = data_frame[labels == label]
        class_count = len(class_rows)

        classes_images_counts.append(class_count)
        print(f"`{LABELS[label]}` size == {class_count}")

    max_images_count = max(classes_images_counts)
    avg_images_count = sum(classes_images_counts) / len(classes_images_counts)
    balance_percent = avg_images_count / max_images_count
    #plt.figure()
    #plt.bar(LABELS, classes_images_counts)
    #plt.show()
    if balance_percent > 0.85:
        print("Classes are balanced")
    else:
        print("Classes are not balanced")


def shuffle_dataframe(data):
    # todo: do i need random state?
    return data.sample(frac=1, random_state=1337)


def split_dataset(data_frame):
    data = list(data_frame['data'].values)
    labels = list(data_frame['label'].values)

    data_dataset = tf.data.Dataset.from_tensor_slices(data)
    labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((data_dataset, labels_dataset))

    train_dataset = dataset.take(TRAIN_SIZE).batch(BATCH_SIZE)
    validation_dataset = dataset.skip(TRAIN_SIZE).take(VALIDATION_SIZE).batch(BATCH_SIZE)
    test_dataset = dataset.skip(TRAIN_SIZE + VALIDATION_SIZE).take(TEST_SIZE).batch(BATCH_SIZE)
    return train_dataset, validation_dataset, test_dataset


def get_stats(train_dataset, validation_dataset, test_dataset):
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

    convolutional_model_stats = get_single_stats(convolutional_model, train_dataset, validation_dataset, test_dataset, 'convolutional')
    pooling_model_stats = get_single_stats(pooling_model, train_dataset, validation_dataset, test_dataset, 'pooling')
    lenet_model_stats = get_single_stats(lenet_model, train_dataset, validation_dataset, test_dataset, 'lenet')

    return convolutional_model_stats, pooling_model_stats, lenet_model_stats


def get_single_stats(model, train_dataset, validation_dataset, test_dataset, model_name):
    model.compile(
        optimizer=tf.keras.optimizers.experimental.SGD(learning_rate=0.01),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy'])

    model_history = model.fit(
        x=train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=None,
        verbose=1
    )

    loss, accuracy = model.evaluate(test_dataset)
    print(f"{model_name}: {accuracy=}, {loss=}")

    return (model_history.history['loss'],
            model_history.history['accuracy'],
            model_history.history['val_loss'],
            model_history.history['val_accuracy'])


def main():
    output_random_images()

    data_frame = create_data_frame()
    verify_balance(data_frame)
    data_frame = shuffle_dataframe(data_frame)

    train_dataset, validation_dataset, test_dataset = split_dataset(data_frame)

    c_stats, p_stats, l_stats = get_stats(train_dataset, validation_dataset, test_dataset)
    plt.figure(figsize=(20, 14))

    plt.subplot(1, 2, 1)
    plt.title('Training and Validation Loss')

    plt.plot(EPOCHS_RANGE, c_stats[0], label='Train Convolutional Loss')
    plt.plot(EPOCHS_RANGE, c_stats[2], label='Validation Convolutional Loss')

    plt.plot(EPOCHS_RANGE, p_stats[0], label='Train Pooling Loss')
    plt.plot(EPOCHS_RANGE, p_stats[2], label='Validation Pooling Loss')

    plt.plot(EPOCHS_RANGE, l_stats[0], label='Train Lenet Loss')
    plt.plot(EPOCHS_RANGE, l_stats[2], label='Validation Lenet Loss')

    plt.subplot(1, 2, 2)
    plt.title('Training and Validation Accuracy')

    plt.plot(EPOCHS_RANGE, c_stats[1], label='Train Convolutional Accuracy')
    plt.plot(EPOCHS_RANGE, c_stats[3], label='Validation Convolutional Accuracy', linestyle='dashed')

    plt.plot(EPOCHS_RANGE, p_stats[1], label='Train Pooling Accuracy')
    plt.plot(EPOCHS_RANGE, p_stats[3], label='Validation Pooling Accuracy', linestyle='dashed')

    plt.plot(EPOCHS_RANGE, l_stats[1], label='Train Lenet Accuracy')
    plt.plot(EPOCHS_RANGE, l_stats[3], label='Validation Lenet Accuracy', linestyle='dashed')

    plt.show()


if __name__ == '__main__':
    main()
