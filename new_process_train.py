import gc
import os
import cv2
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python import SKCompat
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib


tf.logging.set_verbosity(tf.logging.INFO)


def train(features, labels, mode):
    # Input Layer
    input_layer = tf.reshape(features, [-1, 56, 56, 3])
    # Convolutional Layer 1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=64,
        kernel_size=[3, 3],
        strides=1,
        padding="same",
        activation=tf.nn.relu)
    # Convolutional Layer 2
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=64,
        kernel_size=[3, 3],
        strides=1,
        padding="same",
        activation=tf.nn.relu)
    # Pooling Layer 1
    pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    # Dropout Layer 1
    dropout1 = tf.layers.dropout(inputs=pool1, rate=0.4, training=mode == learn.ModeKeys.TRAIN)
    # Convolutional Layer 3
    conv3 = tf.layers.conv2d(
        inputs=dropout1,
        filters=512,
        kernel_size=[3, 3],
        strides=1,
        padding="same",
        activation=tf.nn.relu)
    # Convolutional Layer 4
    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=512,
        kernel_size=[3, 3],
        strides=1,
        padding="same",
        activation=tf.nn.relu)
    # Convolutional Layer 5
    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=512,
        kernel_size=[3, 3],
        strides=1,
        padding="same",
        activation=tf.nn.relu)
    # Pooling Layer 4
    pool2 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)
    # Dropout Layer 2
    dropout2 = tf.layers.dropout(inputs=pool2, rate=0.4, training=mode == learn.ModeKeys.TRAIN)
    # Convolutional Layer 6
    conv6 = tf.layers.conv2d(
        inputs=dropout2,
        filters=512,
        kernel_size=[3, 3],
        strides=1,
        padding="same",
        activation=tf.nn.relu)
    # Convolutional Layer 7
    conv7 = tf.layers.conv2d(
        inputs=conv6,
        filters=512,
        kernel_size=[3, 3],
        strides=1,
        padding="same",
        activation=tf.nn.relu)
    # Convolutional Layer 8
    conv8 = tf.layers.conv2d(
        inputs=conv7,
        filters=512,
        kernel_size=[3, 3],
        strides=1,
        padding="same",
        activation=tf.nn.relu)
    # Pooling Layer 5
    pool3 = tf.layers.max_pooling2d(inputs=conv8, pool_size=[2, 2], strides=2)
    # Dropout Layer 3
    dropout3 = tf.layers.dropout(inputs=pool3, rate=0.4, training=mode == learn.ModeKeys.TRAIN)
    # Flatten the layer
    pool3_flat = tf.reshape(dropout3, [-1, 7 * 7 * 512])
    # Dense Layer 1
    dense1 = tf.layers.dense(inputs=pool3_flat, units=4096, activation=tf.nn.relu)
    # Dense Layer 2
    dense2 = tf.layers.dense(inputs=dense1, units=1000, activation=tf.nn.relu)
    # Dropout Layer 4
    dropout4 = tf.layers.dropout(inputs=dense2, rate=0.4, training=mode == learn.ModeKeys.TRAIN)
    # Logits Layer
    logits = tf.layers.dense(inputs=dropout4, units=17)
    loss = None
    train_op = None
    # Calculate loss for both Train and Eval
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=17)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    # Configure Training
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.001,
            optimizer='Adam')
    # Generate Predictions
    predictions = {
        "classes": tf.argmax(
            input=logits, axis=1),
        "probabilities": tf.nn.softmax(
            logits, name="softmax_tensor")
    }
    # Return the object
    return model_fn_lib.ModelFnOps(mode=mode, predictions=predictions, loss=loss, train_op=train_op)


def main():
    localtime = time.asctime(time.localtime(time.time()))
    print(localtime)
    label_dict = {}
    file_train_data = []
    file_train_label = []
    train_data = []
    train_label = []
    filecount = 0
    labels = {'agriculture': 0, 'artisinal_mine': 1, 'bare_ground': 2, 'blooming': 3, 'blow_down': 4, 'clear': 5,
              'cloudy': 6, 'conventional_mine': 7, 'cultivation': 8, 'habitation': 9, 'haze': 10, 'partly_cloudy': 11,
              'primary': 12, 'road': 13, 'selective_logging': 14, 'slash_burn': 15, 'water': 16}
    df = pd.read_csv('train_v2.csv', header=None, dtype=object)
    for x in df.as_matrix()[1:]:
        label_lst = []
        for y in str(x[1:][0]).split(' '):
            label_lst.append(labels[y])
        label_dict[x[0]] = label_lst
    explore_path = "/Users/praneet/Documents/Kaggle/Amazon/train"
    for root, dirs, files in os.walk(explore_path):
        for file_name in files:
            if file_name.endswith(".jpg"):
                filecount += 1
                print(file_name)
                file_path = os.path.abspath(os.path.join(root, file_name))
                img = cv2.imread(file_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                fixed_size = (56, 56)
                img = cv2.resize(img, dsize=fixed_size)
                file_title = file_name.split('.')[0]
                file_train_data.append(img)
                file_train_label.append(label_dict[file_title])
    print(filecount)
    for x in range(len(file_train_label)):
        for y in range(len(file_train_label[x])):
            train_data.append(file_train_data[x])
            train_label.append(file_train_label[x][y])
    file_train_data = None
    file_train_label = None
    del file_train_data
    del file_train_label
    gc.collect()
    print(len(train_data))
    print(len(train_label))
    train_data = np.array(train_data, dtype=np.float32)/255.
    train_label = np.array(train_label, dtype=np.int32)
    # Create the Estimator
    classifier = SKCompat(learn.Estimator(model_fn=train, model_dir="/Users/praneet/Downloads/model"))
    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
    # Train the model
    classifier.fit(
        x=train_data,
        y=train_label,
        batch_size=100,
        steps=500,
        monitors=[logging_hook])
    localtime = time.asctime(time.localtime(time.time()))
    print(localtime)


if __name__ == '__main__':
    main()
