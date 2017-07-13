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
    input_layer = tf.reshape(features, [-1, 224, 224, 3])
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
    # Convolutional Layer 3
    conv3 = tf.layers.conv2d(
        inputs=pool1,
        filters=128,
        kernel_size=[3, 3],
        strides=1,
        padding="same",
        activation=tf.nn.relu)
    # Convolutional Layer 4
    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=128,
        kernel_size=[3, 3],
        strides=1,
        padding="same",
        activation=tf.nn.relu)
    # Pooling Layer 2
    pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
    # Convolutional Layer 5
    conv5 = tf.layers.conv2d(
        inputs=pool2,
        filters=256,
        kernel_size=[3, 3],
        strides=1,
        padding="same",
        activation=tf.nn.relu)
    # Convolutional Layer 6
    conv6 = tf.layers.conv2d(
        inputs=conv5,
        filters=256,
        kernel_size=[3, 3],
        strides=1,
        padding="same",
        activation=tf.nn.relu)
    # Convolutional Layer 7
    conv7 = tf.layers.conv2d(
        inputs=conv6,
        filters=256,
        kernel_size=[3, 3],
        strides=1,
        padding="same",
        activation=tf.nn.relu)
    # Pooling Layer 3
    pool3 = tf.layers.max_pooling2d(inputs=conv7, pool_size=[2, 2], strides=2)
    # Convolutional Layer 8
    conv8 = tf.layers.conv2d(
        inputs=pool3,
        filters=512,
        kernel_size=[3, 3],
        strides=1,
        padding="same",
        activation=tf.nn.relu)
    # Convolutional Layer 9
    conv9 = tf.layers.conv2d(
        inputs=conv8,
        filters=512,
        kernel_size=[3, 3],
        strides=1,
        padding="same",
        activation=tf.nn.relu)
    # Convolutional Layer 10
    conv10 = tf.layers.conv2d(
        inputs=conv9,
        filters=512,
        kernel_size=[3, 3],
        strides=1,
        padding="same",
        activation=tf.nn.relu)
    # Pooling Layer 4
    pool4 = tf.layers.max_pooling2d(inputs=conv10, pool_size=[2, 2], strides=2)
    # Convolutional Layer 11
    conv11 = tf.layers.conv2d(
        inputs=pool4,
        filters=512,
        kernel_size=[3, 3],
        strides=1,
        padding="same",
        activation=tf.nn.relu)
    # Convolutional Layer 12
    conv12 = tf.layers.conv2d(
        inputs=conv11,
        filters=512,
        kernel_size=[3, 3],
        strides=1,
        padding="same",
        activation=tf.nn.relu)
    # Convolutional Layer 13
    conv13 = tf.layers.conv2d(
        inputs=conv12,
        filters=512,
        kernel_size=[3, 3],
        strides=1,
        padding="same",
        activation=tf.nn.relu)
    # Pooling Layer 5
    pool5 = tf.layers.max_pooling2d(inputs=conv13, pool_size=[2, 2], strides=2)
    # Flatten the layer
    pool5_flat = tf.reshape(pool5, [-1, 7 * 7 * 512])
    # Dense Layer 1
    dense1 = tf.layers.dense(inputs=pool5_flat, units=4096, activation=tf.nn.relu)
    # Dense Layer 2
    dense2 = tf.layers.dense(inputs=dense1, units=4096, activation=tf.nn.relu)
    # Dense Layer 3
    dense3 = tf.layers.dense(inputs=dense2, units=1000, activation=tf.nn.relu)
    # Dropout
    dropout = tf.layers.dropout(inputs=dense3, rate=0.4, training=mode == learn.ModeKeys.TRAIN)
    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=17)
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
            learning_rate=0.01,
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
                fixed_size = (224, 224)
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
    # Alternate between one and two each run
    train_data_one = train_data[:len(train_data) // 2]
    # train_data_two = train_data[len(train_data) // 2:]
    train_data_one = np.array(train_data_one, dtype=np.float32)
    # train_data_two = np.array(train_data_two, dtype=np.float32)
    train_label_one = train_label[:len(train_label) // 2]
    # train_label_two = train_label[len(train_label) // 2:]
    train_label_one = np.array(train_label_one, dtype=np.int32)
    # train_label_two = np.array(train_label_two, dtype=np.int32)
    # Create the Estimator
    classifier = SKCompat(learn.Estimator(model_fn=train, model_dir="/Users/praneet/Downloads/model"))
    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
    # Train the model
    classifier.fit(
        x=train_data_one,
        y=train_label_one,
        batch_size=100,
        steps=4000,
        monitors=[logging_hook])
    localtime = time.asctime(time.localtime(time.time()))
    print(localtime)


if __name__ == '__main__':
    main()
