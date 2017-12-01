# -*-coding:utf-8-*-
# reference:
# http://www.jianshu.com/p/e2f62043d02b
from __future__ import print_function
import tensorflow as tf
import numpy as np
import pandas as pd

n_classes = 50
n_nodes_input = 100
x = tf.placeholder(tf.float32, [None, n_nodes_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)
batch_size = 100

file_name_list = ["/Users/wangguang/PycharmProjects/my_experiment/data/r_1_train_data.csv",
                  "/Users/wangguang/PycharmProjects/my_experiment/data/r_2_train_data.csv",
                  "/Users/wangguang/PycharmProjects/my_experiment/data/r_3_train_data.csv",
                  "/Users/wangguang/PycharmProjects/my_experiment/data/r_4_train_data.csv",
                  "/Users/wangguang/PycharmProjects/my_experiment/data/r_5_train_data.csv"
                  ]


def read_from_csv_list():
    pd_ll = []

    for local_file in file_name_list:
        pd_ll.append(pd.read_csv(local_file, header=0, index_col=0))

    df = pd.concat(pd_ll)
    df = df.sample(frac=1, axis=0) # shuffle
    train_X = np.array(df.iloc[: 4000, : -n_classes].values, dtype=np.float)
    train_Y = np.array(df.iloc[: 4000, -n_classes:].values, dtype=np.int)
    # train_Y = train_Y.reshape(train_Y.shape[0], 1)

    test_X = np.array(df.iloc[4000:, : -n_classes].values, dtype=np.float)
    test_Y = np.array(df.iloc[4000:, -n_classes:].values, dtype=np.int)
    # test_Y = test_Y.reshape(test_Y.shape[0], 50)

    return train_X, train_Y, test_X, test_Y


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.random_normal(shape))


def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolution_network_model(xs):

    x_image = tf.reshape(xs, [-1, 10, 10, 1])

    W_conv1 = weight_variable([5, 5, 1, 32]) # patch 5x5, in size 1, out size 32
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([3*3*64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 3*3*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # what is dropout???

    W_fc2 = weight_variable([1024, n_classes])
    b_fc2 = bias_variable([n_classes])
    prediction = tf.matmul(h_fc1_drop, W_fc2)+b_fc2

    return prediction


def train_cnn(xs):
    prediction = convolution_network_model(xs)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 100
    # saver = tf.train.Saver()
    train_X, train_Y, test_X, test_Y = read_from_csv_list()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_X):
                start = i
                end = i + batch_size
                batch_x = train_X[start: end]
                batch_y = train_Y[start: end]
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
                epoch_loss += c
                i += batch_size
            print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss', epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: test_X, y: test_Y, keep_prob: 0.5}))
        # saver.save(sess, "./logs/1/model.cpkt")

if __name__ == '__main__':
    train_cnn(x)
