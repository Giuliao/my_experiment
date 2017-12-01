# -*-coding:utf-8-*-
# reference:
# [1] tensorboard,
# http://blog.csdn.net/darlingwood2013/article/details/68921800
# [2] tensorboard code,
# https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py
# [3] cnn,
# http://www.jianshu.com/p/e2f62043d02b

from __future__ import print_function
import tensorflow as tf
import pandas as pd
import numpy as np

path_to_log = './logs/tensorboard/'

file_name_list = ["/Users/wangguang/PycharmProjects/my_experiment/data/r_1_train_data_new.csv",
                  "/Users/wangguang/PycharmProjects/my_experiment/data/r_2_train_data_new.csv",
                  "/Users/wangguang/PycharmProjects/my_experiment/data/r_3_train_data_new.csv",
                  "/Users/wangguang/PycharmProjects/my_experiment/data/r_4_train_data_new.csv",
                  "/Users/wangguang/PycharmProjects/my_experiment/data/r_5_train_data_new.csv"]



# file_name_list = ["/Users/wangguang/PycharmProjects/my_experiment/data/r_1_train_data.csv",
#                   "/Users/wangguang/PycharmProjects/my_experiment/data/r_2_train_data.csv",
#                   "/Users/wangguang/PycharmProjects/my_experiment/data/r_3_train_data.csv",
#                   "/Users/wangguang/PycharmProjects/my_experiment/data/r_4_train_data.csv",
#                   "/Users/wangguang/PycharmProjects/my_experiment/data/r_5_train_data.csv"]

# file_name_list = ["/Users/wangguang/PycharmProjects/my_experiment/data/node_6/r_1_train_data.csv",
#                   "/Users/wangguang/PycharmProjects/my_experiment/data/node_6/r_2_train_data.csv",
#                   "/Users/wangguang/PycharmProjects/my_experiment/data/node_6/r_3_train_data.csv"]

# file_name_list = ["/Users/wangguang/PycharmProjects/my_experiment/data/node_6/r_1_train_data_new.csv",
#                   "/Users/wangguang/PycharmProjects/my_experiment/data/node_6/r_2_train_data_new.csv",
#                   "/Users/wangguang/PycharmProjects/my_experiment/data/node_6/r_3_train_data_new.csv"]

# file_name_list = ["./data/r_5_representaion.csv"]


n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
n_nodes_hl4 = 500
n_nodes_hl5 = 500
n_nodes_hl6 = 500
n_nodes_hl7 = 300
n_nodes_hl8 = 200
n_nodes_hl9 = 200
n_nodes_hl10 = 100
n_nodes_input = 100

n_classes = 5
batch_size = 80

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, n_nodes_input])
    y = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)


def read_from_csv_list():

    pd_ll = []
    upper_bound = 2000
    for local_file in file_name_list:
        pd_ll.append(pd.read_csv(local_file, header=0, index_col=0))

    df = pd.concat(pd_ll)
    df = df.sample(frac=1, axis=0) # shuffle
    train_X = np.array(df.iloc[: upper_bound, : -n_classes].values, dtype=np.float)
    train_Y = np.array(df.iloc[: upper_bound, -n_classes:].values, dtype=np.int)
    # train_Y = train_Y.reshape(train_Y.shape[0], 1)

    test_X = np.array(df.iloc[upper_bound:, : -n_classes].values, dtype=np.float)
    test_Y = np.array(df.iloc[upper_bound:, -n_classes:].values, dtype=np.int)
    # test_Y = test_Y.reshape(test_Y.shape[0], 50)

    return train_X, train_Y, test_X, test_Y


def weight_variable(shape):
    """create a weight variable with appropriate initialization..
    :param shape: 
    :return: 
    """
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    """create a bias variable with appropriate initialization.
    :param shape: 
    :return: 
    """
    return tf.Variable(tf.constant(0.1, shape=shape))


def variable_summaries(var):
    """
    :param var: 
    :return: 
    """
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def nn_layer(input_tensor, dim, layer_name, act=tf.nn.relu, cnn=False):
    """
    :param input_tensor: 
    :param input_dim: 
    :param output_dim: 
    :param layer_name: 
    :param act: 
    :return: 
    """
    with tf.name_scope(layer_name):
        with tf.name_scope('weight'):
            weights = weight_variable(dim)
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable(dim[-1:])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            if cnn:
                preactivate = conv2d(input_tensor, weights) + biases
            else:
                preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)

        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        return activations


def neural_network_model(data):

    l1 = nn_layer(data, [n_nodes_input, n_nodes_hl1], 'layer1')
    l2 = nn_layer(l1, [n_nodes_hl1, n_nodes_hl2], 'layer2')
    l3 = nn_layer(l2, [n_nodes_hl2, n_nodes_hl3], 'layer3')
    output = nn_layer(l3, [n_nodes_hl3, n_classes], 'outlayer', tf.identity)

    return output


def convolution_network_model(xs):

    x_image = tf.reshape(xs, [-1, 10, 10, 1])
    h_conv1 = nn_layer(x_image, [5, 5, 1, 32], 'conv1_layer', cnn=True)
    # h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = nn_layer(h_conv1, [3, 3, 32, 64], 'conv2_layer', cnn=True)
    # h_pool2 = max_pool_2x2(h_conv2)
    h_conv3 = nn_layer(h_conv2, [2, 2, 64, 128], 'conv3_layer', cnn=True)

    h_conv3_flat = tf.reshape(h_conv3, [-1, 3*3*128])
    h_fc1 = nn_layer(h_conv3_flat, [3*3*128, 1024], 'fully_connceted')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # what is dropout???

    prediction = nn_layer(h_fc1_drop, [1024, 1000], 'output_layer', act=tf.identity)

    return prediction


def train_neural_network(x):
    with tf.name_scope('Model'):
        # prediction = convolution_network_model(x)
        prediction = neural_network_model(x)

    with tf.name_scope('Loss'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    tf.summary.scalar("loss", cost)

    with tf.name_scope('Optimizer'):
        optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)
    # optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
    with tf.name_scope('Accuracy'):
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    tf.summary.scalar("acc", accuracy)

    merged_summary_op = tf.summary.merge_all()

    hm_epochs = 1000
    total_acc = 0
    saver = tf.train.Saver()
    train_X, train_Y, test_X, test_Y = read_from_csv_list()
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(path_to_log+'10_dnn_train', sess.graph)
        test_writer = tf.summary.FileWriter(path_to_log+'10_dnn_test')
        sess.run(tf.global_variables_initializer())
        j = 0
        for step in range(hm_epochs):
            epoch_loss = 0
            i = 0


            while i < len(train_X):
                start = i
                end = i + batch_size
                batch_x = train_X[start: end]
                batch_y = train_Y[start: end]
                _, c, summary = sess.run([optimizer, cost, merged_summary_op], feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
                j += 1
                epoch_loss += c
                i += batch_size
                train_writer.add_summary(summary, j)

            # acc2 = sess.run([accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
            # print 'train accuracy', acc2
            acc, summary, predic = sess.run([accuracy, merged_summary_op, prediction], feed_dict={x: test_X, y: test_Y, keep_prob: 0.5})
            #print(predic.shape, test_Y.shape)
            #print(np.argmax(predic, 1), '=>', np.argmax(test_Y, 1))
            total_acc += acc
            test_writer.add_summary(summary, step)
            #print('Epoch', step+1, 'completed out of', hm_epochs, 'loss', epoch_loss)
            #print('accuracy:', acc)
        train_writer.close()
        test_writer.close()
    print('average accuracy: %.5f' % (total_acc / hm_epochs))
        # print 'Accuracy:', accuracy.eval({x: test_X, y: test_Y})
    saver.save(sess, "./logs/1/model.cpkt")

if __name__ == '__main__':
   train_neural_network(x)

    # x, y, z, w = read_from_csv_list()
    # print(np.argmax(y, 1).shape)
    # for i in range(y.shape[0]):
    #     print(y[i])
    #     print(np.argmax(y))

