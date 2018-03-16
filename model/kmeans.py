# -*-coding:utf-8-*-
# references
# [1] https://gist.github.com/narphorium/d06b7ed234287e319f18
# [2] http://scikit-learn.org/stable/auto_examples/cluster/
# plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from model import config
from data_processing import DataGenerator
import tensorflow as tf
from sklearn.cluster import KMeans
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale


def kMeansCluster(vector_values, num_clusters, max_num_steps, stop_coeficient=0.0):
    vectors = tf.constant(vector_values)
    centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors),
                                     [0, 0], [num_clusters, -1]))
    old_centroids = tf.Variable(tf.zeros([num_clusters, vector_values.shape[1]]))
    centroid_distance = tf.Variable(tf.zeros([num_clusters, vector_values.shape[1]]))

    expanded_vectors = tf.expand_dims(vectors, 0)
    expanded_centroids = tf.expand_dims(centroids, 1)

    print(expanded_vectors.get_shape())
    print(expanded_centroids.get_shape())

    distances = tf.reduce_sum(
        tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2)

    assignments = tf.argmin(distances, 0)

    means = tf.concat([
        tf.reduce_mean(
            # get specific number from the index, reduction_indices refer to the axes
            # gather function will increase the dimension of  vectors variable
            tf.gather(vectors,
                      # -1 means the program will calc the columns automatically, shape 2 row vec
                      tf.reshape(
                          # return the index that satisfied the condition
                          # return the tensor that have the same dimension as the origin one
                          tf.where(
                              tf.equal(assignments, c)
                          ), [1, -1])
                      ), reduction_indices=[1])  # why it was 1 ??? because it was 3 dimension not 2 !!!
        for c in xrange(num_clusters)],
        0)

    save_old_centroids = tf.assign(old_centroids, centroids)

    update_centroids = tf.assign(centroids, means)
    init_op = tf.global_variables_initializer()

    performance = tf.assign(centroid_distance, tf.subtract(centroids, old_centroids))
    check_stop = tf.reduce_sum(tf.abs(performance))

    centroid_values = None
    assignment_values = None

    with tf.Session() as sess:
        sess.run(init_op)
        for step in xrange(max_num_steps):
            print("Running step " + str(step))
            sess.run(save_old_centroids)
            _, centroid_values, assignment_values = sess.run([update_centroids,
                                                              centroids,
                                                              assignments])

            # sess.run(check_stop)
            current_stop_coeficient = check_stop.eval()
            print("coeficient:", current_stop_coeficient)
            if current_stop_coeficient <= stop_coeficient:
                break

        return centroid_values, assignment_values


def print_dict(local_dict):
    for k in local_dict.keys():
        if isinstance(local_dict[k], int):
            print('%d\t' % k, '|' * local_dict[k], local_dict[k])
        else:
            print('-' * 75)
            print('in class', k)
            print_dict(local_dict[k])


def get_shift_mat(m, new_label):

    row_list = []
    record_dict = {}
    ss = list(set(new_label))
    ss.sort()

    for s in ss:
        for i, l in enumerate(new_label):
            if s == l:
                if s not in record_dict:
                    record_dict[s] = len(row_list)
                    row_list.append(m[[i], :])
                else:
                    idx = record_dict[s]
                    row_list[idx] = np.vstack((row_list[idx], m[[i], :]))

    final_m = None
    for k in row_list:
        if final_m is None:
            final_m = k[:, :]
        else:
            final_m = np.vstack((final_m, k))

    return final_m


def highlight_out_edge(m, new_label):
    new_label.sort()
    for i in range(m.shape[0]):
        for j in range(m.shape[0]):
            if m[i][j] != 0 and new_label[i] != new_label[j]:
                m[i][j] = m[i][j]*5


if __name__ == '__main__':
    con = config.Config()
    digits = DataGenerator(con)

    for data, m, y in digits.get_eigenvectors():
        # data, m = digits.get_eigenvectors()
        # print(data)
        # data = scale(data, axis=1)
        # print(data)
        # print()

        # new_label = kMeansCluster(data.T, 2, 1000)[1]
        print('-'*75)
        print(m)
        print()
        digits.get_visualization(m)

        kmeans = KMeans(n_clusters=4, random_state=0).fit(data.T)
        new_label = kmeans.labels_
        print(new_label)
        H = get_shift_mat(get_shift_mat(m, new_label).T, new_label).T
        print(H)
        highlight_out_edge(H, new_label)
        digits.get_visualization(H)

        kmeans = KMeans(n_clusters=3, random_state=0).fit(data.T)
        new_label = kmeans.labels_
        H = get_shift_mat(get_shift_mat(m, new_label).T, new_label).T
        print(H)
        highlight_out_edge(H, new_label)
        digits.get_visualization(H)

        kmeans = KMeans(n_clusters=2, random_state=0).fit(data.T)
        new_label = kmeans.labels_
        H = get_shift_mat(get_shift_mat(m, new_label).T, new_label).T
        print(H)
        highlight_out_edge(H, new_label)
        digits.get_visualization(H)

        print(y.values)
        print('-'*75)

