from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import pandas as pd
import numpy as np
import traceback
import scipy
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from model import config
import matplotlib

# https://stackoverflow.com/questions/37604289/
# tkinter-tclerror-no-display-name-and-no-display-environment-variable
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


class DataGenerator:
    def __init__(self, con):
        self.file_name_list = con.file_name_list
        self.n_classes = con.n_classes
        self.test_data_ratio = con.test_data_ratio
        self.valid_X, self.valid_Y, self.train_X, \
        self.train_Y, self.test_X, self.test_Y = self.read_from_csv_list()


        self.total_data_num = self.valid_X.shape[0]
        self.total_train = self.train_X.shape[0]
        self.node_num = con.node_num
        self.class_number_list = con.class_number_list
        self.problem_type = con.problem_type_list[0]

        print('=> data init finished')

    def read_from_csv_list(self):
        """ read data from a csv file
        :return: 
        """
        pd_ll = []
        for local_file in self.file_name_list:
            pd_ll.append(pd.read_csv(local_file, header=0, index_col=0))

        # note: the data maybe already sampled because of the preprocess step
        df = pd.concat(pd_ll)

        # sift data
        # df = df.loc[(df.r == 2) & (df.s == 6), :]
        # print(df.head())

        valid_X = df.iloc[:, : -self.n_classes].reset_index(drop=True)
        valid_Y = df.iloc[:, -self.n_classes:].reset_index(drop=True)
        train_X, test_X, train_Y, test_Y = train_test_split(valid_X, valid_Y, test_size=self.test_data_ratio)
        # train_X = df.iloc[:, : -self.n_classes].reset_index(drop=True)
        # train_Y = df.iloc[:, -self.n_classes:].reset_index(drop=True)
        # test_X = df.iloc[-4096:, : -self.n_classes].reset_index(drop=True)
        # test_Y = df.iloc[-4096:, -self.n_classes:].reset_index(drop=True)

        print('=> train data size:', train_X.shape[0])
        print('=> test data size:', test_X.shape[0])
        print(train_X.head())
        print(train_Y.head())
        return valid_X, valid_Y, train_X, train_Y, test_X, test_Y

    def get_train_data(self, batch_size):
        """ get the sampled train data
        :param batch_size: 
        :return:  
        """
        for i in range(0, self.total_train, batch_size):
            yield self.train_X.iloc[i: i + batch_size, :], self.train_Y.iloc[i: i + batch_size, :]

    def get_origin_data(self, batch_size):
        """ get the whole origin data
        :param batch_size: 
        :return: 
        """
        for i in range(0, self.total_data_num, batch_size):
            yield self.valid_X.iloc[i: i + batch_size, :], self.valid_Y.iloc[i: i + batch_size, :]

    def get_test_data(self):
        """ get the test data
        :return: 
        """
        return self.test_X, self.test_Y

    @staticmethod
    def get_visualization(H, shape=None):
        """ visualize a matrix in heat map
        :param H: 
        :param shape: 
        :return: 
        """
        if shape:
            H = H.reshape(shape)
        # H = np.flipud(H)  # flip matrix up 2 down because plt.imshow will flip origin matrix
        sns.heatmap(H, fmt='d', linewidths=0.05)
        # plt.imshow(H, cmap=plt.cm.jet, interpolation='nearest', origin='low')
        plt.show()

    def get_eigenvectors(self):
        """ calc the eignvectors of a matrix 
        :return: 
        """
        for x, y in self.get_origin_data(1):
            m = x.values.reshape((self.node_num, self.node_num))
            # print(m+m.T)
            n = m + m.T
            # n = np.dot(m, m.T) + np.dot(m.T, m)
            e, v = scipy.linalg.eigh(n)  # np.linalg.eig will return the complex data sometimes...
            yield v.astype(np.float32), m, y

    def get_classify_confusion_matrix(self, predict):
        """ 
        :param predict: 
        :return: 
        """
        upper_bound = self.node_num // 2 if self.node_num % 2 == 0 else self.node_num // 2 + 1
        labels = [str((k, j)) for k in range(1, upper_bound + 1) for j in range(1, self.node_num + 1)]
        values = np.zeros((self.node_num * upper_bound, self.node_num * upper_bound)).astype(np.uint32)
        df = pd.DataFrame(values, columns=labels, index=labels[::-1])

        for i in range(len(predict[0])):
            x = np.argmax(predict[0][i]) + 1
            y = np.argmax(predict[1][i]) + 1

            r = np.argmax(self.test_Y.values[i][:self.class_number_list[0]]) + 1
            s = np.argmax(self.test_Y.values[i][self.class_number_list[0]:]) + 1

            idx_predict = str('({}, {})'.format(x, y))
            idx_real = str('({}, {})'.format(r, s))
            df[idx_predict][idx_real] += 1

        f, ax = plt.subplots(figsize=(15, 15))
        # https://seaborn.pydata.org/generated/seaborn.heatmap.html
        sns.heatmap(df, cmap="YlGnBu", annot=True, fmt='d', ax=ax)
        plt.savefig('./assets/confusion_matrix')
        plt.close()

    def get_regression_confusion_matrix(self, predict):
        """ confusion matrix for regression problem
        :param predict: 
        :return: 
        """
        upper_bound = self.node_num // 2 if self.node_num % 2 == 0 else self.node_num // 2 + 1
        labels = [str((k, j)) for k in range(1, upper_bound + 1) for j in range(1, self.node_num + 1)]
        values = np.zeros((self.node_num * upper_bound, self.node_num * upper_bound)).astype(np.uint32)
        df = pd.DataFrame(values, columns=labels, index=labels[::-1])

        for i in range(len(predict[0])):
            x = predict[0][i]
            y = predict[1][i]

            r = int(self.test_Y.values[i][0])
            s = int(self.test_Y.values[i][1])

            x = int(x) + 1 if x - float(int(x)) > 0.5 else int(x)
            y = int(y) + 1 if y - float(int(y)) > 0.5 else int(y)

            if x > upper_bound or y > self.node_num or x <= 0 or y <= 0:
                continue

            idx_predict = str('({}, {})'.format(x, y))
            idx_real = str('({}, {})'.format(r, s))
            df[idx_predict][idx_real] += 1

        f, ax = plt.subplots(figsize=(15, 15))
        sns.heatmap(df, cmap="YlGnBu", annot=True, fmt='d', ax=ax)
        plt.savefig('./assets/confusion_matrix')
        plt.close()

    def get_confusion_matrix(self, predict, problem_type):
        if problem_type:
            self.get_regression_confusion_matrix(predict)
        else:
            self.get_classify_confusion_matrix(predict)


def get_degree_info_from_matix(adjmatrix):
    """ statistical information of a matrix
    :param adjmatrix: 
    :return: 
    """
    in_degree_list = []
    node_num = adjmatrix.shape[0]

    for i in range(node_num):
        count = 0
        for j in range(node_num):
            count += adjmatrix[j][i]
        in_degree_list.append(count)

    min_in_degree = node_num
    min_count = 0
    max_in_degree = -1
    max_count = 0

    my_data = {
        'min_in_degree': [],
        'min_num': [],
        'max_in_degree': [],
        'max_num': [],
        'mean_in_degree': [],
        'var_in_degree': []

    }

    in_degree_sum = 0
    for v in in_degree_list:
        if v > max_in_degree:
            max_in_degree = v
            max_count = 1
        elif v == max_in_degree:
            max_count += 1

        if v < min_in_degree:
            min_in_degree = v
            min_count = 1
        elif v == min_in_degree:
            min_count += 1

        in_degree_sum += v

    my_data['min_in_degree'].append(min_in_degree)
    my_data['min_num'].append(min_count)
    my_data['max_in_degree'].append(max_in_degree)
    my_data['max_num'].append(max_count)

    in_degree_mean = in_degree_sum * 1.0 / node_num
    in_degree_var = 0
    for v in in_degree_list:
        in_degree_var += (v - in_degree_mean) ** 2
    in_degree_var = in_degree_var * 1.0 / node_num

    my_data['mean_in_degree'].append(in_degree_mean)
    my_data['var_in_degree'].append(in_degree_var)

    df = pd.DataFrame(my_data)
    # print(df.head())
    return df


def get_degree_info(path, file_name, node_num, class_num=2):
    """ statistical information of all matrices
    :param path: 
    :param file_name: 
        adjacent matrix file with labels in it
    :param node_num: 
    :param class_num: 
    :return: 
    """
    df = pd.read_csv(path + file_name, header=0, index_col=0).reset_index(drop=True)
    new_df = None
    for i in range(df.shape[0]):
        if new_df is None:
            new_df = get_degree_info_from_matix(df.iloc[i, :-class_num].values.reshape(node_num, node_num))
        else:
            new_df = pd.concat(
                [new_df, get_degree_info_from_matix(df.iloc[i, :-class_num].values.reshape(node_num, node_num))])

    print(new_df.sample(100))
    new_df.info()

    return new_df


def get_cluster_info(dir_path, file_name, node_num, K):
    """ count the number of the eigenvectors that assign to different classes
    :param dir_path: 
    :param file_name: 
        the file include the data after clustering
    :param node_num: 
    :param K: cluster number 
    :return: 
    """
    df = pd.read_csv(dir_path + file_name, header=0, index_col=0).reset_index(drop=True)

    # cluster may not the same as node num, because of data balanced operation
    columns = ['cluster_' + str(i) for i in range(K)]

    new_df = pd.DataFrame(np.zeros((df.shape[0] // node_num, node_num)), dtype=np.int, columns=columns)

    index = 0
    for i in range(0, df.shape[0], node_num):
        for j in range(node_num):
            new_df.iloc[index, df.iloc[i, -1]] += 1
        index += 1

    print(new_df.sample(100))
    # print(new_df.tail(10))

    return new_df


def get_eigen_vectors(dir_path, file_name, node_num):
    """ get eigenvectors from different matrix 
    :param dir_path: 
    :param file_name: 
        the origin adjmatrix file with labels in it
    :param node_num: 
    :return: 
    """

    # read the datagenerator class
    con = config.Config()
    con.file_name_list = [dir_path + file_name]
    con.node_num = node_num
    data = DataGenerator(con)

    df = None
    columns = None
    count = 0

    # v : the matrix of eigenvectors
    # x: adjacent matrix
    # y: labels
    for v, x, y in data.get_eigenvectors():
        #  the column of v corresponding to the eigenvector
        tmp = np.hstack([v.T, np.vstack([y.values] * v.shape[0])])
        if df is None:
            # v.shape[0] is the number of eigenvectors
            columns = [str(i) for i in range(v.shape[0])]
            columns.extend(['r', 's'])
            df = pd.DataFrame(tmp, columns=columns)

        else:
            df = pd.concat([df, pd.DataFrame(tmp, columns=columns)], axis=0)

        count += 1

        if count % 1000 == 0:
            print(df.head(10))
    print('=> eigenvector data:')
    print(df.sample(100))
    df.to_csv(dir_path + "eigenvector_" + file_name)
    print('=> save finished:', dir_path + "eigenvector_" + file_name)


def learning_vector_quantization_cluster_for_eigenvectors(dir_path, file_name, node_num, yita):
    """ Learninig vector quantization cluster algorithm
    :param dir_path: 
    :param file_name: eigenvectors file include the (r, s) label in it
    :param node_num: 
    :param yita: learning rate
    :return: 
        K: the number of clusters
    """
    df = pd.read_csv(dir_path + file_name, header=0, index_col=0, dtype=np.float32).reset_index(
        drop=True)
    df = pd.concat([df, pd.DataFrame(np.zeros((df.shape[0], 1)), columns=['cluster_class'], dtype=np.int32)], axis=1)
    df.r = df.r.astype(np.int32)
    df.s = df.s.astype(np.int32)
    print(df.head())

    import time
    start = time.time()

    # select centroid from different groups of s randomly
    centroid = []
    old_centroid = []
    for i in range(node_num):
        if df.loc[df.s == i + 1, :].shape[0] != 0:
            centroid.append(df.loc[df.s == i + 1, :].sample(1).values.astype(np.float64).reshape(1, -1))
    # number of clusters
    K = len(centroid)
    # print initialized centroid
    print('=> origin centroid')
    print(centroid)

    count = 0
    residual = -1
    while residual > 10 or residual < 0:

        for i in range(len(centroid)):
            old_centroid.append(centroid[i])

        for i in range(df.shape[0]):
            v1 = df.iloc[i, :-3].values.astype(np.float64).reshape(1, -1)

            distance = None

            for j in range(len(centroid)):
                if distance is None:
                    # float overflow if calculating by self methods like np.subtract
                    distance = np.linalg.norm(np.subtract(v1, centroid[j][:, :-3])).copy()
                else:
                    distance = np.hstack(
                        (distance, np.linalg.norm(np.subtract(v1, centroid[j][:, :-3]))))

            # print(distance)
            # print(np.argmin(distance))

            # get the minimum distance which is Euclid distance
            idx = np.argmin(distance)
            # print(idx)
            # print(distance)

            # http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
            df.loc[i, 'cluster_class'] = idx

            # idx maybe the list type
            if df['s'][i] == np.int32(centroid[idx][:, -2]):
                # p = p+(x-p)*yita
                centroid[idx] = np.hstack(
                    (np.add(centroid[idx][:, :-3], np.subtract(v1, centroid[idx][:, :-3], ) * yita),
                     centroid[idx][:, -3:]))
            else:
                # p = p-(x-p)*yita
                centroid[idx] = np.hstack((np.subtract(centroid[idx][:, :-3],
                                                       np.subtract(v1, centroid[idx][:, :-3]) * yita),
                                           centroid[idx][:, -3:]))
                # print(centroid)
                # time.sleep(3)

        # calculate residual in old and new centroid
        residual = 0.0
        for i in range(len(centroid)):
            residual += np.linalg.norm(centroid[i] - old_centroid[i], 1)
        residual = residual / len(centroid)

        print('=> old centroid')
        print(old_centroid)
        print('=> new centroid')
        print(centroid)
        print('=> iterate time {}: '.format(count + 1), time.time() - start)
        print('=> residual in old and new: ', residual)

        del old_centroid
        old_centroid = []

        print(df.head(100))
        print()
        print(df.sample(100))
        print('-' * 75)
        print()

        count += 1
    centroid_df = pd.DataFrame(np.vstack(centroid), columns=df.columns)
    print(centroid_df.head())

    centroid_df.to_csv(dir_path + 'centroid_' + file_name)
    df.to_csv(dir_path + 'lvq_cluster_' + file_name)

    print('=> finished time:', time.time() - start)

    return K


def k_means_cluster_for_eigenvectors(dir_path, file_name, K, class_number):
    """
    :param dir_path: 
    :param file_name: eigenvectors file include the (r, s) label in it
    :param K: number of clusters
    :param class_number: number of class label 
    :return: 
        K, number of clusters
    """
    import time
    start_time = time.time()
    eigenvectors_data = pd.read_csv(dir_path + file_name, header=0, index_col=0).reset_index(drop=True)
    print('=> origin eigenvectors data:')
    print(eigenvectors_data.head(100))

    # http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    kmeans = KMeans(n_clusters=K).fit(eigenvectors_data.iloc[:, :-class_number].values)
    class_labels = pd.DataFrame({'class_labels': kmeans.labels_}).reset_index(drop=True)

    print('=> cluster labels:')
    print(class_labels.head(10))
    # vertical concate it on the
    eigenvectors_data_with_class_labels = pd.concat([eigenvectors_data, class_labels], axis=1)
    print('=> k_means cluster data:')
    print(eigenvectors_data_with_class_labels.sample(100))
    eigenvectors_data_with_class_labels.to_csv(dir_path + 'k_means_cluster_' + file_name)

    print('=> save finished:', dir_path + 'k_means_cluster_' + file_name)

    print('=> finished time', time.time() - start_time)

    return K


def get_shift_mat(m, new_label):
    """ Elementary change of matrix
    :param m: 
        adjacent matrix
    :param new_label: 
        clusters list return from k-means cluster algo 
    :return: 
    """
    # keep different clusters
    row_list = []
    # keep the index of different cluster
    record_dict = {}
    # get the different cluster number
    ss = list(set(new_label))
    # keep the order of cluster
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


def highlight_out_edge(m, new_label, theta=5):
    """ highlight outlier 
    the outliers are the edges outside cluster 
    :param m: 
        the adjacent matrix
    :param new_label: 
        get from the k-means cluster algo
    :param theta:
        the number to be augmented
    :return: 
    """
    # note: the new_label is a reference like the &,
    # will change the origin new_label
    new_label.sort()
    for i in range(m.shape[0]):
        for j in range(m.shape[0]):
            if m[i][j] != 0 and new_label[i] != new_label[j]:
                m[i][j] = m[i][j] * theta

    return m


def get_cluser_matrix_data():
    """ cluster the adjacent matrix 
    3 different k clusters of co-cluster matrix as 3 channels of a image 
    :return: 
    """
    import time
    con = config.Config()
    digits = DataGenerator(con)

    columns = [str(_) for _ in range(3 * (digits.node_num ** 2))]
    columns.extend(['r', 's'])
    df = pd.DataFrame(columns=columns)
    start = time.time()
    for data, m, y in digits.get_eigenvectors():
        data = preprocessing.scale(data)
        # print(data)
        kmeans = KMeans(n_clusters=4, random_state=0).fit(data.T)
        new_label = kmeans.labels_
        # print(new_label)
        H1 = get_shift_mat(get_shift_mat(m, new_label).T, new_label).T
        highlight_out_edge(H1, new_label)

        kmeans = KMeans(n_clusters=3, random_state=0).fit(data.T)
        new_label = kmeans.labels_
        # print(new_label)
        H2 = get_shift_mat(get_shift_mat(m, new_label).T, new_label).T
        highlight_out_edge(H2, new_label)

        kmeans = KMeans(n_clusters=5, random_state=0).fit(data.T)
        new_label = kmeans.labels_
        # print(new_label)
        H3 = get_shift_mat(get_shift_mat(m, new_label).T, new_label).T
        highlight_out_edge(H3, new_label)

        H = np.hstack((H1.reshape(1, -1), H2.reshape(1, -1), H3.reshape(1, -1)))
        # print(H.shape)
        df = df.append(pd.DataFrame(np.hstack((H, y.values)), columns=columns))
        # print(df.head())
    df.to_csv('./data/directed/node_7/c_453_h_r_7_modified.csv')
    print(df.head())
    df.info()
    print('finished, time used:', time.time() - start)


def get_image_data():
    """ graph to 2-d histogram
    :return: 
    """
    con = config.Config()
    data = DataGenerator(con)
    xedges = [_ / 7 for _ in range(-14, 15)]
    yedges = [_ / 7 for _ in range(-14, 15)]

    columns = [str(_) for _ in range(28 * 28 * 5)]
    columns.extend(['r', 's'])
    df = pd.DataFrame()
    try:
        for x, y in data.get_train_data(1):
            if y.iloc[0, 1] not in [2, 3, 4, 6]:
                continue
            e, v = scipy.linalg.eigh(
                x.values.reshape((10, 10)))  # np.linalg.eig will return the complex data sometimes...
            image_data = {}
            for i in range(len(v)):
                new_v = preprocessing.scale(v[i])

                for k in range(0, len(new_v), 2):
                    if k not in image_data:
                        image_data[k] = {}
                        image_data[k][0] = [new_v[k]]
                        image_data[k][1] = [new_v[k + 1]]
                    else:
                        image_data[k][0].append(new_v[k])
                        image_data[k][1].append(new_v[k + 1])

            total_H = None
            for k in image_data.keys():
                H, new_xedges, new_yedges = np.histogram2d(image_data[k][0], image_data[k][1], bins=(xedges, yedges))
                if total_H is None:
                    total_H = H.reshape((-1, 28 * 28))
                else:
                    total_H = np.hstack((total_H, H.reshape((-1, 28 * 28))))

            total_H = np.hstack((total_H, y.values))
            df = df.append(pd.DataFrame(total_H, columns=columns))
            print(df.shape)
            # plt.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
            # plt.show()
    except Exception:
        traceback.print_exc()
    finally:
        print(df.head())
        df.to_csv('./data/undirected/node_10/image_with_r_s_balance.csv')


def test_image():
    """ graph to 2d histogram
    :return: 
    """
    import matplotlib.pyplot as plt

    con = config.Config()
    data = DataGenerator(con)
    xedges = [_ / 4.6 for _ in range(-14, 15)]
    yedges = [_ / 4.6 for _ in range(-14, 15)]
    image_data = {}
    for x, y in data.get_train_data(1):

        e, v = scipy.linalg.eigh(
            x.values.reshape((10, 10)))  # np.linalg.eig will return the complex data sometimes...

        for i in range(1, len(v)):
            new_v = preprocessing.scale(v[i])

            for k in range(0, len(new_v), 2):
                if k not in image_data:
                    image_data[k] = {}
                    image_data[k][0] = [new_v[k]]
                    image_data[k][1] = [new_v[k + 1]]
                else:
                    image_data[k][0].append(new_v[k])
                    image_data[k][1].append(new_v[k + 1])

    for k in image_data.keys():
        H, new_xedges, new_yedges = np.histogram2d(image_data[k][0], image_data[k][1], bins=(xedges, yedges))
        print(H.shape)
        plt.imshow(H.T, cmap=plt.cm.jet, interpolation='nearest', origin='low',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        plt.show()


def pipeline_all_preprocess_data(dir_path, origin_file_name, cluster_file_name, node_num, class_number,
                                 keep_statistical_features=False):
    """ make the final features
    :param dir_path: 
    :param cluster_file_name: 
    :param origin_file_name:
    :param node_num:
    :param class_number:
    :param keep_statistical_features:
        whether save the statistical features
    :return: 
    """
    import time

    start_time = time.time()

    cluster_data = pd.read_csv(dir_path + cluster_file_name, header=0, index_col=0).reset_index(drop=True)
    print('=> cluster data:')
    print(cluster_data.head(10))
    origin_data = pd.read_csv(dir_path + origin_file_name, header=0, index_col=0).reset_index(drop=True)
    print('=> origin data')
    print(origin_data.head(10))

    labels = origin_data.iloc[:, -class_number:]

    statistical_info_data = get_degree_info(dir_path, origin_file_name, node_num).reset_index(drop=True)
    if keep_statistical_features:
        statistical_features = pd.concat([statistical_info_data, labels], axis=1)
        print('=> statistical_features')
        print(statistical_features.sample(10))
        statistical_features.to_csv(dir_path + 'statistical_features_' + origin_file_name)
        print('=> save finished:', dir_path + 'statistical_features_' + origin_file_name)

    # K is the number of clusters
    cluster_data = get_cluster_info(dir_path, cluster_file_name, node_num, K=node_num).reset_index(drop=True)

    final_features = pd.concat([statistical_info_data, cluster_data, labels], axis=1)
    print('=> final features: ')
    final_features.info()
    print(final_features.describe())
    print(final_features.head(100))
    print(final_features.tail(100))

    final_features.to_csv(dir_path + 'features_' + cluster_file_name)
    print('=> save finished:', dir_path + 'features_' + cluster_file_name)

    print('=> finished time', time.time() - start_time)


if __name__ == '__main__':
    # con = config.Config()
    # data = DataGenerator(con)
    # k_means_cluster_for_eigenvectors('./data/non-isomorphism/convert_data/', 'eignvector_r_8.csv', 8, 2)
    pipeline_all_preprocess_data('./data/non-isomorphism/convert_data/', 'node_num_8.csv',
                                 "k_means_cluster_eignvector_r_8.csv", 8, 2, keep_statistical_features=True)
