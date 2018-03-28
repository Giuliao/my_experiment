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
import consensus_algo
import matplotlib
import os

# https://stackoverflow.com/questions/37604289/
# tkinter-tclerror-no-display-name-and-no-display-environment-variable
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


class DataGenerator:
    def __init__(self, con):
        self.file_name_list = con.file_name_list
        self.dir_path = con.dir_path
        self.n_classes = con.n_classes
        self.test_data_ratio = con.test_data_ratio

        self.origin_df = None
        self.valid_X = None
        self.valid_Y = None
        self.train_X = None
        self.train_Y = None
        self.test_X = None
        self.test_Y = None
        self.read_from_csv_list()

        self.total_data_num = self.valid_X.shape[0]
        self.total_train = self.train_X.shape[0]
        self.node_num = con.node_num
        self.class_number_list = con.class_number_list
        self.problem_type = con.problem_type_list[0]

        print('=> data init finished')

    def read_from_csv_list(self):
        """ init data from a csv file
        :return: 
        """
        pd_ll = []
        for local_file in map(lambda x: self.dir_path+x, self.file_name_list):
            pd_ll.append(pd.read_csv(local_file, header=0, index_col=0))

        # note: the data maybe already sampled because of the preprocess step
        self.origin_df = pd.concat(pd_ll)

        # select data
        # df = df.loc[(df.r == 2) & (df.s == 6), :]
        # print(df.head())

        self.valid_X = self.origin_df.iloc[:, : -self.n_classes].reset_index(drop=True)
        self.valid_Y = self.origin_df.iloc[:, -self.n_classes:].reset_index(drop=True)
        self.train_X, self.test_X, self.train_Y, self.test_Y = train_test_split(
            self.valid_X, self.valid_Y, test_size=self.test_data_ratio
        )

        print('=> train data size:', self.train_X.shape[0])
        print('=> test data size:', self.test_X.shape[0])
        print(self.train_X.head())
        print(self.train_Y.head())

    def get_cross_valid_data_set(self, fold=10):
        """ cross valid
        
        :param fold: 
        :return: 
        """
        if not os.path.exists(self.dir_path+'cv/'):
            os.mkdir(self.dir_path+'cv/')

        df = self.origin_df.sample(frac=1).reset_index(drop=True)
        total_len = df.shape[0]
        stride = total_len - total_len // fold

        count = 0
        i = 0
        while True:
            print('-' * 75)
            if i + stride > total_len:
                t = pd.concat([df.iloc[i:, :], df.iloc[:stride - (total_len - i)]], axis=0)
                count += 1
                i = stride - (total_len - i)

            else:
                t = df.iloc[i: i + stride, :]
                count += 1
                i += stride

            print('=> count', count)
            print('=> size', t.shape[0])
            print(t.head())
            print(t.tail())
            t.to_csv(self.dir_path + 'cv/{}_fold_cv_data_{}'.format(fold, count))
            print('=>', self.dir_path + 'cv/{}_fold_cv_data_{}'.format(fold, count))

            if count == fold:
                break

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

    def get_eigenvectors(self):
        """ calc the eignvectors of a matrix 
        :return: 
        """
        for x, y in self.get_origin_data(1):
            m = x.values.reshape((self.node_num, self.node_num))
            data = consensus_algo.NetworkAlgo(adjMatrix=m, directed=True)
            e, v = data.get_eigen_vectors(sym_func=data.get_laplacian_matrix)
            # why the type must be np.float32, i forgot >.<
            yield v.astype(np.float32), m, y

    def get_classify_confusion_matrix(self, predict):
        """ 
        :param predict: 
        :return: 
        """
        upper_bound = self.node_num // 2 if self.node_num % 2 == 0 else self.node_num // 2 + 1
        labels = [str((k, j)) for k in range(1, upper_bound + 1) for j in range(1, self.node_num + 1)]
        values = np.zeros((self.node_num * upper_bound, self.node_num * upper_bound)).astype(np.uint32)
        df = pd.DataFrame(values, columns=labels, index=labels)

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
        df = pd.DataFrame(values, columns=labels, index=labels)

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


def get_degree_info_from_matix(adjmatrix):
    """ statistical information of a matrix
    :param adjmatrix: 
    :return: 
    """
    indegree_list = []
    outdegree_list = []
    node_num = adjmatrix.shape[0]

    for i in range(node_num):
        in_count = 0
        out_count = 0
        for j in range(node_num):
            in_count += adjmatrix[j][i]
            out_count += adjmatrix[i][j]
        indegree_list.append(in_count)
        outdegree_list.append(out_count)

    df = pd.DataFrame({
        'indegree': indegree_list,
        'outdegree': outdegree_list
    })

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
    columns_in = [
        'indegree_mean', 'indegree_mode_mean', 'indegree_var', 'indegree_max',
        'indegree_max_count', 'indegree_min', 'indegree_min_count', 'indegree_median'
    ]
    columns_out = [
        'outdegree_mean', 'outdegree_mode_mean', 'outdegree_var', 'outdegree_max',
        'outdegree_max_count', 'outdegree_min', 'outdegree_min_count', 'outdegree_median'
    ]

    new_df = None
    # global_degree_info = None
    for i in range(df.shape[0]):
        # get data degree info
        degree_df = get_degree_info_from_matix(df.iloc[i, :-2].values.reshape((node_num, node_num)))
        out_degree = degree_df.outdegree
        in_degree = degree_df.indegree

        # if global_degree_info is None:
        #     global_degree_info = degree_df
        # else:
        #     global_degree_info = pd.concat([global_degree_info, degree_df], axis=0)
        # continue

        # gather max/min intdegree
        in_max, in_min = \
            in_degree.max(), in_degree.min()
        # gather max/min indegree count
        in_max_count, in_min_count = \
            in_degree[in_degree == in_max].count(), in_degree[in_degree == in_min].count()
        # gather basic statistics
        in_mean, in_mode, in_var, in_median = \
            in_degree.mean(), in_degree.mode().mean(), in_degree.var(), in_degree.median()

        in_degree_list = \
            [in_mean, in_mode, in_var, in_max, in_max_count, in_min, in_min_count, in_median]

        in_degree_dict = dict(list(zip(columns_in, in_degree_list)))
        in_degree_df = pd.DataFrame(in_degree_dict, index=[0])

        # gather max/min outdegree
        out_max, out_min = \
            out_degree.max(), out_degree.min()
        # gather max/min outdegree count
        out_max_count, out_min_count = \
            out_degree[out_degree == out_max].count(), out_degree[out_degree == out_min].count()
        # gather basic statistics
        out_mean, out_mode, out_var, out_median = \
            out_degree.mean(), out_degree.mode().mean(), out_degree.var(), out_degree.median()

        out_degree_list = \
            [out_mean, out_mode, out_var, out_max, out_max_count, out_min, out_min_count, out_median]

        out_degree_dict = dict(list(zip(columns_out, out_degree_list)))
        out_degree_df = pd.DataFrame(out_degree_dict, index=[0])

        concat_degree_df = pd.concat([in_degree_df, out_degree_df], axis=1)
        if new_df is None:
            new_df = concat_degree_df
        else:
            new_df = pd.concat([new_df, concat_degree_df], axis=0)

        if i == 100:
            print(new_df.head(10))

    return new_df
    # return global_degree_info


def get_residual_info():
    pass


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
            new_df.iloc[index, df.iloc[i + j, -1]] += 1
        index += 1

    print(new_df.sample(100))
    # print(new_df.tail(10))

    return new_df


def get_node2vec_vectors(dir_path, file_name, node_num):
    con = config.Config()
    con.file_name_list = [dir_path + file_name]
    con.node_num = node_num
    data = DataGenerator(con)

    from model import node2vec
    import consensus_algo
    import time
    print('=> start node2vec')
    start = time.time()
    df = None
    num_walks = 10
    walk_length = 5
    is_directed = True
    columns = None

    # Return hyperparameter. Default is 1.
    p = 1
    # Inout hyperparameter. Default is 1.
    q = 1

    for x, y in data.get_origin_data(1):
        x = x.values.reshape((node_num, node_num))
        m = consensus_algo.NetworkAlgo(adjMatrix=x, directed=True)
        G = node2vec.Graph(m.G, is_directed, p, q)
        G.preprocess_transition_probs()
        node2vec.walks = G.simulate_walks(num_walks, walk_length)
        # size need to be a variable
        vectors = node2vec.learn_embeddings(node2vec.walks, size=7)
        if df is None:
            # the size of columns need to be a variable
            columns = ['d_{}'.format(k) for k in range(7)]
            columns.extend(y.columns)
            tmp = np.hstack([vectors, np.vstack([y.values] * vectors.shape[0])])
            # dtype may be a potential problem
            df = pd.DataFrame(tmp, dtype=np.float, columns=columns)
        else:
            tmp = np.hstack([vectors, np.vstack([y.values] * vectors.shape[0])])
            df = pd.concat([df, pd.DataFrame(tmp, columns=columns, dtype=np.float)], axis=0)

    print('=> embedding vectors data:')
    print(df.sample(100))
    df.to_csv(dir_path + "node2vec_emb_" + file_name)
    print('=> save finished:', dir_path + "node2vec_emb_" + file_name)
    print('=> time used', time.time() - start)


def get_eigen_vectors(dir_path, file_name, node_num):
    """ get eigenvectors from different matrix 
    :param dir_path: 
    :param file_name: 
        the origin adjmatrix file with labels in it
    :param node_num: 
    :return: 
    """

    # read the data generator class
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
        # v or v.T, it' s a question...
        tmp = np.hstack([v, np.vstack([y.values] * v.shape[0])])
        if df is None:
            # v.shape[0] is the number of eigenvectors
            columns = [str(i) for i in range(v.shape[0])]
            columns.extend(y.columns)
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


def get_matrix_data_with_channel():
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
    
    --------
    examples:
        before call this function:
            get_eigen_vectors('./data/non-isomorphism/convert_data/', 'node_num_8.csv', 8)
            k_means_cluster_for_eigenvectors('./data/non-isomorphism/convert_data/',
                                                        'eigenvector_node_num_8.csv', 8, 2)
        then:
            pipeline_all_preprocess_data('./data/non-isomorphism/convert_data/', 'node_num_8.csv',
                                 "k_means_cluster_eignvector_r_8.csv", 8, 2, keep_statistical_features=True)
    
    :param dir_path: 
    :param cluster_file_name: 
        after executing: 
            get_eigen_vectors, 
            *_cluster_for_eigenvectors(e.g., k_means_cluster_for_eigenvectors)
    :param origin_file_name:
            orgin adjacent matrix with r, s labels
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


def in_degree_count(adj_matrix):
    local_dict = {}

    for j in range(adj_matrix.shape[0]):
        count = 0
        for i in range(adj_matrix.shape[1]):
            if adj_matrix[i][j] == 1:
                count += 1

        if count not in local_dict:
            local_dict[count] = 1
        else:
            local_dict[count] += 1

    return local_dict


def statistic_in_degree(r_5_data_r_1_s_1, node_num):
    global_dict = {}
    global_dict_degree = {'indegree': [], 'count': []}

    for i in range(r_5_data_r_1_s_1.shape[0]):
        adj_matrix = r_5_data_r_1_s_1.iloc[i, :-2].values.reshape((node_num, node_num)).astype(np.int8)
        # print(adj_matrix)
        local_dict = in_degree_count(adj_matrix)
        # print(local_dict)
        if local_dict:
            for k in local_dict.keys():
                if 'in_degree_' + str(k) not in global_dict:
                    global_dict['in_degree_' + str(k)] = [local_dict[k]]
                else:
                    global_dict['in_degree_' + str(k)][0] += local_dict[k]
                global_dict_degree['indegree'].append('indegree_' + str(k))
                global_dict_degree['count'].append(local_dict[k])

    return global_dict, global_dict_degree


def draw_bar(dir_path, file_name, node_num):
    r_5_data = pd.read_csv(dir_path + file_name[0], header=0, index_col=0).reset_index(drop=True)
    r_6_data = pd.read_csv(dir_path + file_name[1], header=0, index_col=0).reset_index(drop=True)
    r_7_data = pd.read_csv(dir_path + file_name[2], header=0, index_col=0).reset_index(drop=True)
    for r in range(2):
        for s in range(2):

            data0 = r_5_data.loc[(r_5_data.r == r) & (r_5_data.s == s), :]
            data1 = r_6_data.loc[(r_6_data.r == r) & (r_6_data.s == s), :]
            data2 = r_7_data.loc[(r_7_data.r == r) & (r_7_data.s == s), :]
            # degree_df problem with returning empty list
            local_dict0, degree_count_dict0 = statistic_in_degree(data0, node_num[0])
            local_dict1, degree_count_dict1 = statistic_in_degree(data1, node_num[1])
            local_dict2, degree_count_dict2 = statistic_in_degree(data2, node_num[2])

            if local_dict0 and local_dict1 and local_dict2:
                plt.subplot(131)
                sns.boxplot(x='indegree', y='count',
                            data=pd.DataFrame(degree_count_dict0).sort_values(by='indegree', ascending=True))
                plt.subplot(132)
                sns.boxplot(x='indegree', y='count',
                            data=pd.DataFrame(degree_count_dict1).sort_values(by='indegree', ascending=True))
                plt.subplot(133)
                sns.boxplot(x='indegree', y='count',
                            data=pd.DataFrame(degree_count_dict2).sort_values(by='indegree', ascending=True))
                plt.show()


if __name__ == '__main__':
    con = config.Config()
    data = DataGenerator(con)
    data.get_cross_valid_data_set()
    # get_eigen_vectors('./data/directed/node_7/', 'r_7_modified.csv', 7)
    # k_means_cluster_for_eigenvectors('./data/directed/node_7/', 'eigenvector_r_7_modified.csv', 7, 2)

    # pipeline_all_preprocess_data('./data/directed/node_7/', 'r_7_modified.csv',
    #                              'k_means_cluster_eigenvector_r_7_modified.csv', 7, 2)

    # draw_bar('./data/directed/', ['node_5/r_5.csv', 'node_6/r_6.csv', 'node_7/r_7.csv'], [5, 6, 7])

    ## gather degree information
    # labels = pd.read_csv('./data/directed/node_7/r_7_modified.csv',
    #                      header=0, index_col=0).iloc[:,-2:].reset_index(drop=True)
    # df = get_degree_info('./data/directed/node_7/', 'r_7_modified.csv', 7, 2).reset_index(drop=True)
    # df = pd.concat([df, labels], axis=1)
    # print(df.sample(10))
    # df.to_csv('./data/directed/node_7/degree_statistical_r_7_modified.csv')

    ## draw correlation graph
    # from matplotlib.backends.backend_pdf import PdfPages
    #
    # # pp = PdfPages('./assets/corr.pdf')
    # df = pd.read_csv('./data/directed/node_7/degree_statistical_r_7_modified.csv', header=0, index_col=0)
    # f, ax = plt.subplots(figsize=(18, 18))
    # sns.heatmap(df.corr(), cmap="YlGnBu", annot=True, linewidths=.5, fmt='.1f', ax=ax)
    # # pp.savefig()
    # # pp.close()
    # plt.show()

    ## draw distibute
    # df2 = get_degree_info('./data/directed/node_7/', 'r_7.csv', 7, 2).reset_index(drop=True)
    # plt.subplot(121)
    # sns.distplot(df2.indegree)
    #
    # plt.subplot(122)
    # sns.distplot(df2.outdegree)
    # plt.savefig('./assets/distribute')
    # plt.show()

    ## draw boxplot
    # print(df2.head(10))
    # df3 = pd.DataFrame(np.zeros(df2.shape), dtype=np.int8, columns=['label']).reset_index(drop=True)
    # df = pd.concat([df2, df3], axis=1)
    # sns.boxplot(x=df.columns[1], y=df.columns[0], data=df)
    # plt.show()
