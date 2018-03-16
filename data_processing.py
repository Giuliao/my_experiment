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
import matplotlib.pyplot as plt
import seaborn as sns


class DataGenerator:
    def __init__(self, con):
        self.file_name_list = con.file_name_list
        self.n_classes = con.n_classes
        self.valid_X, self.valid_Y, self.train_X, self.train_Y, self.test_X, self.test_Y = self.read_from_csv_list(
            test_size=0.3)

        self.total_data_num = self.valid_X.shape[0]
        self.total_train = self.train_X.shape[0]
        self.node_num = con.node_num
        self.class_number_list = con.class_number_list
        self.problem_type = con.problem_type_list[0]
        print('=> data init finished')

    def read_from_csv_list(self, test_size):
        pd_ll = []
        for local_file in self.file_name_list:
            pd_ll.append(pd.read_csv(local_file, header=0, index_col=0))

        df = pd.concat(pd_ll)
        # sift data
        # df = df.loc[(df.r == 2) & (df.s == 6), :]
        # print(df.head())
        valid_X = df.iloc[:, : -self.n_classes].reset_index(drop=True)
        valid_Y = df.iloc[:, -self.n_classes:].reset_index(drop=True)
        train_X, test_X, train_Y, test_Y = train_test_split(valid_X, valid_Y, test_size=test_size)
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
        for i in range(0, self.total_train, batch_size):
            yield self.train_X.iloc[i: i + batch_size, :], self.train_Y.iloc[i: i + batch_size, :]

        return

    def get_origin_data(self, batch_size):
        for i in range(0, self.total_data_num, batch_size):
            yield self.valid_X.iloc[i: i + batch_size, :], self.valid_Y.iloc[i: i + batch_size, :]

    def get_test_data(self):
        return self.test_X, self.test_Y

    @staticmethod
    def get_visualization(H, shape=None):
        if shape:
            H = H.reshape(shape)
        # H = np.flipud(H)  # flip matrix up 2 down because plt.imshow will flip origin matrix
        # print(H)
        sns.heatmap(H, fmt='d', linewidths=0.05)
        # plt.imshow(H, cmap=plt.cm.jet, interpolation='nearest', origin='low')
        plt.show()

    def get_eigenvectors(self):
        for x, y in self.get_origin_data(1):
            m = x.values.reshape((self.node_num, self.node_num))
            # print(m+m.T)
            n = m + m.T
            # n = np.dot(m, m.T) + np.dot(m.T, m)
            e, v = scipy.linalg.eigh(n)  # np.linalg.eig will return the complex data sometimes...
            yield v.astype(np.float32), m, y

    def get_classify_confusion_matrix(self, predict):
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
            df[idx_real][idx_predict] += 1

        f, ax = plt.subplots(figsize=(18, 18))
        sns.heatmap(df, annot=True, linewidths=2, fmt='d', ax=ax)
        plt.savefig('./assets/confusion_matrix')
        plt.close()

    def get_regression_confusion_matrix(self, predict):

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
            df[idx_real][idx_predict] += 1

        f, ax = plt.subplots(figsize=(18, 18))
        sns.heatmap(df, annot=True, linewidths=2, fmt='d', ax=ax)
        plt.savefig('./assets/confusion_matrix')
        plt.close()

    def get_confusion_matrix(self, predict, problem_type):
        if problem_type:
            self.get_regression_confusion_matrix(predict)
        else:
            self.get_classify_confusion_matrix(predict)


def get_degree_info(adjmatrix):
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
    print(df.head())
    return df


def learning_vector_quantization_cluster(node_num, yita):
    df = pd.read_csv('./data/directed/node_7/eignvector_r_7.csv', header=0, index_col=0, dtype=np.float32).reset_index(
        drop=True)
    df = pd.concat([df, pd.DataFrame(np.zeros((df.shape[0], 1)), columns=['cluster_class'], dtype=np.int32)], axis=1)
    df.r = df.r.astype(np.int32)
    df.s = df.s.astype(np.int32)
    print(df.head())

    centroid = []

    for i in range(node_num):
        if df.loc[df.s == i + 1, :].shape[0] != 0:
            centroid.append(df.loc[df.s == i + 1, :].sample(1).values.astype(np.float32).reshape(1, -1))
    # old_centroid = [None] * len(centroid)
    print(centroid)

    count = 0
    # delta = 10000
    while count < 10:

        for i in range(df.shape[0]):
            v1 = df.iloc[i, :-3].values.astype(np.float32).reshape(1, -1)

            distance = []

            for j in range(len(centroid)):
                distance.append(np.mean(np.square(np.subtract(v1, centroid[j][:, -3])), axis=1))

            # print(distance)
            # print(np.argmin(distance))
            idx = np.argmin(distance)

            # save old centroid
            # for i in range(len(centroid)):
            #     old_centroid[i] = centroid[i].copy()

            # http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
            df.loc[i, 'cluster_class'] = idx
            if df['s'][i] == np.int32(centroid[idx][:, -2]):
                centroid[idx] = np.hstack((np.add(centroid[idx][:, -3], v1 * yita), centroid[idx][:, -3:]))
            else:
                centroid[idx] = np.hstack((np.subtract(centroid[idx][:, -3], v1 * yita), centroid[idx][:, -3:]))

        print(centroid)
        print(df.head(100))
        print(df.tail(100))

        count += 1




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
                m[i][j] = m[i][j] * 5

    return m


def get_cluser_data():
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


def make_receptive_by_sub_graph():
    df = pd.read_csv('./data/undirected/node_10/r_1.csv', index_col=0, header=0)
    for i in range(df.shape[0]):
        tt = df.iloc[i, :-2].values
        tt = tt.reshape((10, 10))
        print(tt)


if __name__ == '__main__':
    # test_image()
    # make_receptive_by_sub_graph()
    # get_cluser_data()
    # con = config.Config()
    # data = DataGenerator(con)
    # import consensus_algo
    learning_vector_quantization_cluster(7, 0.5)
    # adj = consensus_algo.NetworkAlgo(p=0.5).adjMatrix
    # print(adj)
    # get_degree_info(adj)

    # df = None
    # columns = None
    # count = 0
    # for v, x, y in data.get_eigenvectors():
    #
    #     tmp = np.hstack([v, np.vstack([y.values]*v.shape[0])])
    #     if df is None:
    #         columns = [str(i) for i in range(v.shape[0])]
    #         columns.extend(['r', 's'])
    #         df = pd.DataFrame(tmp, columns=columns)
    #
    #     else:
    #         df = pd.concat([df, pd.DataFrame(tmp, columns=columns)], axis=0)
    #
    #     count += 1
    #
    #     if count % 2000 == 0:
    #         print(df.head(10))
    #
    # df.to_csv("./data/directed/node_7/eignvector_r_7.csv")
