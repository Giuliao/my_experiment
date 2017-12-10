from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import pandas as pd
import numpy as np
import traceback
import scipy
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from model import config


class DataGenerator:
    def __init__(self, con):
        self.file_name_list = con.file_name_list
        self.n_classes = con.n_classes
        self.train_X, self.train_Y, self.test_X, self.test_Y = self.read_from_csv_list()
        self.total_train = self.train_X.shape[0]
        print('=> data init finished')

    def read_from_csv_list(self):
        pd_ll = []
        for local_file in self.file_name_list:
            pd_ll.append(pd.read_csv(local_file, header=0, index_col=0))

        df = pd.concat(pd_ll)
        # print(df.head())
        valid_X = df.iloc[:, : -self.n_classes].reset_index(drop=True)
        valid_Y = df.iloc[:, -self.n_classes:].reset_index(drop=True)
        train_X, test_X, train_Y, test_Y = train_test_split(valid_X, valid_Y, test_size=0.2)
        # test_X = df.iloc[-4096:, : -self.n_classes].reset_index(drop=True)
        # test_Y = df.iloc[-4096:, -self.n_classes:].reset_index(drop=True)

        print('=> train data size:', train_X.shape[0])
        print('=> test data size:', test_X.shape[0])
        print(train_X.head())
        print(train_Y.head())
        return train_X, train_Y, test_X, test_Y

    def get_train_data(self, batch_size):
        for i in range(0, self.total_train, batch_size):
            yield self.train_X.iloc[i: i + batch_size, :], self.train_Y.iloc[i: i + batch_size, :]

        return

    def get_test_data(self):
        return self.test_X, self.test_Y


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
    xedges = [_ / 7 for _ in range(-14, 15)]
    yedges = [_ / 7 for _ in range(-14, 15)]
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
            print(H)
            plt.imshow(H, cmap=plt.cm.gray, interpolation='nearest', origin='low',
                       extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
            plt.show()


if __name__ == '__main__':
   get_image_data()
