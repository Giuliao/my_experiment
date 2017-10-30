from __future__ import print_function
from __future__ import division
import pandas as pd
import numpy as np


class Data:
    def __init__(self):
        self.X = None
        self.adjacent_matriX = None
        self.label = None


class Graph:
    def __init__(self, file_name_list, n_classes):
        self.file_name_list = file_name_list
        self.n_classes = n_classes
        self.label = None
        self.train_X, self.train_Y = self.read_from_csv_list()
        self.N = self.train_X.shape[0]
        self.is_epoch_end = False
        self.st = 0

    def read_from_csv_list(self):
        pd_ll = []
        # upper_bound = 3000
        for local_file in self.file_name_list:
            pd_ll.append(pd.read_csv(local_file, header=0, index_col=0))

        df = pd.concat(pd_ll)
        df = df.sample(frac=1, axis=0)  # shuffle
        train_X = np.array(df.iloc[:, : -self.n_classes].values, dtype=np.float)
        train_Y = np.array(df.iloc[:, -self.n_classes:].values, dtype=np.float)
        # train_X = np.array(df.iloc[: upper_bound, : -self.n_classes].values, dtype=np.float)
        # train_Y = np.array(df.iloc[: upper_bound, -self.n_classes:].values, dtype=np.float)
        # train_Y = train_Y.reshape(train_Y.shape[0], 1)

        # test_X = np.array(df.iloc[upper_bound:, : -self.n_classes].values, dtype=np.float)
        # test_Y = np.array(df.iloc[upper_bound:, -self.n_classes:].values, dtype=np.float)
        # test_Y = test_Y.reshape(test_Y.shape[0], 50)

        return train_X, train_Y


    def __negativeSample(self, ngSample, count, edges):
       pass

    def sample(self, batch_size, do_shuffle=True, with_label=False):

        if self.is_epoch_end:
            self.st = 0
            self.is_epoch_end = False

        mini_batch = Data()

        mini_batch.X = self.train_X[self.st].reshape((10, 10))
        mini_batch.adjacent_matriX = self.train_X[self.st].reshape((10, 10))

        self.st += 1
        if self.st == self.train_X.shape[0]:
            self.st = 0
            self.is_epoch_end = True

        return mini_batch








