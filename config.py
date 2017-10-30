from __future__ import print_function
from __future__ import division


class Config:
    def __init__(self):
        self.file_name_list = [
            "/Users/wangguang/PycharmProjects/my_experiment/data/r_1_train_data.csv",
            "/Users/wangguang/PycharmProjects/my_experiment/data/r_2_train_data.csv",
            "/Users/wangguang/PycharmProjects/my_experiment/data/r_3_train_data.csv",
            "/Users/wangguang/PycharmProjects/my_experiment/data/r_4_train_data.csv",
            "/Users/wangguang/PycharmProjects/my_experiment/data/r_5_train_data.csv"
        ]

        self.n_classes = 50

        self.structure = [None, 9, 5]

        self.alpha = 500
        self.gamma = 1
        self.reg = 1
        self.beta = 10

        self.batch_size = 10
        self.epochs_limits = 10
        self.learning_rate = 0.01
        self.display = 1

        self.DBN_init = True
        self.dbn_epochs = 10
        self.dbn_batch_size = 10
        self.dbn_learning_rate = 0.1

        self.sparse_dot = False
        self.ng_sample_ratio = 0.0



