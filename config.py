from __future__ import print_function
from __future__ import division


class Config:
    def __init__(self):
        self.file_name_list = [
            "/Users/wangguang/PycharmProjects/my_experiment/data/r_1_train_data_new.csv",
            "/Users/wangguang/PycharmProjects/my_experiment/data/r_2_train_data_new.csv",
            "/Users/wangguang/PycharmProjects/my_experiment/data/r_3_train_data_new.csv",
            "/Users/wangguang/PycharmProjects/my_experiment/data/r_4_train_data_new.csv",
            "/Users/wangguang/PycharmProjects/my_experiment/data/r_5_train_data_new.csv"
        ]

        self.n_classes = 5
        self.input_size = 100

        self.structure = {
            'structure': ['cnn1', 'cnn2', 'pool1', 'cnn3', 'cnn4',
                          'pool2', 'full1', 'full2', 'full3', 'out_layer'],
            'cnn1': {
                'struct': [5, 5, 1, 32],
                'padding': 'SAME',
                'strides': [1, 1, 1, 1],
            },

            'cnn2': {
                'struct': [5, 5, 32, 64],
                'padding': 'SAME',
                'strides': [1, 1, 1, 1]
            },
            'pool1': {
                'ksize': [1, 2, 2, 1],
                'padding': 'SAME',
                'strides': [1, 2, 2, 1]
            },
            'cnn3': {
                'struct': [3, 3, 64, 128],
                'padding': 'SAME',
                'strides': [1, 1, 1, 1],
                'dropout': True
            },
            'cnn4': {
                'struct': [3, 3, 128, 256],
                'padding': 'SAME',
                'strides': [1, 1, 1, 1],
                'dropout': False
            },
            'pool2': {
                'ksize': [1, 2, 2, 1],
                'padding': 'SAME',
                'strides': [1, 2, 2, 1]
            },

            'full1': {
                'struct': [3*3*256, 1000],
                'dropout': False
            },
            'full2': {
                'struct': [1000, 500],
                'dropout': True
            },
            'full3': {
                'struct': [500, 250],
                'dropout': False
            },
            'out_layer': {
                'struct': [250, self.n_classes]
            }
        }

        self.test_path_to_log = './logs/test_10_node/'
        self.train_path_to_log = './logs/train_10_node/'
        self.epochs = 100
        self.model_path = './saved_model/'
        self.alpha = 500
        self.gamma = 1
        self.reg = 1
        self.beta = 10
        self.keep_prob = 0.7

        self.batch_size = 100
        self.epochs_limits = 10
        self.learning_rate = 0.01
        self.display = 1

        self.DBN_init = True
        self.dbn_epochs = 10
        self.dbn_batch_size = 10
        self.dbn_learning_rate = 0.1

        self.sparse_dot = False
        self.ng_sample_ratio = 0.0






