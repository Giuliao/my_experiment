from __future__ import print_function
from __future__ import division
import tensorflow as tf


class Config:
    def __init__(self):
        self.file_name_list = [
            "./data/node_10/r_1_train_data_new.csv",
            "./data/node_10/r_2_train_data_new.csv",
            "./data/node_10/r_3_train_data_new.csv",
            "./data/node_10/r_4_train_data_new.csv",
            "./data/node_10/r_5_train_data_new.csv"
        ]
        # self.file_name_list = [
        #     "./data/node_10/r_1_train_origin_data.csv",
        #     "./data/node_10/r_2_train_origin_data.csv",
        #     "./data/node_10/r_3_train_origin_data.csv",
        #     "./data/node_10/r_4_train_origin_data.csv",
        #     "./data/node_10/r_5_train_origin_data.csv"
        # ]

        # self.file_name_list = [
        #     "./data/node_6/r_1_train_origin_data.csv",
        #     "./data/node_6/r_2_train_origin_data.csv",
        #     "./data/node_6/r_3_train_origin_data.csv",
        #     "./data/node_6/r_4_train_origin_data.csv",
        #     "./data/node_6/r_5_train_origin_data.csv"
        # ]

        self.n_classes = 5
        self.input_size = 100
        self.reg = 1
        self.alpha = 100
        self.beta = 1
        self.node_size = 10
        self.decay_steps = 200
        self.decay_rate = 0.99
        self.structure = {
            # 'structure': ['cnn1', 'cnn2', 'pool1', 'cnn3', 'cnn4',
            #               'pool2', 'full1', 'full2', 'full3', 'out_layer'],
            'structure': ['full1', 'full2', 'full3','full3.5', 'full4', 'full5',
                          'full6', 'full7', 'full8', 'full9', 'out_layer'],
            # 'cnn1': {
            #     'struct': [5, 5, 1, 32],
            #     'padding': 'SAME',
            #     'strides': [1, 1, 1, 1],
            # },
            #
            # 'cnn2': {
            #     'struct': [5, 5, 32, 64],
            #     'padding': 'SAME',
            #     'strides': [1, 1, 1, 1]
            # },
            # 'pool1': {
            #     'ksize': [1, 2, 2, 1],
            #     'padding': 'SAME',
            #     'strides': [1, 2, 2, 1]
            # },
            # 'cnn3': {
            #     'struct': [3, 3, 64, 128],
            #     'padding': 'SAME',
            #     'strides': [1, 1, 1, 1],
            #     'dropout': True
            # },
            # 'cnn4': {
            #     'struct': [3, 3, 128, 256],
            #     'padding': 'SAME',
            #     'strides': [1, 1, 1, 1],
            #     'dropout': False
            # },
            # 'pool2': {
            #     'ksize': [1, 2, 2, 1],
            #     'padding': 'SAME',
            #     'strides': [1, 2, 2, 1]
            # },

            'full1': {
                'struct': [self.input_size, 128],
                'dropout': False
            },
            'full2': {
                'struct': [128, 256],
                'dropout': True
            },
            'full3': {
                'struct': [256, 360],
                'dropout': False
            },
            'full3.5': {
                'struct': [360, 512],
                'dropout': False
            },
            'full4': {
                'struct': [512, 256],
                'dropout': True
            },
            'full5': {
                'struct': [256, 128],
                'dropout': False
            },
            'full6': {
                'struct': [128, 64],
                'dropout': False
            },
            'full7': {
                'struct': [64, 32],
                'dropout': True
            },
            'full8': {
                'struct': [32, 16],
                'dropout': False
            },
            'full9': {
                'struct': [16, 8],
                'dropout': True
            },
            'out_layer': {
                'struct': [8, self.n_classes],
                'act': tf.identity
            }
        }

        self.test_path_to_log = './logs/test_10_node_with_10full/'
        self.train_path_to_log = './logs/train_10_node_with_10full/'
        self.epochs = 10000
        self.model_path = './model/logs/saved_model_node_10_with_10full/'
        self.alpha = 500
        self.gamma = 1
        self.reg = 1
        self.beta = 10
        self.keep_prob = 0.7

        self.batch_size = 256
        self.epochs_limits = 10
        self.learning_rate = 0.001
        self.display = 1

        self.DBN_init = True
        self.dbn_epochs = 10
        self.dbn_batch_size = 10
        self.dbn_learning_rate = 0.002

        self.sparse_dot = False
        self.ng_sample_ratio = 0.0






