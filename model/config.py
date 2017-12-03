from __future__ import print_function
from __future__ import division
from  __future__ import absolute_import
import tensorflow as tf

class Config:
    def __init__(self):
        # self.file_name_list = [
        #     "/Users/wangguang/PycharmProjects/my_experiment/data/r_1_train_data_new.csv",
        #     "/Users/wangguang/PycharmProjects/my_experiment/data/r_2_train_data_new.csv",
        #     "/Users/wangguang/PycharmProjects/my_experiment/data/r_3_train_data_new.csv",
        #     "/Users/wangguang/PycharmProjects/my_experiment/data/r_4_train_data_new.csv",
        #     "/Users/wangguang/PycharmProjects/my_experiment/data/r_5_train_data_new.csv"
        # ]

        self.file_name_list = [
            "./data/undirected/node_10/image.csv",
        ]

        self.decay_steps = 200
        self.decay_rate = 0.99
        self.n_classes = 2
        self.input_size = 3920
        self.image_size = 28

        self.structure = {
            'structure': ['cnn1', 'cnn2', 'pool1', 'cnn3', 'cnn4',
                          'pool2', 'full1', 'full2', 'full3', 'full4',
                          'full5', 'full6', 'full7', 'out_layer'],

            'cnn1': {
                'struct': [5, 5, 5, 32],
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
                'dropout': False
            },
            'cnn4': {
                'struct': [3, 3, 128, 256],
                'padding': 'SAME',
                'strides': [1, 1, 1, 1],
                'dropout': True
            },
            'pool2': {
                'ksize': [1, 2, 2, 1],
                'padding': 'SAME',
                'strides': [1, 2, 2, 1]
            },

            'full1': {
                'struct': [7*7*256, 1024],
                'dropout': False
            },
            'full2': {
                'struct': [1024, 512],
                'dropout': True
            },
            'full3': {
                'struct': [512, 256],
                'dropout': False
            },
            'full4': {
                'struct': [256, 128],
                'dropout': False
            },
            'full5': {
                'struct': [128, 64],
                'dropout': False
            },
            'full6': {
                'struct': [64, 32],
                'dropout': False
            },
            'full7': {
                'struct': [32, 16],
                'dropout': False
            },
            'out_layer': {
                'struct': [16, self.n_classes],
                'act': tf.identity
            }
        }

        self.test_path_to_log = './logs/tensorboard/test_10_node/'
        self.train_path_to_log = './logs/tensorboard/train_10_node/'
        # self.path_to_save_predict = './data/predict/{0}/{1}.csv'.format(self.mall_id, self.shop_id)
        self.epochs = 100
        self.model_path = './logs/saved_model/saved'
        self.alpha = 500
        self.gamma = 1
        self.reg = 1
        self.beta = 10
        self.keep_prob = 0.7

        self.batch_size = 10
        self.learning_rate = 0.01
        self.display = 1

        self.DBN_init = True
        self.dbn_epochs = 10
        self.dbn_batch_size = 10
        self.dbn_learning_rate = 0.1

        self.sparse_dot = False
        self.ng_sample_ratio = 0.0

        self.num_visible = 10
        self.num_hidden = 9
        self.visible_unit_type = 'bin'
        self.main_dir = './logs/'
        self.model_name = 'rbm_model'
        self.gibbs_sampling_steps = 1

        self.num_epochs = 50
        self.stddev = 0.01
        self.verbose = 0





