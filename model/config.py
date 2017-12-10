from __future__ import print_function
from __future__ import division
from  __future__ import absolute_import
import json


class Config:
    def __init__(self, json_file=None):
        if not json_file:
            self.file_name_list = [
                "./data/undirected/node_10/r_1.csv",
                "./data/undirected/node_10/r_2.csv",
                "./data/undirected/node_10/r_3.csv",
                "./data/undirected/node_10/r_4.csv",
                "./data/undirected/node_10/r_5.csv"
            ]

            self.decay_steps = 200
            self.decay_rate = 0.99
            self.n_classes = 2
            self.input_size = 3920
            self.image_size = 28
            self.channel = 5
            self.structure = {
                'structure': ['cnn1', 'pool1', 'cnn2', 'pool2',
                              'cnn3', 'cnn4', 'pool3', 'full1',
                              'full2', 'full3', 'full4', 'out_layer'
                             ],

                'cnn1': {
                    'struct': [5, 5, 5, 32],
                    'padding': 'SAME',
                    'strides': [1, 1, 1, 1],
                },
                'pool1': {
                    'ksize': [1, 2, 2, 1],
                    'padding': 'SAME',
                    'strides': [1, 2, 2, 1]
                },

                'cnn2': {
                    'struct': [3, 3, 32, 32],
                    'padding': 'SAME',
                    'strides': [1, 1, 1, 1]
                },
                'pool2': {
                    'ksize': [1, 2, 2, 1],
                    'padding': 'SAME',
                    'strides': [1, 2, 2, 1]
                },
                'cnn3': {
                    'struct': [3, 3, 32, 64],
                    'padding': 'SAME',
                    'strides': [1, 1, 1, 1],
                    'dropout': False
                },
                'cnn4': {
                    'struct': [3, 3, 64, 64],
                    'padding': 'SAME',
                    'strides': [1, 1, 1, 1],
                    'dropout': True
                },
                'pool3': {
                    'ksize': [1, 2, 2, 1],
                    'padding': 'SAME',
                    'strides': [1, 2, 2, 1]
                },
                'full1': {
                    'struct': [4*4*64, 1024],
                    'dropout': True
                },
                'full2': {
                    'struct': [1024, 512],
                    'dropout': False
                },
                'full3': {
                    'struct': [512, 256],
                    'dropout': False
                },
                'full4': {
                    'struct': [256, 16],
                    'dropout': False
                },
                'out_layer': {
                    'struct': [16, self.n_classes],
                    'act': "identity"
                }
            }

            self.test_path_to_log = './logs/tensorboard_with_reg/test_10_node/'
            self.train_path_to_log = './logs/tensorboard_with_reg/train_10_node/'
            # self.path_to_save_predict
            self.epochs = 1000  # 80# 65
            self.model_path = './logs/saved_model/v_with_reg_saved/'
            self.alpha = 500
            self.gamma = 1
            self.reg = 1
            self.beta = 1
            self.keep_prob = 0.7

            self.batch_size = 256
            self.learning_rate = 0.001
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
        else:
            with open(json_file, 'r') as f:
                json_dict = json.loads(f.read())
                self.file_name_list=json_dict['file_name_list']
                self.structure = json_dict['structure']
                self.decay_steps = json_dict['decay_steps']
                self.decay_rate = json_dict['decay_rate']
                self.n_classes = json_dict['n_classes']
                self.input_size = json_dict['input_size']
                self.image_size = json_dict['image_size']
                self.channel = json_dict['channel']
                self.test_path_to_log = json_dict['test_path_to_log']
                self.train_path_to_log = json_dict['train_path_to_log']
                self.epochs = json_dict['epochs']
                self.model_path = json_dict['model_path']
                self.alpha = json_dict['alpha']
                self.gamma = json_dict['gamma']
                self.reg = json_dict['reg']
                self.beta = json_dict['beta']
                self.keep_prob = json_dict['keep_prob']

                self.batch_size = json_dict['batch_size']
                self.learning_rate = json_dict['learning_rate']
                print('=> json file read finished...')








