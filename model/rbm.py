from __future__ import print_function
from __future__ import division
from basic_model import basic
import tensorflow as tf
import numpy as np


class rbm(basic):
    def __init__(self, shape, para):
        basic.__init__(self)
        self.para = para
        self.sess = tf.Session()
        stddev = 1.0 / np.sqrt(shape[0])
        self.W = self.weight_variable(shape, stddev=stddev)
        self.bv = tf.Variable(tf.zeros(shape[0]))
        self.bh = tf.Variable(tf.zeros(shape[1]))
        self.vis = tf.placeholder("float", [None, shape[0]])
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.buildModel()
        print('rbm init completely')

    def nn_layer(self, input_tensor, weights, biases, layer_name, act=tf.nn.relu):
        """
        :param input_tensor: 
        :param weights: 
        :param biases: 
        :param layer_name: 
        :param act: 
        :return: 
        """
        with tf.name_scope(layer_name):
            with tf.name_scope('weight'):
                self.variable_summaries(weights)
            with tf.name_scope('biases'):
                self.variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):
                pre_activate = tf.matmul(input_tensor, weights) + biases
                tf.summary.histogram('pre_activations', pre_activate)

            activations = act(pre_activate, name='activation')
            tf.summary.histogram('activations', activations)
            return activations

    def buildModel(self):
        v_2_h_layer1 = self.nn_layer(self.vis, self.W, self.bh, 'visible_2_hidden_layer_1', act=tf.sigmoid)
        self.h = self.sample(v_2_h_layer1)

        # gibbs sample
        h_2_v_layer1 = self.nn_layer(self.h, tf.transpose(self.W), self.bv, 'hidden_2_visible_layer_1', act=tf.sigmoid)
        v_sample = self.sample(h_2_v_layer1)

        v_2_h_layer2 = self.nn_layer(v_sample, self.W, self.bh, 'visible_2_hidden_layer_2', act=tf.sigmoid)
        h_sample = self.sample(v_2_h_layer2)

        lr = self.para['learning_rate'] / tf.to_float(self.para['batch_size'])
        W_addr = self.W.assign(lr*(tf.matmul(tf.transpose(self.vis), self.h)-tf.matmul(tf.transpose(v_sample), h_sample)))
        bv_addr = self.bv.assign(lr*tf.reduce_mean(self.vis-v_sample, 0))
        bh_addr = self.bh.assign(lr*tf.reduce_mean(self.h-h_sample, 0))

        self.upt = [W_addr, bv_addr, bh_addr]
        self.error = tf.reduce_sum(tf.pow(self.vis-v_sample, 2))

    def sample(self, probs):
        return tf.floor(probs+tf.random_uniform(tf.shape(probs), 0, 1))

    def fit(self, data):
        _, ret = self.sess.run((self.upt, self.error), feed_dict={self.vis: data})
        return ret

    def getWb(self):
        return self.sess.run([self.W, self.bv, self.bh])

    def getH(self, data):
        return self.sess.run(self.h, feed_dict={self.vis: data})

    def close(self):
        self.sess.close()
