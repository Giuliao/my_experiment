from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import data_processing
from model import config
import traceback


class NeuralNetwork:

    def __init__(self, con):

        self.struct = con.structure

        self.layers_number = len(self.struct['structure'])
        self.W = {}
        self.b = {}

        self.batch_size = con.batch_size
        self.epochs = con.epochs
        self.input_size = con.input_size
        self.n_classes = con.n_classes
        self.my_keep_prob = con.keep_prob
        self.image_size = con.image_size
        self.learning_rate = con.learning_rate

        self.alpha = con.alpha
        self.beta = con.beta
        self.reg_lambda = con.reg

        self.decay_steps = con.decay_steps
        self.decay_rate = con.decay_rate

        # self.path_to_save_predict = con.path_to_save_predict
        self.model_path = con.model_path

        self.input = None
        self.target = None
        self.keep_prob = None

        self.epoch_optimizer = None
        self.epoch_step = None

        self.global_step = None
        self.optimizer = None
        self.loss = None
        self.acc = None
        self.out = None
        self.train_merged_summary_op = None
        self.test_merged_summary_op = None
        self.sess = None

    def init_session(self, con):
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = False
        self.sess = tf.Session(config=tf_config)

        self.train_writer = tf.summary.FileWriter(con.train_path_to_log, self.sess.graph)
        self.test_writer = tf.summary.FileWriter(con.test_path_to_log)

        self.model_path = con.model_path
        # self.path_to_save_predict = con.path_to_save_predict

        if os.path.exists(self.model_path):
            self.restore_model()
        else:
            self.sess.run(tf.global_variables_initializer())

    def _make_optimizer(self):
        with tf.name_scope('Optimizer'):
            local_step = tf.Variable(0, trainable=False, name="global_step")
            learning_rate = tf.train.exponential_decay(
                learning_rate=self.learning_rate,
                global_step=local_step,
                decay_steps=self.decay_steps,
                decay_rate=self.decay_rate,
                staircase=True
            )
            optimize = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=local_step)
        return optimize, local_step

    def _make_loss(self):
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.out, labels=self.target))

        def get_reg_loss(weight, biases):
            ret = tf.add_n([tf.nn.l2_loss(w) for w in weight.values()])
            ret = ret + tf.add_n([tf.nn.l2_loss(b) for b in biases.values()])
            return ret

        loss = tf.reduce_sum(tf.pow(self.out - self.target, 2)) + \
               self.reg_lambda*get_reg_loss(self.W, self.b)
        return loss

    def variable_summaries(self, var, name):
        """
        :param var: 
        :return: 
        """
        with tf.name_scope(name):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def nn_layer(self, input_tensor, dimension, layer_name, act=tf.nn.relu, cnn=False, pool=False):

        if pool:
            with tf.name_scope(layer_name):
                activations = tf.nn.max_pool(input_tensor,
                                             ksize=dimension,
                                             strides=self.struct[layer_name]['strides'],
                                             padding=self.struct[layer_name]['padding'])
            return activations

        with tf.name_scope(layer_name):
            with tf.name_scope('weight'):
                weights = tf.Variable(tf.random_normal(dimension, stddev=0.1), name=layer_name)
                self.W[layer_name] = weights
                # self.variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = tf.Variable(tf.random_normal(dimension[-1:], mean=0, stddev=0.1), name=layer_name)
                self.b[layer_name] = biases
                # self.variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):
                if cnn:
                    preactivate = tf.nn.conv2d(input_tensor, weights,
                                               strides=self.struct[layer_name]['strides'],
                                               padding=self.struct[layer_name]['padding'])
                else:
                    preactivate = tf.matmul(input_tensor, weights) + biases

                if 'drop_out' in self.struct[layer_name] and self.struct[layer_name]['dropout']:
                    preactivate = tf.nn.dropout(preactivate, keep_prob=self.keep_prob, name='dropout')
                    # tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activation')

            # tf.summary.histogram('activations', activations)
            return activations

    def build_network_model(self):

        def make_computing_graph(origin_data):

            if 'cnn1' in self.struct['structure'] or \
                            'cnn' in self.struct['structure']:
                local_input = tf.reshape(origin_data, [-1, self.image_size, self.image_size, 5])
            else:
                local_input = origin_data

            for i in range(self.layers_number):
                layer_name = self.struct['structure'][i]
                if 'cnn' in layer_name:
                    dimension = self.struct[layer_name]['struct']
                    local_input = self.nn_layer(local_input, dimension, layer_name, cnn=True)
                elif 'pool' in layer_name:
                    dimension = self.struct[layer_name]['ksize']
                    local_input = self.nn_layer(local_input, dimension, layer_name, pool=True)
                elif 'full' in layer_name:
                    dimension = self.struct[layer_name]['struct']
                    if local_input.shape[1] != dimension[0]:
                        local_input = tf.reshape(local_input, [-1, dimension[0]])
                    local_input = self.nn_layer(local_input, dimension, layer_name)
                elif 'out' in layer_name:
                    dimension = self.struct[layer_name]['struct']
                    local_input = self.nn_layer(local_input, dimension, layer_name, act=self.struct[layer_name]['act'])

            return local_input

        test_sum = []

        self.input = tf.placeholder(tf.float32, [None, self.input_size], name='input')
        self.target = tf.placeholder(tf.float32, [None, self.n_classes], name='labels')

        with tf.name_scope('Model'):
            self.keep_prob = tf.placeholder(tf.float32)
            self.out = make_computing_graph(self.input)

        with tf.name_scope('Loss'):
            self.loss = self._make_loss()
        test_sum.append(tf.summary.scalar("loss", self.loss))

        with tf.name_scope('Optimizer'):
            self.optimizer, self.global_step = self._make_optimizer()

	with tf.device('/cpu:0'):
            with tf.name_scope('Epoch_optimizer'):
                self.epoch_optimizer, self.epoch_step = self._make_optimizer()

        with tf.name_scope('Accuracy'):
            self.acc = self.get_accuracy()
        test_sum.append(tf.summary.scalar("acc", self.acc))

        self.train_merged_summary_op = tf.summary.merge_all()
        self.test_merged_summary_op = tf.summary.merge(test_sum)

    def save_model(self):
        saver = tf.train.Saver(tf.global_variables())
        saver.save(self.sess, self.model_path)

    def restore_model(self):
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(self.sess, self.model_path)
        return self.sess

    def close(self):
        if self.sess:
            self.sess.close()
            if self.test_writer:
                self.test_writer.close()
            if self.train_writer:
                self.train_writer.close()

    def fit(self, train_X, train_Y):
        feed_dict = {self.input: train_X, self.target: train_Y, self.keep_prob: self.my_keep_prob}
        ret, _, out, step, summ = self.sess.run([self.loss, self.optimizer, self.out, self.global_step, self.train_merged_summary_op], feed_dict=feed_dict)
        return ret, out, step, summ

    def get_accuracy(self):
        # correct = tf.equal(tf.argmax(self.out, 1), tf.argmax(self.target, 1))
        # acc = tf.reduce_mean(tf.cast(correct, 'float'))
        error = tf.reduce_sum(self.out-self.target)
        return error

    def get_W(self):
        return self.sess.run(self.W)

    def get_b(self):
        return self.sess.run(self.b)

    def predict_eval_data(self, input_data):
        pass
        # return predict

    def run_train(self, input_data):
        test_X, test_Y = input_data.get_test_data()
        ss = self.sess.run(self.epoch_step)
        try:
            if ss < self.epochs:
                for k in range(self.epochs):
                    epoch_loss = 0
                    number_of_batch = 0
                    for train_X, train_Y in input_data.get_train_data(self.batch_size):
                        # train_X, test_X, train_Y, test_Y = train_test_split(train_X, train_Y, test_size=0.2)
                        cost, predic, step, summ = self.fit(train_X, train_Y)
                        # print(predic)
                        epoch_loss += cost
                        self.train_writer.add_summary(summ, step)
                        number_of_batch += 1
                        # print('batch loss =>', cost)

                    _, epoch_step, summ = self.sess.run([self.epoch_optimizer,
                                                        self.epoch_step, self.test_merged_summary_op],
                                                       feed_dict={self.input: test_X, self.target: test_Y,
                                                                  self.keep_prob: 1})
                    self.test_writer.add_summary(summ, epoch_step)
                    print('epoch => %d, average loss => %f' % (
                    k + 1, epoch_loss / number_of_batch))

            else:
                print('-------------------------> mission already complete')
        except Exception as e:
            traceback.print_exc()
        finally:
            self.save_model()
            self.close()

if __name__ == '__main__':
    con = config.Config()
    data = data_processing.DataGenerator(con)
    model = NeuralNetwork(con)
    model.run_train(data)








