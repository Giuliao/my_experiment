from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import data_processing
from model import config
import traceback
import pandas as pd


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
        self.epoch_step = None
        self.increment_epoch_step = None
        self.global_step = None
        self.increment_global_step = None
        self.optimizer = None
        self.optimizer1 = None
        self.loss = None
        self.loss1 = None
        self.acc = None
        self.out = None
        self.train_merged_summary_op = None
        self.test_merged_summary_op = None
        self.sess = None

    def init_session(self, con):
        # tf_config = tf.ConfigProto()
        # false meams fully occupied
        # tf_config.gpu_options.allow_growth = False
        # self.sess = tf.Session(config=tf_config)
        self.sess = tf.Session()
        self.train_writer = tf.summary.FileWriter(con.train_path_to_log, self.sess.graph)
        self.test_writer = tf.summary.FileWriter(con.test_path_to_log)

        self.model_path = con.model_path
        # self.path_to_save_predict = con.path_to_save_predict

        if os.path.exists(self.model_path):
            self.restore_model()
            print('=> restore finished')
        else:
            self.sess.run(tf.global_variables_initializer())

    def _make_optimizer(self, loss):
        local_step = tf.Variable(0, trainable=False, name='local_step')
        with tf.name_scope('Optimizer'):
            learning_rate = tf.train.exponential_decay(
                learning_rate=self.learning_rate,
                global_step=self.global_step,
                decay_steps=self.decay_steps,
                decay_rate=self.decay_rate,
                staircase=True
            )
            optimize = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=local_step)
        return optimize

    def _make_loss(self):
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.out, labels=self.target))
        # loss3 = self.reg_lambda*get_reg_loss(self.W, self.b)
        # loss = self.alpha*loss1 + self.beta*loss2 + loss3*self.reg_lambda
        def get_reg_loss(weight, biases):
            ret = tf.add_n([tf.nn.l2_loss(w) for w in weight.values()])
            ret = ret + tf.add_n([tf.nn.l2_loss(b) for b in biases.values()])
            return ret

        with tf.name_scope('loss1'):
            loss1 = tf.reduce_sum(tf.pow(self.out[:, 0] - self.target[:, 0], 2)) + get_reg_loss(self.W, self.b)
        with tf.name_scope('loss2'):
            loss2 = tf.reduce_sum(tf.pow(self.out[:, 1] - self.target[:, 1], 2)) + get_reg_loss(self.W, self.b)

        return loss1, loss2

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

        test_summ = []

        self.input = tf.placeholder(tf.float32, [None, self.input_size], name='input')
        self.target = tf.placeholder(tf.float32, [None, self.n_classes], name='labels')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.epoch_step = tf.Variable(0, trainable=False, name="epoch_step")
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        with tf.name_scope('Model'):
            self.out = make_computing_graph(self.input)

        with tf.name_scope('Loss'):
            self.loss, self.loss1 = self._make_loss()
        loss_summ = tf.summary.scalar("loss_r", self.loss)
        loss1_summ = tf.summary.scalar("loss_s", self.loss1)

        test_summ.append(loss1_summ)
        test_summ.append(loss_summ)

        with tf.name_scope('Optimizer'):
            self.optimizer = self._make_optimizer(self.loss)

        with tf.name_scope('Optimizer1'):
            self.optimizer1 = self._make_optimizer(self.loss1)

        with tf.name_scope('Accuracy'):
            self.acc = self.get_accuracy()
        acc_r_summ = tf.summary.scalar("acc_r", self.acc[0])
        acc_s_summ = tf.summary.scalar("acc_s", self.acc[1])
        test_summ.append(acc_r_summ)
        test_summ.append(acc_s_summ)

        self.increment_global_step = tf.assign_add(self.global_step, 1, name='increment_global_step')
        self.increment_epoch_step = tf.assign_add(self.epoch_step, 1, name='increment_global_step')

        self.train_merged_summary_op = tf.summary.merge(test_summ)
        self.test_merged_summary_op = tf.summary.merge(test_summ)

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

    def fit(self, **kwargs):
        train_X = kwargs.get('train_X', None)
        train_Y = kwargs.get('train_Y', None)
        keep_prob = kwargs.get('keep_prob', None)
        ops = kwargs.get('ops', None)
        feed_dict = {self.input: train_X, self.target: train_Y, self.keep_prob: keep_prob}
        ret_list = self.sess.run(ops, feed_dict=feed_dict)
        return ret_list

    def get_accuracy(self):
        # correct = tf.equal(tf.argmax(self.out, 1), tf.argmax(self.target, 1))
        correct_r = tf.abs(self.out[:, 0] - self.target[:, 0]) < 0.5
        acc_r = tf.reduce_mean(tf.cast(correct_r, 'float'))
        correct_s = tf.abs(self.out[:, 1] - self.target[:, 1]) < 0.5
        acc_s = tf.reduce_mean(tf.cast(correct_s, 'float'))
        return acc_r, acc_s

    def get_W(self):
        return self.sess.run(self.W)

    def get_b(self):
        return self.sess.run(self.b)

    def predict_eval_data(self, input_data):
        test_X, test_Y = input_data.get_test_data()
        acc_r, acc_s = self.get_accuracy()
        # remember when there is no [] in ops then the output is not a list!!!
        predict, acc_r, acc_s = self.sess.run([self.out, acc_r, acc_s],
                                              feed_dict={self.target: test_Y, self.input: test_X, self.keep_prob: 1})
        # pd.DataFrame(predict[0]).reset_index(drop=True).join(test_Y.reset_index(drop=True)).to_csv('predict.csv')
        for k in range(len(predict)):
            print(predict[k], '=>', test_Y.iloc[k, :])
        print('=> acc_r: %.2f%%, acc_s: %.2f%%' % (acc_r * 100, acc_s * 100))

    def run_train(self, input_data):
        test_X, test_Y = input_data.get_test_data()
        ss = self.sess.run(self.epoch_step)
        step = self.sess.run(self.global_step)
        train_ops = [self.loss, self.loss1, self.out, self.train_merged_summary_op, self.optimizer, self.optimizer1,
                     self.increment_global_step]
        test_ops = [self.test_merged_summary_op]
        try:
            for k in range(ss, self.epochs):
                epoch_loss = 0
                epoch_loss1 = 0
                epoch_acc_r = 0
                epoch_acc_s = 0
                number_of_batch = 0
                for train_X, train_Y in input_data.get_train_data(self.batch_size):
                    # train_X, test_X, train_Y, test_Y = train_test_split(train_X, train_Y, test_size=0.2)
                    cost, cost1, predic, summ = self.fit(train_X=train_X, train_Y=train_Y, keep_prob=self.my_keep_prob,
                                                         ops=train_ops)[:4]
                    epoch_acc_r += self.fit(ops=self.acc[0], train_X=train_X, train_Y=train_Y, keep_prob=1.0)
                    epoch_acc_s += self.fit(ops=self.acc[1], train_X=train_X, train_Y=train_Y, keep_prob=1.0)
                    # print(predic)
                    epoch_loss += cost
                    epoch_loss1 += cost1
                    self.train_writer.add_summary(summ, step)
                    self.global_step += 1
                    step += 1
                    number_of_batch += 1

                # summ = self.fit(ops=test_ops, train_X=test_X, train_Y=test_Y, keep_prob=1.0)[0]
                test_acc_r = self.fit(ops=self.acc[0], train_X=test_X, train_Y=test_Y, keep_prob=1.0)
                test_acc_s = self.fit(ops=self.acc[1], train_X=test_X, train_Y=test_Y, keep_prob=1.0)
                # self.test_writer.add_summary(summ, k)
                self.sess.run(self.increment_epoch_step)
                print('epoch => ', k + 1)
                print('average loss_r => %f, average loss_s => %f' % (
                epoch_loss / number_of_batch, epoch_loss1 / number_of_batch))
                print('average acc_r=> %.2f%%, average acc_s => %.2f%%' % (
                epoch_acc_r / number_of_batch * 100, epoch_acc_s / number_of_batch * 100))
                print('test acc_r=> %.2f%%, test acc_s => %.2f%%' % (test_acc_r * 100, test_acc_s * 100))
                # print(predic)
                # print(train_Y)

            print('=> mission complete')

        except Exception as e:
            traceback.print_exc()
            self.close()
            print('=> session closed')
        finally:
            self.save_model()
            print('=> save finished')


if __name__ == '__main__':
    con = config.Config()
    data = data_processing.DataGenerator(con)
    model = NeuralNetwork(con)
    model.run_train(data)
