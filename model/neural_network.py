from __future__ import print_function
from __future__ import division
import tensorflow as tf
import data_processing
import config
import os


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

        self.input = tf.placeholder(tf.float32, [None, self.input_size], name='input')
        self.target = tf.placeholder(tf.float32, [None, self.n_classes], name='labels')
        self.keep_prob = tf.placeholder(tf.float32)

        self.optimizer = None
        self.loss = None
        self.acc = None
        self.out = None
        self.train_merged_summary_op = None
        self.test_merged_summary_op = None

        self.build_network_model()

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_config)

        self.train_writer = tf.summary.FileWriter(con.train_path_to_log, self.sess.graph)
        self.test_writer = tf.summary.FileWriter(con.test_path_to_log)
        self.model_path = con.model_path

        if os.path.exists(self.model_path):
            self.restore_model()
        else:
            self.sess.run(tf.global_variables_initializer())

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

    def _make_optimizer(self):
        with tf.name_scope('Optimizer'):
            optimize = tf.train.AdamOptimizer().minimize(self.loss)
        return optimize

    def _make_loss(self):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.out, labels=self.target))
        return loss

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
                weights = tf.Variable(tf.random_normal(dimension), name=layer_name)
                self.W[layer_name] = weights
                # self.variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = tf.Variable(tf.zeros(dimension[-1]), name=layer_name)
                self.b[layer_name] = biases
                # self.variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):
                if cnn:
                    preactivate = tf.nn.conv2d(input_tensor, weights,
                                               strides=self.struct[layer_name]['strides'],
                                               padding=self.struct[layer_name]['padding'])
                else:
                    preactivate = tf.matmul(input_tensor, weights) + biases

                if self.struct[layer_name].has_key('dropout') and self.struct[layer_name]['dropout']:
                    preactivate = tf.nn.dropout(preactivate, keep_prob=self.keep_prob)
                # tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activation')


            # tf.summary.histogram('activations', activations)
            return activations

    def build_network_model(self):

        def make_computing_graph(input):
            local_input = tf.reshape(input, [-1, 10, 10, 1])
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
                    local_input = tf.reshape(local_input, [-1, dimension[0]])
                    local_input = self.nn_layer(local_input, dimension, layer_name)
                elif 'out' in layer_name:
                    dimension = self.struct[layer_name]['struct']
                    local_input = self.nn_layer(local_input, dimension, layer_name, act=tf.identity)

            return local_input

        test_sum = []
        with tf.name_scope('Model'):
            self.out = make_computing_graph(self.input)

        with tf.name_scope('Loss'):
            self.loss = self._make_loss()
        tf.summary.scalar("loss", self.loss)

        with tf.name_scope('Optimizer'):
            self.optimizer = self._make_optimizer()

        with tf.name_scope('Accuracy'):
            correct = tf.equal(tf.argmax(self.out, 1), tf.argmax(self.target, 1))
            self.acc = tf.reduce_mean(tf.cast(correct, 'float'))
        test_sum.append(tf.summary.scalar("acc", self.acc))

        self.train_merged_summary_op = tf.summary.merge_all()
        self.test_merged_summary_op = tf.summary.merge(test_sum)

    def save_model(self):
        saver = tf.train.Saver(tf.global_variables())
        saver.save(self.sess, self.model_path)

    def restore_model(self):
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(self.sess, self.model_path)

    def close(self):
        if self.sess:
            self.sess.close()
        if self.test_writer:
            self.test_writer.close()
        if self.train_writer:
            self.train_writer.close()

    def fit(self, train_X, train_Y):
        feed_dict = {self.input: train_X, self.target: train_Y, self.keep_prob: self.my_keep_prob}
        ret, _, out, summary = self.sess.run([self.loss, self.optimizer, self.out, self.train_merged_summary_op], feed_dict=feed_dict)
        return ret, out, summary

    def get_accuracy(self):
        with tf.name_scope('Accuracy'):
            correct = tf.equal(tf.argmax(self.out, 1), tf.argmax(self.target, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        return accuracy

    def run_train(self, input_data):
        test_X, test_Y = input_data.get_test_data(self.batch_size)
        step = 0
        try:
            for k in range(self.epochs):
                epoch_loss = 0
                for batch_X, batch_Y in input_data.get_train_data(self.batch_size):
                    cost, predic, summ = self.fit(batch_X, batch_Y)
                    epoch_loss += cost
                    self.train_writer.add_summary(summ, step)
                    step += 1
                    # print('batch loss =>', cost)
                acc, summ = self.sess.run([self.acc, self.test_merged_summary_op],
                                          feed_dict={self.input: test_X, self.target: test_Y, self.keep_prob: 1})
                self.test_writer.add_summary(summ, k)
                print("accuracy", acc)
                print('epoch => %d, loss => %f' % (k+1, epoch_loss/self.batch_size))
        except Exception as e:
            raise e
        finally:
            self.save_model()
            self.close()


if __name__ == '__main__':
    con = config.Config()
    data = data_processing.DataGenerator(con)
    model = NeuralNetwork(con)
    model.run_train(data)








