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
        self.activate_func = {'identity': tf.identity,
                              'relu': tf.nn.relu,
                              'sigmoid': tf.nn.sigmoid}

        self.layers_number = len(self.struct['structure'])

        # save weights by key mapping to value
        self.W = {}
        # save biases by key mapping to value
        self.b = {}

        self.batch_size = con.batch_size
        self.epochs = con.epochs
        self.input_size = con.input_size
        self.n_classes = con.n_classes
        self.my_keep_prob = con.keep_prob
        self.image_size = con.image_size
        self.channel = con.channel
        self.learning_rate = con.learning_rate

        self.alpha = con.alpha
        self.beta = con.beta
        self.reg_lambda = con.reg
        self.decay_steps = con.decay_steps
        self.decay_rate = con.decay_rate
        self.model_path = con.model_path

        self.acc_name_list = con.acc_name_list
        self.optimizer_name_list = con.optimizer_name_list
        self.loss_name_list = con.loss_name_list
        self.problem_type = con.problem_type_list
        self.class_number_list = con.class_number_list

        # self.path_to_save_predict = con.path_to_save_predict

        self.input = None
        self.target = None
        self.middle_out = []
        self.out = []
        self.optimizer = []
        self.loss = []
        self.acc = []
        self.summ = []
        self.keep_prob = None
        self.epoch_step = None
        self.global_step = None
        self.increment_epoch_step = None
        self.increment_global_step = None
        self.train_merged_summary_op = None
        self.test_merged_summary_op = None
        self.sess = None

    def init_session(self, con):
        tf_config = tf.ConfigProto()
        # false means fully occupied
        tf_config.gpu_options.allow_growth = False
        self.sess = tf.Session(config=tf_config)
        # self.sess = tf.Session()
        self.train_writer = tf.summary.FileWriter(con.train_path_to_log, self.sess.graph)
        self.test_writer = tf.summary.FileWriter(con.test_path_to_log)

        self.model_path = con.model_path
        # self.path_to_save_predict = con.path_to_save_predict

        if os.path.exists(self.model_path):
            self.restore_model()
            print('=> restore finished')
        else:
            self.sess.run(tf.global_variables_initializer())

    def _make_optimizer(self, loss, name):

        learning_rate = tf.train.exponential_decay(
            learning_rate=self.learning_rate,
            global_step=self.global_step,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            staircase=True
        )
        with tf.name_scope(name):
            optimize = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=self.global_step)
            # optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        return optimize, learning_rate

    def _make_loss(self, output, target, name, regression=True):

        def get_reg_loss(weight, biases):
            ret = tf.add_n([tf.nn.l2_loss(w) for w in weight.values()])
            ret += tf.add_n([tf.nn.l2_loss(b) for b in biases.values()])

            return ret

        with tf.name_scope(name):
            if regression:
                loss = tf.reduce_sum(tf.pow(output - target, 2))
            else:
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=target))

            if self.W:
                with tf.name_scope('weight_decay'):
                    weight_decay = get_reg_loss(self.W, self.b)
                    loss += self.reg_lambda * weight_decay

        return loss

    def get_accuracy(self, output, target, name, regression=True):
        with tf.name_scope(name):
            if regression:
                correct = (tf.abs(output - target) < 0.5)

            else:
                correct = tf.equal(tf.argmax(output, 1), tf.argmax(target, 1))

            acc = tf.reduce_mean(tf.cast(correct, 'float'))

        return acc

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

    def _make_weights_biases(self, dimension, name, mean=0, stddev=0.1, weight_decay=False):
        """customized weights and biases constructor
        :param dimension: 
        :param name: 
        :param mean: 
        :param stddev: 
        :return: weights and biases
        """
        weights = tf.Variable(tf.random_normal(dimension, stddev=stddev), name='weights')
        # self.variable_summaries(weights)
        if weight_decay:
            self.W[name + 'weights'] = weights
        biases = tf.Variable(tf.random_normal(dimension[-1:], mean=mean, stddev=stddev), name='biases')
        # self.variable_summaries(biases)
        if weight_decay:
            self.b[name + 'biases'] = biases

        return weights, biases

    def fully_connected(self, local_struct, input_tensor, dimension, layer_name, act=tf.nn.relu):
        """customized fully connected network constructor
        :param local_struct: a struct includes the parameters
        :param input_tensor: 
        :param dimension: 
        :param layer_name: the name of layer
        :param act: activate function
        :return: a tensor
        """
        with tf.name_scope(layer_name) as ns:

            if 'weight_decay' in local_struct[layer_name]:
                weight_decay = local_struct[layer_name]['weight_decay']
            else:
                weight_decay = False

            with tf.name_scope('Wx_plus_b'):
                weights, biases = self._make_weights_biases(dimension=dimension,
                                                            name=ns,
                                                            weight_decay=weight_decay)

                preactivate = tf.matmul(input_tensor, weights) + biases

            if 'dropout' in local_struct[layer_name] and local_struct[layer_name]['dropout']:
                preactivate = tf.nn.dropout(preactivate, keep_prob=self.keep_prob, name='dropout')
                # tf.summary.histogram('pre_activations', preactivate)

            activations = act(preactivate, name='activation')
            # tf.summary.histogram('activations', activations)

            return activations

    def conv2d(self, local_struct, input_tensor, dimension, layer_name, act=tf.nn.relu):
        """customized 2d convolution network constructor
        :param local_struct: 
            a struct includes the parameters
        :param input_tensor: 
        :param dimension: 
        :param layer_name: 
        :param act: 
            activate function
        :return: 
            a tensor
        """

        with tf.name_scope(layer_name) as ns:
            if 'weight_decay' in local_struct[layer_name]:
                weight_decay = local_struct[layer_name]['weight_decay']
            else:
                weight_decay = False

            with tf.name_scope('Wx_plus_b'):
                kernel, biases = self._make_weights_biases(dimension, ns, weight_decay=weight_decay)
                preactivate = tf.nn.conv2d(
                    input_tensor,
                    kernel,
                    strides=local_struct[layer_name]['strides'],
                    padding=local_struct[layer_name]['padding']
                )
            if 'dropout' in local_struct[layer_name] and local_struct[layer_name]['dropout']:
                preactivate = tf.nn.dropout(preactivate, keep_prob=self.keep_prob, name='dropout')
                # tf.summary.histogram('pre_activations', preactivate)

            activations = act(preactivate, name='activation')
            # tf.summary.histogram('activations', activations)
            if "visualization" in local_struct[layer_name] and local_struct[layer_name]["visualization"]:
                # reference: https://stackoverflow.com/questions/35759220/how-to-visualize-learned-filters-on-tensorflow
                with tf.variable_scope('visualization'):
                    # # scale weights to [0 1], type is still float
                    x_min = tf.reduce_min(kernel)
                    x_max = tf.reduce_max(kernel)
                    kernel_0_to_1 = (kernel - x_min) / (x_max - x_min)
                    kernel_transposed = tf.transpose(kernel_0_to_1, [3, 0, 1, 2])
                    tf.summary.image(layer_name+'/filters', kernel_transposed)

            return activations

    def max_pool(self, local_struct, input_tensor, dimension, layer_name):
        """customized max pool constructor
        :param local_struct:
            a struct includes the parameters
        :param input_tensor: 
        :param dimension: 
        :param layer_name: 
        :return: 
        """
        with tf.name_scope(layer_name):
            activations = tf.nn.max_pool(
                input_tensor,
                ksize=dimension,
                strides=local_struct[layer_name]['strides'],
                padding=local_struct[layer_name]['padding']
            )

            if 'dropout' in local_struct[layer_name] and local_struct[layer_name]['dropout']:
                activations = tf.nn.dropout(activations, keep_prob=self.keep_prob, name='dropout')
                # tf.summary.histogram('activations', activations)
        return activations

    def avg_pool(self, local_struct, input_tensor, dimension, layer_name):
        """customized max pool constructor
        :param local_struct:
            a struct includes the parameters
        :param input_tensor: 
        :param dimension: 
        :param layer_name: 
        :return: 
        """
        with tf.name_scope(layer_name):
            activations = tf.nn.avg_pool(
                input_tensor,
                ksize=dimension,
                strides=local_struct[layer_name]['strides'],
                padding=local_struct[layer_name]['padding']
            )

            if 'dropout' in local_struct[layer_name] and local_struct[layer_name]['dropout']:
                activations = tf.nn.dropout(activations, keep_prob=self.keep_prob, name='dropout')
                # tf.summary.histogram('activations', activations)
        return activations

    def local_response_normalization(self, local_struct, input_tensor, layer_name):
        with tf.name_scope(layer_name):
            activations = tf.nn.lrn(
                input_tensor,
                depth_radius=local_struct[layer_name]['depth_radius'],
                bias=local_struct[layer_name]['bias'],
                alpha=local_struct[layer_name]['alpha'],
                beta=local_struct[layer_name]['beta'],
                name=layer_name
            )
        return activations

    def _make_computing_graph(self, local_input, local_struct):
        """make computing graph by parsing the local_struct recursively.
        
         read the definition of a network from a json file, 
        'inception', constrcut network parallelly.
        'cnn', construct a cnn layer.
        'pool', construct a pool layer.
        
        :param local_input: 
        :param local_struct: 
        :return: 
            a tensor
        """

        for i in range(len(local_struct['structure'])):
            # a struct contains layer name for reading layers by order
            layer_name = local_struct['structure'][i]
            if 'inception' in layer_name:
                with tf.name_scope(layer_name):
                    if "image_size" in local_struct[layer_name]:
                        local_image_size = local_struct[layer_name]['image_size']
                        local_channel = local_struct[layer_name]['channel']
                        local_input = tf.reshape(local_input, [-1, local_image_size, local_image_size, local_channel])

                    local_input_total = []
                    for key in local_struct[layer_name]["structure"]:
                        local_input1 = self._make_computing_graph(
                            local_input,
                            local_struct[layer_name][key]
                        )

                        if local_input1 is not None:
                            # dimension multiply,
                            # https://stackoverflow.com/questions/44275212/how-to-multiply-dimensions-of-a-tensor
                            new_shape = tf.reduce_prod(local_input1.shape[1:])
                            local_input_total.append(tf.reshape(local_input1, [-1, new_shape]))

                    if len(local_input_total) > 0:
                        # axis=1, concatenate the horizon!!!
                        local_input = tf.concat(local_input_total, axis=1, name=layer_name + 'concate_op')
                    else:
                        local_input = None

            # cnn layer
            elif 'conv' in layer_name:
                dimension = local_struct[layer_name]['struct']
                # reshape and alignment
                if len(local_input.shape) != 4 or local_input.shape[-1] != dimension[-2]:
                    # [-1, filter_size, filter_size, channel_size]
                    # default [-1, self.image_size, self.image_size, self.channel]
                    local_input = tf.reshape(local_input, [-1, self.image_size, self.image_size, self.channel])

                dimension = local_struct[layer_name]['struct']

                local_input = self.conv2d(
                    local_struct,
                    local_input,
                    dimension,
                    layer_name,
                )

            # max pool layer
            elif 'max_pool' in layer_name:
                if len(local_input.shape) != 4:
                    local_image_size = local_struct[layer_name]['image_size']
                    local_channel = local_struct[layer_name]['channel']
                    local_input = tf.reshape(local_input, [-1, local_image_size, local_image_size, local_channel])

                dimension = local_struct[layer_name]['ksize']
                local_input = self.max_pool(
                    local_struct,
                    local_input,
                    dimension,
                    layer_name,
                )

            # avg pool layer
            elif 'avg_pool' in layer_name:
                if len(local_input.shape) != 4:
                    local_image_size = local_struct[layer_name]['image_size']
                    local_channel = local_struct[layer_name]['channel']
                    local_input = tf.reshape(local_input, [-1, local_image_size, local_image_size, local_channel])

                dimension = local_struct[layer_name]['ksize']
                local_input = self.avg_pool(
                    local_struct,
                    local_input,
                    dimension,
                    layer_name,
                )

            # fully connected layer
            elif 'full' in layer_name:
                dimension = local_struct[layer_name]['struct']
                # reshape and alignment
                if local_input.shape[1] != dimension[0]:
                    local_input = tf.reshape(local_input, [-1, dimension[0]])

                local_input = self.fully_connected(
                    local_struct,
                    local_input,
                    dimension,
                    layer_name,
                )

            # local response normalization layer
            elif 'lrn' in layer_name:
                local_input = self.local_response_normalization(
                    local_struct,
                    local_input,
                    layer_name,
                )

            # out layer, actually the same as fully connected layer
            elif 'out' in layer_name:
                dimension = local_struct[layer_name]['struct']
                local_input = self.fully_connected(
                    local_struct,
                    local_input,
                    dimension,
                    layer_name,
                    act=self.activate_func[local_struct[layer_name]['act']]
                )
                if 'tear' in layer_name:

                    idx = 0
                    for i in range(len(self.optimizer_name_list)):
                        self.out.append(tf.slice(local_input, [0, idx], [-1, self.class_number_list[i]]))
                        idx += self.class_number_list[i]
                else:
                    self.out.append(local_input)

                local_input = None

            if 'save_output' in local_struct[layer_name] and \
                    local_struct[layer_name]['save_output']:
                self.middle_out.append(local_input)

        return local_input

    def build_network_model(self):
        """ build the network and initialize variables
        :return: 
        """

        self.input = tf.placeholder(tf.float32, [None, self.input_size], name='input')
        self.target = tf.placeholder(tf.float32, [None, self.n_classes], name='labels')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.epoch_step = tf.Variable(0, trainable=False, name="epoch_step")
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        with tf.name_scope('Model'):
            self._make_computing_graph(self.input, self.struct)

        with tf.name_scope('Loss'):
            # here 'for' step need to optimize, when self.out[index].shape[1] != 1
            idx = 0  # idx may solve the problem above
            for index in range(len(self.out)):
                self.loss.append(self._make_loss(self.out[index],
                                                 tf.slice(self.target, [0, idx], [-1, self.out[index].shape[1]]),
                                                 self.loss_name_list[index],
                                                 regression=self.problem_type[index]))

                # reference:
                # https://stackoverflow.com/questions/40666316/how-to-get-tensorflow-tensor-dimensions-shape-as-int-values
                # idx += self.out[index].get_shape().as_list()[1]
                idx += self.class_number_list[index]

        loss_summ = []
        for index, name in enumerate(self.loss_name_list):
            loss_summ.append(tf.summary.scalar(name, self.loss[index]))
        self.summ.extend(loss_summ)

        # learning_rate_name = ['learning_rate_k']
        with tf.name_scope('Optimizer'):
            for index in range(len(self.loss)):
                optimizer, _ = self._make_optimizer(self.loss[index],
                                                    self.optimizer_name_list[index])
                self.optimizer.append(optimizer)

        with tf.name_scope('Accuracy'):
            # here 'for' step need to optimize, when self.out[index].shape != 1
            idx = 0  # idx may solve the problem above
            for index in range(len(self.out)):
                self.acc.append(self.get_accuracy(self.out[index],
                                                  tf.slice(self.target, [0, idx], [-1, self.out[index].shape[1]]),
                                                  self.acc_name_list[index],
                                                  regression=self.problem_type[index]))

                idx += self.class_number_list[index]

        acc_summ = []
        for index, name in enumerate(self.acc_name_list):
            acc_summ.append(tf.summary.scalar(name, self.acc[index]))
        self.summ.extend(acc_summ)

        self.increment_global_step = tf.assign_add(self.global_step, 1, name='increment_global_step')
        self.increment_epoch_step = tf.assign_add(self.epoch_step, 1, name='increment_epoch_step')

        self.train_merged_summary_op = tf.summary.merge_all()
        self.test_merged_summary_op = tf.summary.merge(self.summ)

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

    def get_W(self):
        return self.sess.run(self.W)

    def get_b(self):
        return self.sess.run(self.b)

    def predict_eval_data(self, input_data):
        pass

    def run_train(self, input_data):
        test_X, test_Y = input_data.get_test_data()
        ss = self.sess.run(self.epoch_step)
        step = self.sess.run(self.global_step)
        train_ops = [self.loss, self.train_merged_summary_op, self.target, self.out, self.optimizer,
                     self.increment_global_step]
        test_ops = [self.test_merged_summary_op]
        try:
            for k in range(ss, self.epochs):
                epoch_loss = [0.0] * len(self.loss)
                epoch_acc = [0.0] * len(self.loss)
                test_acc = [0.0] * len(self.loss)
                number_of_batch = 0
                for train_X, train_Y in input_data.get_train_data(self.batch_size):
                    # train_X, test_X, train_Y, test_Y = train_test_split(train_X, train_Y, test_size=0.2)
                    cost, summ, target, predict = self.fit(train_X=train_X, train_Y=train_Y,
                                                           keep_prob=self.my_keep_prob, ops=train_ops)[:4]

                    for index in range(len(cost)):
                        epoch_loss[index] += cost[index]

                    for index in range(len(self.acc)):
                        epoch_acc[index] += self.fit(ops=self.acc[index], train_X=train_X, train_Y=train_Y,
                                                     keep_prob=1.0)

                    # print(predic) # debug
                    # print(len(predic))
                    self.train_writer.add_summary(summ, step)
                    step += 1
                    number_of_batch += 1

                for index in range(len(self.acc)):
                    test_acc[index] = self.fit(ops=self.acc[index], train_X=test_X, train_Y=test_Y, keep_prob=1.0)

                summ = self.fit(ops=test_ops, train_X=test_X, train_Y=test_Y, keep_prob=1.0)[0]
                self.test_writer.add_summary(summ, k)
                self.sess.run(self.increment_epoch_step)
                print('epoch => ', k + 1)
                for index in range(len(self.loss)):
                    print('average %s => %f' % (self.loss_name_list[index], epoch_loss[index] / number_of_batch))
                    print(
                        'average %s => %.2f%%' % (self.acc_name_list[index], epoch_acc[index] / number_of_batch * 100))
                    print('test %s => %.2f%%' % (self.acc_name_list[index], test_acc[index] * 100))


                # drawing confusion matrix
                # if (k + 1) % 10 == 0:
                #     # print('-' * 75)
                #     feed_dict = {self.input: test_X, self.target: test_Y, self.keep_prob: 1}
                #     epoch_out = self.sess.run(self.out, feed_dict=feed_dict)
                #     input_data.get_confusion_matrix(epoch_out)
                #     # print('-' * 75)

                print()

                del epoch_loss
                del epoch_acc
                del test_acc
                del number_of_batch

            print('=> mission complete')
        except Exception as e:
            traceback.print_exc()
            self.close()
            self.sess = None
            print('=> session closed')
        finally:
            if self.sess is not None:
                self.save_model()
                print('=> save finished')


if __name__ == '__main__':
    con = config.Config()
    data = data_processing.DataGenerator(con)
    model = NeuralNetwork(con)
    model.run_train(data)
