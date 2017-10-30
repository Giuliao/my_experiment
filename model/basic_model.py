from __future__ import print_function
from __future__ import division
import tensorflow as tf



class basic:
    def __init__(self):
        pass

    def weight_variable(self, shape, stddev=0.1):
        """create a weight variable with appropriate initialization..
        :param shape: 
        :return: 
        """
        return tf.Variable(tf.truncated_normal(shape, stddev=stddev))

    def bias_variable(self, shape):
        """create a bias variable with appropriate initialization.
        :param shape: 
        :return: 
        """
        return tf.Variable(tf.constant(0.1, shape=shape))

    def variable_summaries(self, var):
        """
        :param var: 
        :return: 
        """
        with tf.name_scope("summaries"):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
