import tensorflow as tf

class MNISTModel(object):

    def __init__(self, input_dim, output_size):
        self.input_dim = input_dim
        self.output_size = output_size

    def model_layers(self, inputs, is_training):
        reshaped_inputs = tf.reshape(inputs, (-1, self.input_dim, self.input_dim, 1))
        conv1 = tf.layers.conv2d(reshaped_inputs, filters=32, kernel_size=[5,5], padding='same', activation=tf.nn.relu, name='conv1')
        pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2,2], strides=2, name='pool1')
        conv2 = tf.layers.conv2d(pool1, filters=64, kernel_size=[5,5], padding='same', activation=tf.nn.relu, name='conv2')
        pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2,2], strides=2, name='pool2')

