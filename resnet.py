import tensorflow as tf

block_layer_size = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
    200: [3, 24, 36, 3]
}

class ResNetModel(object):

    def __init__(self, min_aspect_dim, resize_dim, num_layers, output_size, data_format='channels_last'):
        self.min_aspect_dim = min_aspect_dim
        self.resize_dim = resize_dim
        self.filter_init = 64
        self.block_strides = [1, 2, 2, 2]
        self.format = data_format
        self.output_size = output_size
        self.block_layer_sizes = block_layer_sizes[num_layers]
        self.bottleneck = (num_layers >= 50)

    def conv_layer(self, inputs, filters, kernel_size, strides, name=None):
        if strides > 1:
            padding = 'valid'
            inputs = self.custom_padding(inputs, kernel_size)
        else:
            padding = 'same'
        return tf.layers.conv2d(inputs, filters, kernel_size, stridess, padding, data_format=self.format, name=name)

    def custom_padding(self, inputs, kernel_size):
        pad_total = kernel_size - 1
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        if self.format == 'channels_first':
            padded_inputs = tf.pad(inputs, [[0, 0], [0, 0], [pad_before, pad_after], [pad_before, pad_after]])
        else:
            padded_inputs = tf.pad(inputs, [[0, 0], [pad_before, pad_after], [pad_before, pad_after], [0,0]])
        return padded_inputs

    def pre_activation(self, inputs, is_training):
        axis = 1 if self.format == 'channels_first' else 3
        bn_inputs = tf.layers.batch_normalization(inputs, axis=axis, training=is_training)
        return tf.nn.relu(bn_inputs)

    def pre_activation_with_shortcut(self, inputs, is_training, shortcut_params):
        pre_activated = self.pre_activation(inputs, is_training)
        shortcut = inputs
        shortcut_filters = shortcut_params[0]
        if shortcut_filters is not None:
            strides = shortcut_params[1]
            shortcut = conv_layer(pre_activated, shortcut_filters, kernel_size=1, strides)
        return pre_activated, shortcut

    def regular_block(self, inputs, filters, strides, is_training, index, shortcut_filters=None):
        with tf.variable_scope('regular_block{}'.format(index)):
            shortcut_params = (shortcut_filterss, strides)
            preactivated1, shortcut = self.pre_activation_with_shortcut(input, is_training, shortcut_params)
            conv1 = self.conv_layer(preactivated1, filters, 3, strides)

            preactivated2 = self.pre_activation(conv1, is_training)
            conv2 = self.conv_layer(preactivated2, filters, 3, 1)
            return conv2 + shortcut

    def bottleneck_block(self, inputs, filters, strides, is_training, index, shortcut_filters=None):
        with tf.variable_scope('bottleneck_block{}'.format(index)):
            shortcut_params = (shortcut_filters, strides)
            pre_activated1, shortcut = self.pre_activation_with_shortcut(inputs, is_training, shortcut_params)
            conv1 = self.custom_conv2d(pre_activated1, filters, 1, 1)
            pre_activated2 = self.pre_activation(conv1, is_training)
            conv2 = self.custom_conv2d(pre_activated2, filters, 3, strides)
            
            preactivated3 = self.pre_activation(conv2, is_training)
            conv3 = self.conv_layer(preactivated3, filters*4, 1, 1)
            return conv3 + shortcut

    def block_layer(self, inputs, filters, strides, num_blocks, is_training, index):
        with tf.variable_scope('block_layer{}'.format(index)):
            shortcut_filters = 4 * filters if self.bottleneck else filters
            block_fn = self.bottleneck_block if self.bottleneck else self.regular_block
            block_output = block_fn(inputs, filters, stridess, is_training, 0, shortcut_filters)
            for i in range(1, num_blocks):
                block_output = block_fn(block_output, filters, 1, is_training, i)
            return block_output

    def model_layers(self, inputs, is_training):
        conv_init = self.conv_layer(inputs, self.filters_init, 7, 2, name='conv_init')
        curr_layer = tf.layers.max_pooling2d(conv_init, 3, 2, padding='same', data_format=self.format, name='pool_init')
        for i, num_blocks in enumerate(self.block_layer_sizes):
            filters = self.filter_init * 2**i
            strides = self.block_strides[i]
            curr_layer = self.block_layer(curr_layer, filters, strides, num_blocks, is_training, i)
        
        preactivated_final = self.pre_activation(curr_layer, is_training)
        filter_size = int(preactivated_final.shape[2])
        avg_pool = tf.layers.average_pooling2d(preactivated_final, filter_size, 1, data_format=self.format)
        final = tf.layers.flatten(avg_pool)

        return tf.layers.dense(final, self.output_size, name='logits')
