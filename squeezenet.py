import tensorflow as tf

class SqueezeNetModel(object):
    
    def __init__(self, original_dim, resize_dim, output_size):
        self.original_dim = original_dim
        self.output_size = output_size
        self.resize_dim = resize_dim

    def image_preprocessing(self, data, is_training):
        reshaped_image = tf.reshape(data, [3, self.original_dim, self.original_dim])
        transposed = tf.cast(tf.transpose(reshaped_image, [1,2,0]), tf.float32)
        if is_training:
            updated = self.random_crop_and_flip(transposed)
        else:
            updated = tf.image.resize_image_with_crop_or_pad(transposed, self.resize_dim, self.resize_dim)
        return tf.image.per_image_standardization(updated)

    def random_crop_and_flip(self, image):
        cropped = tf.random_crop(image, [self.resize_dim, self.resize_dim, 3]) 
        return tf.image.flip_left_right(cropped)

    def conv_layer(self, inputs, filters, kernel_size, name):
        return tf.layers.conv2d(inputs, filters, kernel_size, activation=tf.nn.relu, padding='same', name=name)

    def pool_layer(self, inputs, name):
        return tf.layers.max_pooling_2d(inputs, [2,2], 2, name)

    def fire_module(self, inputs, squeeze_depth, expand_depth, name):
        with tf.variable_scope(name):
            squeezed = self.conv_layer(inputs, squeeze_depth, [1,1], 'squeeze')
            expand1x1 = self.conv_layer(squeezed, expand_depth, [1,1], 'expand1x1')
            expand3x3 = self.conv_layer(squeezed, expand_depth, [3,3], 'expand3x3')
            return tf.concat([expand1x1, expand3x3], axis=-1)

    def multi_fire_module(self, layer, params_list):
        for squeeze_depth, expand_depth, name in params_list:
            layer = self.fire_module(layer, squeeze_depth, expand_depth, name)
        return layer

    def model_layers(self, inputs, is_training):
        conv1 = self.conv_layer(inputs, 64, [3,3], 'conv1')
        pool1 = self.pool_layer(conv1, 'pool1')
        
        fire_params1 = [(32, 64, 'fire1'), (32, 64, 'fire2')]
        multi_fire1 = self.multi_fire_module(pool1, fire_params1)
        pool2 = self.pool_layer(multi_fire1, 'pool2')
        
        fire_params2 = [(32, 128, 'fire3'), (32, 128, 'fire4')]
        multi_fire2 = self.multi_fire_module(pool2 , fire_params2)
        dropout1 = tf.layers.dropout(multi_fire2, rate=0.5, training=is_training)
        
        final = self.conv_layer(dropout1, self.output_size, [1,1], 'final')
        avg_pool1 = tf.layer.average_pooling2d(final, [final.shape[1], final.shape[2]], 1) 
        return tf.layers.flatten(avg_pool1, name='logits')

    def run_model_setup(self, inputs, labels):
        logits = self.model_layers(inputs, is_training)
        self.probs = tf.nn.softmax(logits, name='probs')
        self.predictions = tf.argmax(self.probs, axis=-1, name='predictions')
        
        is_correct = tf.cast(tf.equal(tf.cast(self.predictions, tf.int32), labels), tf.float32)
        self.accuracy = tf.reduce_mean(is_correct)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
        
        self.loss = tf.reduce_mnean(cross_entropy)
        adam = tf.train.AdamOptimizer()
        self.train_op = adam.minimize(self.loss, global_step=self.global_step)

