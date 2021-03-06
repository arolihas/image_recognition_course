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
        dense = self.create_fc(pool2)
        dropout = self.apply_dropout(dense, is_training)
        return tf.layers.dense(dropout, output_size=self.output_size, name='logits')

    def create_fc(self, pool2):
        hwc = pool2.shape.as_list()[1:]
        flattened_size = hwc[0] * hwc[1] * hwc[2] 
        pool2_flat = tf.reshape(pool2, [-1, flattened_size])
        return tf.layers.dense(pool2_flat, output_size=1024, activation=tf.nn.relu, name='dense')

    def apply_dropout(dense, is_training):
        return tf.layers.dropout(dense, rate=0.4, training=is_training)

    def run_model_setup(self, inputs, labels, is_training):
        logits = self.model_layers(inputs, is_training)
        
        # Predictions
        self.probs = tf.nn.softmax(logits, name='probs')
        self.predictions = tf.argmax(self.probs, axis=-1, name='predictions')
        class_labels = tf.argmax(labels, axis=-1)

        #Accuracy 
        is_correct = tf.cast(tf.equal(self.predictions, class_labels), tf.float32)
        self.accuracy = tf.reduce_mean(is_correct)

        #Training
        if self.is_training:
            labels_f = tf.cast(labels, tf.float32)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_f, logits=logits)
            self.loss = tf.reduce_mean(cross_entropy)

            adam = tf.train.AdamOptimizer()
            self.train_op = adam.minimize(self.loss, global_step=self.global_step)


