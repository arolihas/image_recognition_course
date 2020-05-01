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


