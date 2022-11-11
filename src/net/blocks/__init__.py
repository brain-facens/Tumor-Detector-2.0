# libraries
from tensorflow._api.v2.nn import conv2d,\
                                  max_pool2d,\
                                  conv2d_transpose,\
                                  batch_normalization,\
                                  relu,\
                                  sigmoid

from tensorflow._api.v2.math import add,\
                                    multiply

import tensorflow as tf


# model parameters
# - batch normalization
MEAN = 0.0
VARIANCE = 0.01
EPSILON = 0.01
OFFSET = None
SCALE = None
# - weights and bias
STDDEV = 0.001


# down-sampling block
class DownSamplingBlock(tf.Module):
    
    def __init__(self, input_channels: int, output_channels: int, name: str = 'DownSamplingBlock') -> None:
        
        # input and output channels
        self.__f_input = input_channels
        self.__f_output = output_channels
        super().__init__(name + '_' + str(output_channels))
        
        # initializing weights and bias
        self.__w1 = tf.Variable(
            initial_value=tf.random.normal(shape=[3, 3, self.__f_input, self.__f_output], stddev=STDDEV),
            name='w1'
        )
        self.__b1 = tf.Variable(
            initial_value=tf.random.normal(shape=[self.__f_output], stddev=STDDEV),
            name='b1'
        )
        
        self.__w2 = tf.Variable(
            initial_value=tf.random.normal(shape=[3, 3, self.__f_output, self.__f_output], stddev=STDDEV),
            name='w2'
        )
        self.__b2 = tf.Variable(
            initial_value=tf.random.normal(shape=[self.__f_output], stddev=STDDEV),
            name='b2'
        )
    
    def __call__(self, x) -> any:
        
        # forward propagation
        x = conv2d(input=x, filters=self.__w1, strides=1, padding='SAME') + self.__b1
        x = batch_normalization(x=x, mean=MEAN, variance=VARIANCE, offset=OFFSET, scale=SCALE, variance_epsilon=EPSILON)
        x = relu(features=x)
        
        x = conv2d(input=x, filters=self.__w2, strides=1, padding='SAME') + self.__b2
        x = batch_normalization(x=x, mean=MEAN, variance=VARIANCE, offset=OFFSET, scale=SCALE, variance_epsilon=EPSILON)
        x = relu(features=x)

        down = max_pool2d(input=x, ksize=2, strides=2, padding='VALID')
        
        return x, down
    
# up-sampling block
class UpSamplingBlock(tf.Module):
    
    def __init__(self, input_channels: int, output_channels: int, name: str = 'UpSamplingBlock') -> None:
        
        # input and output channels
        self.__f_input = input_channels
        self.__f_output = output_channels
        super().__init__(name + '_' + str(output_channels))
        
        # initializing weights and bias
        self.__w1 = tf.Variable(
            initial_value=tf.random.normal(shape=[3, 3, self.__f_input, self.__f_output], stddev=STDDEV),
            name='w1'
        )
        self.__b1 = tf.Variable(
            initial_value=tf.random.normal(shape=[self.__f_output], stddev=STDDEV),
            name='b1'
        )
        
        self.__w2 = tf.Variable(
            initial_value=tf.random.normal(shape=[3, 3, self.__f_output, self.__f_output], stddev=STDDEV),
            name='w2'
        )
        self.__b2 = tf.Variable(
            initial_value=tf.random.normal(shape=[self.__f_output], stddev=STDDEV),
            name='b2'
        )
        
        self.__w3 = tf.Variable(
            initial_value=tf.random.normal(shape=[3, 3, self.__f_output, self.__f_output], stddev=STDDEV),
            name='w3'
        )
        self.__b3 = tf.Variable(
            initial_value=tf.random.normal(shape=[self.__f_output], stddev=STDDEV),
            name='b3'
        )
    
    def __call__(self, x) -> any:
        
        # forward propagation
        x = conv2d(input=x, filters=self.__w1, strides=1, padding='SAME') + self.__b1
        x = batch_normalization(x=x, mean=MEAN, variance=VARIANCE, offset=OFFSET, scale=SCALE, variance_epsilon=EPSILON)
        x = relu(features=x)
        
        x = conv2d(input=x, filters=self.__w2, strides=1, padding='SAME') + self.__b2
        x = batch_normalization(x=x, mean=MEAN, variance=VARIANCE, offset=OFFSET, scale=SCALE, variance_epsilon=EPSILON)
        x = relu(features=x)

        up = conv2d_transpose(input=x, filters=self.__w3, output_shape=[int(x.shape[1]*2), int(x.shape[2]*2)], strides=2, padding='SAME') + self.__b3
        
        return x, up

# attention gate block
class AttentionGate(tf.Module):
    
    def __init__(self, input_filter: int, output_filter: int, name: str = 'AttentionGate') -> None:
        super().__init__(name)
        
        # initializing weights and bias
        self.__w1_x = tf.Variable(
            initial_value=tf.random.normal(shape=[1, 1, input_filter, input_filter], stddev=STDDEV),
            name='w1'
        )
        self.__b1_x = tf.Variable(
            initial_value=tf.random.normal(shape=[input_filter], stddev=STDDEV),
            name='b1'
        )
        
        self.__w1_g = tf.Variable(
            initial_value=tf.random.normal(shape=[1, 1, output_filter, input_filter], stddev=STDDEV),
            name='w1'
        )
        self.__b1_g = tf.Variable(
            initial_value=tf.random.normal(shape=[input_filter], stddev=STDDEV),
            name='b1'
        )
        
        self.__w2 = tf.Variable(
            initial_value=tf.random.normal(shape=[1, 1, input_filter, 1], stddev=STDDEV),
            name='w2'
        )
        self.__b2 = tf.Variable(
            initial_value=tf.random.normal(shape=[1], stddev=STDDEV)
        )
        
        self.__w3 = tf.Variable(
            initial_value=tf.random.normal(shape=[2, 2, input_filter, 1], stddev=STDDEV),
            name='w3'
        )
        self.__b3 = tf.Variable(
            initial_value=tf.random.normal(shape=[input_filter], stddev=STDDEV)
        )
    
    def __call__(self, x, g) -> any:
        
        # saving output size
        __size = x.shape[1]
        
        # concating x and g
        _x = conv2d(input=x, filters=self.__w1_x, strides=2, padding='SAME') + self.__b1_x
        _g = conv2d(input=g, filters=self.__w1_g, strides=1, padding='VALID') + self.__b1_g
        add_x_g = add(x=_x, y=_g)
        
        # relu
        add_relu = relu(features=add_x_g)
        
        # wise
        wise = conv2d(input=add_relu, filters=self.__w2, strides=1, padding='VALID') + self.__b2
        
        # sigmoid
        sig = sigmoid(x=wise)
        
        # resample
        resample = conv2d_transpose(input=sig, filters=self.__w3, output_shape=[__size, __size], strides=2, padding='SAME') + self.__b3
        
        # multiply
        mult = multiply(x=resample, y=x)
        
        return mult
    
# convolutional block
class Conv2DBlock(tf.Module):
    
    def __init__(self, input_filter: int, output_filter: int, name: str = 'Conv2DBlock'):
        super().__init__(name)
        
        # initializing weights and bias
        self.__w = tf.Variable(
            initial_value=tf.random.normal(shape=[3, 3, input_filter, output_filter]),
            name='w'
        )
        self.__b = tf.Variable(
            initial_value=tf.random.normal(shape=[output_filter]),
            name='b'
        )
        
    def __call__(self, x) -> any:
        
        # forward propagation
        x = conv2d(input=x, filters=self.__w, strides=1, padding='SAME') + self.__b
        x = batch_normalization(x=x, mean=MEAN, variance=VARIANCE, offset=OFFSET, scale=SCALE, variance_epsilon=EPSILON)
        x = relu(features=x)
        
        return x