# libraries
from src.net.blocks import DownSamplingBlock,\
                           UpSamplingBlock,\
                           AttentionGate,\
                           Conv2DBlock,\
                           STDDEV

from tensorflow._api.v2.nn import conv2d
from tensorflow._api.v2.math import argmax

import tensorflow as tf


# model
class TumorSegmentationDetector(tf.Module):
    
    def __init__(self, name: str = 'TumorSegmentationDetector') -> None:
        super().__init__(name)
        
        # initializeing blocks
        self.__block_1 = DownSamplingBlock(3, 32)
        self.__block_2 = DownSamplingBlock(32, 64)
        self.__block_3 = DownSamplingBlock(64, 128)
        self.__block_4 = DownSamplingBlock(128, 256)
        self.__block_5 = DownSamplingBlock(256, 512)
        
        self.__block_6 = UpSamplingBlock(512, 1024)
        self.__block_7 = UpSamplingBlock(1536, 512)
        self.__block_8 = UpSamplingBlock(768, 256)
        self.__block_9 = UpSamplingBlock(384, 128)
        self.__block_10 = UpSamplingBlock(192, 64)
        
        self.__block_11 = Conv2DBlock(96, 32)
        self.__block_12 = Conv2DBlock(32, 32)
        
        self.__attention_1 = AttentionGate(512, 1024)
        self.__attention_2 = AttentionGate(256, 512)
        self.__attention_3 = AttentionGate(128, 256)
        self.__attention_4 = AttentionGate(64, 128)
        self.__attention_5 = AttentionGate(32, 64)
        
        # initializing weights and bias
        self.__w = tf.Variable(
            initial_value=tf.random.normal(shape=[1, 1, 32, 2], stddev=STDDEV),
            name='w'
        )
        self.__b = tf.Variable(
            initial_value=tf.random.normal(shape=[2], stddev=STDDEV),
            name='b'
        )
        
    def __call__(self, x) -> any:
        
        # forward propagation
        x1, down1 = self.__block_1(x)
        x2, down2 = self.__block_2(down1)
        x3, down3 = self.__block_3(down2)
        x4, down4 = self.__block_4(down3)
        x5, down5 = self.__block_5(down4)
        
        g5, up1 = self.__block_6(down5)
        attention1 = self.__attention_1(x5, g5)
        concat = tf.concat(values=[attention1, up1], axis=-1)
        
        g4, up2 = self.__block_7(concat)
        attention2 = self.__attention_2(x4, g4)
        concat = tf.concat(values=[attention2, up2], axis=-1)
        
        g3, up3 = self.__block_8(concat)
        attention3 = self.__attention_3(x3, g3)
        concat = tf.concat(values=[attention3, up3], axis=-1)
        
        g2, up4 = self.__block_9(concat)
        attention4 = self.__attention_4(x2, g2)
        concat = tf.concat(values=[attention4, up4], axis=-1)
        
        g1, up5 = self.__block_10(concat)
        attention5 = self.__attention_5(x1, g1)
        concat = tf.concat(values=[attention5, up5], axis=-1)
        
        # final blocks
        x = self.__block_11(concat)
        x = self.__block_12(x)
        
        x = conv2d(input=x, filters=self.__w, strides=1, padding='SAME') + self.__b
        x = argmax(input=x, axis=-1)
        x = tf.reshape(tensor=x, shape=[-1, 256, 256, 1], name='mask-output-layer')
        
        return x
