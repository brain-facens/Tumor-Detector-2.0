# libraries
from sklearn.metrics import precision_recall_fscore_support,\
                            accuracy_score
from PIL import Image

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf


def load_model(version: str) -> tf.Module:

    # checking if version is available
    __path = os.path.join('weights', 'tumor_segmentation_' + version + '.h5')
    assert os.path.exists(path=__path), 'model does not exist.'

    return tf.keras.models.load_model(
        filepath=__path
    )

tfrecord_reader = {
    'image': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    'mask': tf.io.FixedLenFeature(shape=[], dtype=tf.string)
}

def parser_dataset(example_proto):
    
    content = tf.io.parse_single_example(example_proto, tfrecord_reader)

    image = content['image']
    mask = content['mask']

    image = tf.io.parse_tensor(image, tf.float32)
    mask = tf.io.parse_tensor(mask, tf.float32)

    image = tf.reshape(image, [256,256,3])
    mask = tf.reshape(mask, [256,256,1])

    return image, mask

def image_dataset(folder):
    __data = tf.data.TFRecordDataset(filenames=os.path.join(folder, 'test.tfrecord'))
    __data = __data.map(parser_dataset).batch(1).shuffle(True)
    return __data

def applyMask(img, mask, pred):
    
    img = img.copy()

    # applying three channels in mask
    newMask = np.zeros( img.shape )
    newMask[:,:,0] = mask
    if pred.sum() > 500000:
        newMask[:,:,1] = pred

    maskedimg = img * newMask # multiply `im` by the mask to zero out non-border pixels
    maskedimg = np.clip(maskedimg, 0, 255)
    maskedimg = maskedimg.astype(np.uint8)

    img[ maskedimg>0 ] = maskedimg[ maskedimg>0 ] # multiply `im` by the mask to zero out non-border pixels

    return img

def estimateResults(label, predict):
    accuracy = accuracy_score(label, predict)
    precision, recall, _, _ = precision_recall_fscore_support(label, predict)
    return precision, recall, accuracy