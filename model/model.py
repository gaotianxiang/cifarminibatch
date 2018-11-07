import keras
import tensorflow as tf
import numpy as np
from tensorflow.layers import Conv2D, MaxPooling2D, BatchNormalization, Dense, Flatten, Dropout
from ..utils import data_loader


class CifarNet:
    def __init__(self, params):
        self.dataset_name = params['dataset_name']
        self.dataset_path = params['dataset_path']
        self.batch_size = params['batch_size']
        self.image_mode = params['image_mode']
        self.standardization = params['standardization']
        self.use_knn = params['use_knn']
        self.knn = params['knn']
        self.use_pca = params['use_pca']

        self.data_loader = data_loader.DataLoader(
            dataset_name=params['dataset_name'],
            dataset_path=params['dataset_path'],
            batch_size=params['batch_size'],
            image_mode=params['image_mode'],
            standardization=params['standardization'],
            use_knn=params['use_knn'],
            knn=params['knn'],
            use_pca=params['use_pca']
        )

        self.x1 = tf.placeholder(dtype=tf.float32, shape=[None, self.data_loader.dims])
        self.y1 = tf.placeholder(dtype=tf.int32, shape=[None, ])
        self.x2 = tf.placeholder(dtype=tf.float32, shape=[None, self.data_loader.k, self.data_loader.dims])
        self.y2 = tf.placeholder(dtype=tf.int32, shape=[None, self.data_loader.k, self.data_loader.num_outputs])

    def build_graph(self):
        x1 = Conv2D(filters=)


if __name__ == '__main__':
    pass
