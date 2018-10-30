import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Dense, ReLU, Flatten
from ..utils import data_loader


class CifarNet:
    def __init__(self):
        self.fetcher = data_loader.DataLoader(dataset='cifar-10', dataset_path='../datasets/cifar-10-batches-py',
                                              batch_size=32)
        self.sess = tf.Session()
        self.num_batches = 10000
        pass

    def build_graph(self):
        self.X_placeholder = tf.placeholder(shape=[-1, 32, 32, 3], dtype=tf.float32)
        self.Y_placeholder = tf.placeholder(shape=[-1, 1], dtype=tf.int32)

        z = Conv2D(filters=32, kernel_size=[5, 5], strides=[1, 1], padding='same')(self.X_placeholder)
        z = ReLU(z)
        z = BatchNormalization(z)
        z = Conv2D(filters=64, kernel_size=[5, 5], strides=[1, 1], padding='same')(z)
        z = ReLU(z)
        z = BatchNormalization(z)
        z = MaxPool2D(pool_size=[2, 2], strides=[2, 2])(z)
        z = Flatten(z)
        z = Dense(units=128, activation='relu')
        z = Dense(units=128, activation='relu')

        self.logits = Dense(units=10, activation='sigmoid')
        self.loss = tf.losses.sparse_softmax_cross_entropy(logits=self.logits, labels=self.Y_placeholder)
        self.accuracy = tf.reduce_sum(
            tf.cast(tf.equal(tf.cast(tf.argmax(self.logits, axis=-1), tf.int32), self.Y_placeholder),
                    tf.int32)) / len(self.fetcher.test_x)
        self.optimizor = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
        self.train = self.optimizor.minimize(self.loss)

    def fit(self):
        self.sess.run(tf.global_variables_initializer())
        for index in range(self.num_batches):
            x, y = self.fetcher.training_next_batch()
            loss, _ = self.sess.run([self.loss, self.train], feed_dict={self.X_placeholder: x,
                                                                        self.Y_placeholder: y})
            if index % 10 == 0:
                print('batch index {} loss {}'.format(index, loss))


if __name__ == '__main__':
    trainer = CNNModel()
    trainer.fit()
