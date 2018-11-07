import keras
import tensorflow as tf
import numpy as np
from tensorflow.layers import conv2d, max_pooling2d, batch_normalization, dense, flatten, dropout
from tensorflow.nn import relu as ReLU
from utils import data_loader
from utils import options as opts
from tqdm import trange
from tqdm import tqdm


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
        self.model_architecture = params['model_architecture']
        self.epochs = params['epochs']
        self.lr = params['lr']

        self.data_loader = data_loader.DataLoader(
            dataset_name=params['dataset_name'],
            dataset_path=params['dataset_path'],
            batch_size=params['batch_size'],
            image_mode=params['image_mode'],
            standardization=params['standardization'],
            use_knn=params['use_knn'],
            knn=params['knn'],
            use_pca=params['use_pca'],
            perc_var=params['perc_var']
        )

        self.x1 = tf.placeholder(dtype=tf.float32, shape=(None,) + self.data_loader.dims)
        self.y1 = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.x2 = tf.placeholder(dtype=tf.float32, shape=(None, self.data_loader.k) + self.data_loader.dims)
        self.y2 = tf.placeholder(dtype=tf.int32, shape=(None, self.data_loader.k, self.data_loader.num_outputs))

        self.loss, self.pred, self.err = self.build_graph()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = self.optimizer.minimize(self.loss)

        self.sess = tf.Session()

    def build_graph(self):
        def vgg(inputs, ys, knn_inputs, knn_ys):
            # block 1
            x = conv2d(inputs, filters=64, kernel_size=(3, 3), padding='same', activation=ReLU)
            x = conv2d(x, filters=64, kernel_size=(3, 3), padding='same', activation=ReLU)
            x = max_pooling2d(x, pool_size=(2, 2), strides=(2, 2))

            # block 2
            x = conv2d(x, filters=128, kernel_size=(3, 3), padding='same', activation=ReLU)
            x = conv2d(x, filters=128, kernel_size=(3, 3), padding='same', activation=ReLU)
            x = max_pooling2d(x, pool_size=(2, 2), strides=(2, 2))

            # final
            x = flatten(x)
            x = dense(x, units=1024, activation=ReLU)
            x = dropout(x, rate=0.5, training=)
            x = dense(x, units=1024, activation=ReLU)
            x = dense(x, units=self.data_loader.num_outputs)

            return x

        def network(inputs, ys, knn_inputs, knn_ys):
            if self.model_architecture == 'vgg':
                fcout = vgg(inputs, ys, knn_inputs, knn_ys)
            else:
                fcout = None
            loss = tf.losses.sparse_softmax_cross_entropy(labels=ys, logits=fcout)
            pred = tf.to_int32(tf.argmax(fcout, axis=-1))
            err = tf.reduce_sum(1.0 - tf.to_float(tf.equal(pred, ys)))
            return loss, pred, err

        return network(inputs=self.x1, ys=self.y1, knn_inputs=self.x2, knn_ys=self.y2)

    def fit(self):
        self.sess.run(tf.global_variables_initializer())
        loss_val = 0.0
        print('initial test error: {}'.format(self.evaluate()))
        for i in trange(self.epochs, desc='Epochs: '):
            train_data = self.data_loader.training_data(loss_val)
            for x1, y1, x2, y2 in train_data:
                loss_val, _ = self.sess.run(
                    fetches=(self.loss, self.train_op),
                    feed_dict={
                        self.x1: x1,
                        self.y1: y1,
                        self.x2: x2,
                        self.y2: y2
                    }
                )
                train_data.set_description('Train loss: {0:.4f}'.format(loss_val))
            print('test error after {} epoch(s): {}'.format(i + 1, self.evaluate()))

    def evaluate(self):
        counts = 0
        sum_err = 0.0
        data = self.data_loader.testing_data()

        for x1, y1, x2, y2 in data:
            counts += len(y1)
            sum_err += self.sess.run(
                self.err,
                feed_dict={
                    self.x1: x1,
                    self.y1: y1,
                    self.x2: x2,
                    self.y2: y2
                }
            )
            # print(sum_err)
        return sum_err / counts


def main(options=None, **kwargs):
    if options is None:
        options = opts.read_options(do_print=False)
    for key, val in kwargs.items():
        options[key] = val
    opts.print_parsed(options)

    net = CifarNet(options)
    net.fit()


if __name__ == '__main__':
    main(None)
