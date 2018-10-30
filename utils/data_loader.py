import numpy as np
import pickle
import os
from sklearn.neighbors import NearestNeighbors
from keras.utils import to_categorical as onehot
from tqdm import tqdm


class DataLoader:
    def __init__(self, dataset_name, dataset_path, batch_size, image_mode=True, standardization=False, use_knn=False,
                 knn=8, use_pca=False):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.train_x = []
        self.train_y = []
        self.test_x = []
        self.test_y = []

        self.image_mode = image_mode
        self.standardization = standardization
        self.use_knn = use_knn
        self.k = knn

        self.batch_size = batch_size

        if self.dataset_name == 'cifar-10':
            for i in range(1, 6):
                try:
                    with open(os.path.join(self.dataset_path, 'data_batch_{}'.format(i)), 'rb') as fo:
                        dict = pickle.load(fo, encoding='bytes')
                        self.train_x.append(dict[b'data'])
                        self.train_y.append(dict[b'labels'])
                except FileNotFoundError:
                    print('Please check the path of dataset.')
                    exit(0)

            try:
                with open(os.path.join(self.dataset_path, 'test_batch'), 'rb') as fo:
                    dict = pickle.load(fo, encoding='bytes')
                    self.test_x = dict[b'data']
                    self.test_y = dict[b'labels']
            except FileNotFoundError:
                print('Please check the path of dataset.')
                exit(0)

        if self.dataset_name == 'cifar-100':
            try:
                with open(os.path.join(self.dataset_path, 'train'), 'rb') as fo:
                    dict = pickle.load(fo, encoding='bytes')
                    self.train_x = dict[b'data']
                    self.train_y = dict[b'labels']
            except FileNotFoundError:
                print('Please check the path of dataset')
                exit(0)

            try:
                with open(os.path.join(self.dataset_path, 'test'), 'rb') as fo:
                    dict = pickle.load(fo, encoding='bytes')
                    self.test_x = dict[b'data']
                    self.test_y = dict[b'labels']
            except FileNotFoundError:
                print('Please check the path of dataset')
                exit(0)

        print('data loaded...')

        self.train_x = np.concatenate(self.train_x, axis=0).astype(np.float32)
        self.train_y = np.array(self.train_y).astype(np.int32)
        self.test_x = np.array(self.test_x).astype(np.float32)
        self.test_y = np.array(self.test_y).astype(np.int32)

        self.num_outputs = np.max(self.train_y) + 1
        self.onehot_train_y = onehot(self.train_y, num_classes=self.num_outputs)
        self.onehot_test_y = onehot(self.test_y, num_classes=self.num_outputs)

        if self.use_knn:
            print('find the nearest neighbors...')
            cache_path = os.path.join(self.dataset_path, 'nn_cache')
            if os.path.exists(cache_path):
                print('nearest neighbors have been cached...')
                self.nn_indices_train = np.load('cifar_10_nn_training_data.npy')
                self.nn_indices_test = np.load('cifar_10_nn_test_data.npy')
                print('nearest neighbors loaded')
            else:
                print('nearest neighbors have not been cached, find and cache now...')
                nn = NearestNeighbors(n_neighbors=self.k, n_jobs=-1)
                nn.fit(self.train_x)
                self.nn_indices_train = nn.kneighbors(self.train_x, n_neighbors=self.k + 1, return_distance=False)[:, 1:]
                self.nn_indices_test = nn.kneighbors(self.test_x, return_distance=False)
                np.save(os.path.join(cache_path, 'cifar_10_nn_training_data.npy'), self.nn_indices_train)
                np.save(os.path.join(cache_path, 'cifar_10_nn_test_data.npy'), self.nn_indices_test)
                print('nearest neighbors got and cached...')
        else:
            self.nn_indices_train = None
            self.nn_indices_test = None

        if self.image_mode:
            '''
            channel last mode
            '''
            self.train_x = self.train_x.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
            self.test_x = self.test_x.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)

        if self.standardization:
            mean = np.mean(self.train_x, axis=0)
            std = np.std(self.train_x, axis=0)
            self.train_x = (self.train_x - mean) / std
            self.test_x = (self.test_x - mean) / std

        self.num_train = np.shape(self.train_x)[0]
        self.num_test = np.shape(self.test_x)[0]

        self.num_batches_train = self.num_train // self.batch_size
        self.num_batches_test = self.num_test // self.batch_size

        self.dims = np.shape(self.train_x)[1:]

        print(np.shape(self.train_x))
        print(np.shape(self.train_y))
        print(np.shape(self.test_x))
        print(np.shape(self.test_y))

    def training_next_batch(self):
        num = self.num_train
        start = 0
        end = self.batch_size
        rprm = np.random.permutation(num)

        while end < num:
            if self.use_knn and self.k is not None:
                nn_indices = self.nn_indices_train[rprm[start:end]]
                yield (
                    self.train_x[rprm[start:end]],
                    self.train_y[rprm[start:end]],
                    self.train_x[nn_indices],
                    self.onehot_train_y[nn_indices]
                )
            else:
                yield (
                    self.train_x[rprm[start:end]],
                    self.train_y[rprm[start:end]],
                    np.zeros(shape=(self.batch_size, self.k, self.dims)),
                    np.zeros(shape=(self.batch_size, self.k, self.num_outputs))
                )
            start = end
            end += self.batch_size

    def training_data(self, loss=0.0):
        return tqdm(
            iterable=self.training_next_batch(),
            desc='Train loss: {:.4f}'.format(loss),
            total=self.num_batches_train,
            mininterval=1.0
        )

    def testing_next_batch(self):
        num = self.num_test
        start = 0
        end = self.batch_size

        while end < num:
            if self.use_knn and self.k is not None:
                nn_indices = self.nn_indices_test[start:end]
                yield (
                    self.test_x[start:end],
                    self.test_y[start:end],
                    self.train_x[nn_indices],
                    self.onehot_train_y[nn_indices]
                )
            else:
                yield (
                    self.test_x[start:end],
                    self.test_y[start:end],
                    np.zeros(shape=())
                )

    def testing_data(self):
        return tqdm(
            iterable=self.testing_next_batch(),
            desc='Test Iterations: ',
            total=self.num_batches_test
        )


if __name__ == '__main__':
    data_loader = DataLoader(
        dataset_name='cifar-10',
        dataset_path='../datasets/cifar-10-batches-py/',
        batch_size=32,
        image_mode=True,
        standardization=False,
        use_knn=True,
        knn=8
    )
    training_data = data_loader.training_data(loss=0)
    for a, b, c, d in training_data:
        print(np.shape(a))
        print(np.shape(b))
        print(np.shape(c))
        print(np.shape(d))
