import numpy as np
import pickle
import os
from sklearn.neighbors import NearestNeighbors
from keras.utils import to_categorical as onehot


class DataLoader:
    def __init__(self, dataset_name, dataset_path, batch_size, image_mode=True, standardization=False, use_knn=False,
                 knn=8):
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

        self.train_x = np.concatenate(self.train_x, axis=0).astype(np.float32)
        self.train_y = np.array(self.train_y).astype(np.int32)
        self.test_x = np.array(self.test_x).astype(np.float32)
        self.test_y = np.array(self.test_y).astype(np.int32)

        self.num_outputs = np.max(self.train_y) + 1
        self.onehot_train_y = onehot(self.train_y, num_classes=self.num_outputs)
        self.onehot_test_y = onehot(self.test_y, num_classes=self.num_outputs)

        if self.use_knn:
            nn = NearestNeighbors(n_neighbors=self.k, n_jobs=-1)
            nn.fit(self.train_x)
            self.nn_indices_train = nn.kneighbors(self.train_x, n_neighbors=self.k + 1, return_distance=False)[:, 1:]
            self.nn_indices_test = nn.kneighbors(self.test_x, return_distance=False)
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

        self.batch_indict = 0

        print(np.shape(self.train_x))
        print(np.shape(self.train_y))
        print(np.shape(self.test_x))
        print(np.shape(self.test_y))

    def training_next_batch(self):
        index = np.random.randint(0, len(self.train_x), self.batch_size)
        return self.train_x[index], self.train_y[index]

    def testing_next_batch(self):
        index = np.random.randint(0, len(self.test_x), self.batch_size)
        return self.test_x[index], self.test_y[index]

    def get_test(self):
        return self.test_x, self.test_y


if __name__ == '__main__':
    fetcher = DataLoader(dataset_name='cifar-10', dataset_path='../datasets/cifar-10-batches-py')
    # fetcher_100 = DataLoader(dataset='cifar-100', dataset_path='../datasets/cifar-100-python')
