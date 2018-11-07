import numpy as np
import pickle
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from keras.utils import to_categorical as onehot
from tqdm import tqdm


class DataLoader:
    def __init__(self, dataset_name, dataset_path, batch_size, image_mode=True, standardization=False, use_knn=False,
                 knn=8, use_pca=False, perc_var=0.99):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.train_x = []
        self.train_y = []
        self.test_x = []
        self.test_y = []

        self.image_mode = image_mode
        self.standardization = standardization
        self.use_knn = use_knn
        self.use_pca = use_pca
        self.perc_var = perc_var
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
        self.train_y = np.concatenate(self.train_y, axis=0).astype(np.int32)
        self.test_x = np.concatenate(self.test_x, axis=0).astype(np.float32)
        self.test_y = np.concatenate(self.test_y, axis=0).astype(np.int32)

        self.num_outputs = np.max(self.train_y) + 1
        self.onehot_train_y = onehot(self.train_y, num_classes=self.num_outputs)
        self.onehot_test_y = onehot(self.test_y, num_classes=self.num_outputs)

        if self.use_pca:
            print('first use pca to reduce the dimension...')
            pca_cache_dir = os.path.join(self.dataset_path, 'pca_cache')
            pca_cache_train_path = os.path.join(pca_cache_dir, 'pca_cache_train.npy')
            pca_cache_test_path = os.path.join(pca_cache_dir, 'pca_cache_test.npy')
            if os.path.exists(pca_cache_test_path) and os.path.exists(pca_cache_train_path):
                print('pca caches have already done...')
                self.train_x_pca = np.load(pca_cache_train_path)
                self.test_x_pca = np.load(pca_cache_test_path)
                print('pca representation for training and test data loaded...')
            else:
                print('pca caches have not done, get them now...')
                os.makedirs(pca_cache_dir, exist_ok=True)
                pca = PCA()
                pca.fit(self.train_x)
                var = np.cumsum(pca.explained_variance_ratio_)
                pca_dims = np.searchsorted(var, self.perc_var) + 1
                self.train_x_pca = pca.transform(self.train_x)[:, :pca_dims]
                self.test_x_pca = pca.transform(self.test_x)[:, :pca_dims]
                np.save(pca_cache_train_path, self.train_x_pca)
                np.save(pca_cache_test_path, self.test_x_pca)
                print('pca dims is {}. pca caches are saved...'.format(pca_dims))
        else:
            self.train_x_pca = self.train_x
            self.test_x_pca = self.test_x

        if self.use_knn:
            print('find the nearest neighbors...')
            nn_cache_path = os.path.join(self.dataset_path, 'nn_cache')
            nn_cache_train_path = os.path.join(nn_cache_path, 'cifar_10_nn_training_data.npy')
            nn_cache_test_path = os.path.join(nn_cache_path, 'cifar_10_nn_test_data.npy')
            if os.path.exists(nn_cache_train_path) and os.path.exists(nn_cache_test_path):
                print('nearest neighbors have been cached...')
                self.nn_indices_train = np.load(nn_cache_train_path)[:, 1:self.k + 1]
                self.nn_indices_test = np.load(nn_cache_test_path)[:, 0:self.k]
                print('nearest neighbors loaded')
            else:
                print('nearest neighbors have not been cached, find and cache 500 nearest neighbors now...')
                os.makedirs(nn_cache_path, exist_ok=True)
                nn = NearestNeighbors(n_neighbors=500, n_jobs=-1, algorithm='auto')
                nn.fit(self.train_x)
                nn_indices_train = nn.kneighbors(self.train_x, return_distance=False)
                nn_indices_test = nn.kneighbors(self.test_x, return_distance=False)
                np.save(nn_cache_train_path, nn_indices_train)
                np.save(nn_cache_test_path, nn_indices_test)
                self.nn_indices_train = nn_indices_train[:, 1:self.k + 1]
                self.nn_indices_test = nn_indices_test[:, 0:self.k]
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
        knn=8,
        use_pca=True
    )
    training_data = data_loader.training_data(loss=0)
    for a, b, c, d in training_data:
        print(np.shape(a))
        print(np.shape(b))
        print(np.shape(c))
        print(np.shape(d))
