import numpy as np
import pickle
import os


class DataLoader:
    def __init__(self, **kwargs):
        self.dataset_name = kwargs['dataset']
        self.dataset_path = kwargs['dataset_path']
        self.train_x = []
        self.train_y = []
        self.test_x = []
        self.test_y = []
        if self.dataset_name == 'cifar-10':
            for i in range(5):
                try:
                    with open(os.path.join(self.dataset_path, 'data_batch_{}'.format(i)), 'rb') as fo:
                        dict = pickle.load(fo)
                        self.train_x.append(dict['data'])
                        self.train_y.append(dict['labels'])
                except FileNotFoundError:
                    print('Please check the path of dataset.')

            try:
                with open(os.path.join(self.dataset_path, 'test_batch'), 'rb') as fo:
                    dict = pickle.load(fo)
                    self.test_x = dict['data']
                    self.test_y = dict['labels']
            except FileNotFoundError:
                print('Please check the path of dataset.')


