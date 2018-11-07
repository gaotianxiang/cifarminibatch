import argparse
import os
import sys


def read_options(do_print=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str)
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('batch_size', default=32, type=int)
    parser.add_argument('image_mode', default=True, type=bool)
    parser.add_argument('standardization', default=True, type=bool)
    parser.add_argument('use_knn', default=False, type=bool)
    parser.add_argument('knn', default=8, type=int)
    parser.add_argument('use_pca', default)