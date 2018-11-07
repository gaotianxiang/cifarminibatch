import argparse
import os
import sys


def read_options(do_print=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='cifar-10')
    parser.add_argument('--dataset_path', type=str, default='./datasets/cifar-10-batches-py/')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--image_mode', default=True, type=bool)
    parser.add_argument('--standardization', default=True, type=bool)
    parser.add_argument('--use_knn', default=False, type=bool)
    parser.add_argument('--knn', default=8, type=int)
    parser.add_argument('--use_pca', default=True, type=bool)
    parser.add_argument('--model_architecture', type=str, default='vgg')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--perc_var', type=float, default=0.99)

    try:
        parsed = vars(parser.parse_args())
    except IOError as ioerr:
        parser.error(str(ioerr))
        exit(-1)

    if do_print:
        print_parsed(parsed)
    return parsed


def print_parsed(parsed):
    # print and return
    maxLen = max([len(ii) for ii in parsed.keys()])
    fmtString = '\t%' + str(maxLen) + 's : %s'
    print('Arguments:')
    for keyPair in sorted(parsed.items()):
        print(fmtString % keyPair)


if __name__ == '__main__':
    read_options()
