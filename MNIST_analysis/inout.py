import pandas as pd
import numpy as np
import pickle


def read_MNIST_from_csv():
    train = pd.read_csv('/media/geoff/SHARED/Kaggle/MNIST/train.csv', header=0)
    test = pd.read_csv('/media/geoff/SHARED/Kaggle/MNIST/test.csv', header=0)
    train['label'] += 1
    test['label'] = np.zeros(len(test))
    data = pd.concat((train, test))
    X = np.array(data.drop('label'))
    Y = np.array(data['label']).astype(int)
    return X, Y


def read_MNIST_from_pickle():
    # TODO: implement
    pass
