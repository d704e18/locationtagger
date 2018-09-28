import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setattr__


class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


def load_data(file_path, normalize=True):

    data = pd.read_pickle(file_path)

    print("Columns:\n {}".format(data.columns))
    print("Data shape: {}\n".format(data.shape))

    train_val_split = '2018-09-11 10:00'
    val_test_split = '2018-09-12 12:00'

    train_x = data[:train_val_split].iloc[:, 0:-1].values.astype(np.float32)
    train_y = data[:train_val_split].iloc[:, -1:].values.astype(np.int32)

    validation_x = data[train_val_split:val_test_split].iloc[:, 0:-1].values.astype(np.float32)
    validation_y = data[train_val_split:val_test_split].iloc[:, -1:].values.astype(np.int32)

    test_x = data[val_test_split:].iloc[:, 0:-1].values.astype(np.float32)
    test_y = data[val_test_split:].iloc[:, -1:].values.astype(np.int32)

    if normalize:
        transformer = Normalizer()
        train_x = transformer.fit_transform(train_x)
        validation_x = transformer.transform(validation_x)
        test_x = transformer.transform(test_x)

    print("Train shape: {}".format(train_x.shape))
    print("Validation shape: {}".format(validation_x.shape))
    print("Test shape: {}".format(test_x.shape))

    return train_x, one_hot(train_y), validation_x, one_hot(validation_y), test_x, one_hot(test_y)


def one_hot(y):
    num_classes = np.unique(y).shape[0]

    return np.squeeze(np.eye(num_classes)[y.reshape(-1).astype(int)]).astype(np.int32)


if __name__ == "__main__":

    path = os.getcwd()
    parent = '/'.join(path.split('/')[:-1])

    train_x, train_y, validation_x, validation_y, test_x, test_y = load_data(parent+"/data/whole_week.pkl")

    print(train_x.dtype)
    print(train_y.dtype)
