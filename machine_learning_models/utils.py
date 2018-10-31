import os

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


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


def load_data(file_path, normalize=True, drop_device=True):
    if file_path.endswith('pkl'):
        data = pd.read_pickle(file_path)
    else:
        data = pd.read_csv(file_path, parse_dates=True, dayfirst=False, index_col=0)

    if drop_device and 'Device' in data.columns:
        data = data.drop("Device", axis=1)

    print("Columns:\n {}".format(data.columns))
    print("Data shape: {}\n".format(data.shape))

    x = data.iloc[:, 0:-1].values.astype(np.float32)
    y = data.iloc[:, -1:].values.astype(np.int32)

    transformer = StandardScaler()
    if normalize:
        x = transformer.fit_transform(x)

    return x, one_hot(y), transformer


def load_prediction_data(file_path, device_id, transformer):
    if file_path.endswith('pkl'):
        data = pd.read_pickle(file_path)
    else:
        data = pd.read_csv(file_path, parse_dates=True, dayfirst=False, index_col=0)
    data = data.loc[data['Device'] == device_id]
    timestamps = data.index
    x = data.iloc[:, 0:-2].values.astype(np.float32)
    y = data.iloc[:, -1]

    return transformer.transform(x), y, timestamps


def load_data_and_split(file_path, normalize=True, drop_device=True, k=5):
    if file_path.endswith('pkl'):
        data = pd.read_pickle(file_path)
    else:
        data = pd.read_csv(file_path, parse_dates=True, dayfirst=False, index_col=0)

    if drop_device and 'Device' in data.columns:
        data = data.drop('Device', axis=1)

    print("Columns:\n {}".format(data.columns))
    print("Data shape: {}\n".format(data.shape))

    n = data.shape[0]

    percentage = (k-1) / k

    split = int(round(n*percentage))

    train_x = data[:split].iloc[:, 0:-1].values.astype(np.float32)
    train_y = data[:split].iloc[:, -1:].values.astype(np.int32)

    validation_x = data[split:].iloc[:, 0:-1].values.astype(np.float32)
    validation_y = data[split:].iloc[:, -1:].values.astype(np.int32)

    transformer = StandardScaler()
    if normalize:
        train_x = transformer.fit_transform(train_x)
        validation_x = transformer.transform(validation_x)

    print("Train shape: {}".format(train_x.shape))
    print("Validation shape: {}".format(validation_x.shape))

    return train_x, train_y, validation_x, one_hot(validation_y), transformer


def one_hot(y):
    num_classes = np.unique(y).shape[0]

    return np.squeeze(np.eye(num_classes)[y.reshape(-1).astype(int)]).astype(np.int32)


def make_sgd_classifier(eta0=0.00003):
    return SGDClassifier(loss='log', shuffle=True, max_iter=100, penalty=None, class_weight='balanced', eta0=eta0,
                         learning_rate='adaptive')


if __name__ == "__main__":

    path = os.getcwd()
    parent = '/'.join(path.split('/')[:-1])

    train_x, train_y, validation_x, validation_y, test_x, test_y = load_data(parent+"/data/whole_week.pkl")

    print(train_x.dtype)
    print(train_y.dtype)
