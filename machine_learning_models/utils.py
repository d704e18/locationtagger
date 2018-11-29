import os

import numpy as np
import pandas as pd
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
        data = pd.read_csv(
            file_path, parse_dates=True, dayfirst=False, index_col=0)

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
        data = pd.read_csv(
            file_path, parse_dates=True, dayfirst=False, index_col=0)
    data = data.loc[data['Device'] == device_id]
    timestamps = data.index
    x = data.iloc[:, 0:-2].values.astype(np.float32)
    y = data.iloc[:, -1]

    return transformer.transform(x), y, timestamps


def scale_df(df, drop_columns, end1, start2):
    scalable_df = df.drop(drop_columns, axis=1)
    transformer = StandardScaler().fit(scalable_df[:end1].append(
        scalable_df[start2:]))

    scalable_df = pd.DataFrame(
        transformer.transform(scalable_df),
        index=scalable_df.index,
        columns=scalable_df.columns)

    # Keep the unscaled columns without modifying them
    scalable_df[drop_columns] = df[drop_columns]
    return scalable_df, transformer


def get_example_generator(file_path, repeat=True, clip=None):
    """
    Returns three generators:
    train_gen contains the first 5 days
    val_gen   contains the second to last day
    test_gen  contains the last day
    """
    # validation_start = '2018-09-11 10:00'
    # test_start = '2018-09-12 12:00'
    validation_start = '2018-09-09 18:00'
    validation_end = '2018-09-10 18:00'

    if file_path.endswith('.csv'):
        df = pd.read_csv(
            file_path, dayfirst=False, parse_dates=True, index_col=0)
    elif file_path.endswith('.pkl'):
        df = pd.read_pickle(file_path)
    else:
        raise ValueError("Unknown filetype", file_path)

    areas = df.iloc[:, -1]
    hot_areas = one_hot(areas)
    m, n = hot_areas.shape
    area_names = []
    for i in range(n):
        area_name = 'area{}'.format(i)
        area_names += [area_name]
        df[area_name] = hot_areas[:, i]

    df = df.drop('areas', axis=1)

    # Drop the columns that should not be scaled
    unscalable_columns = ['Device'] + area_names
    df, transformer = scale_df(df, unscalable_columns, validation_start,
                           validation_end)
    df = df.sort_index()

    # Train on everything outside of the validation interval
    train_df = df[:validation_start].append(df[validation_end:])
    validation_df = df[validation_start:validation_end]
    # Split validation into test and validation
    test_df = validation_df.iloc[validation_df.shape[0] // 2:]
    validation_df = validation_df.iloc[:validation_df.shape[0] // 2]

    def get_generator(df):
        groups = df.groupby('Device')
        for device, group in groups:
            person = group.drop('Device', axis=1)
            labels = person[area_names]
            labels = labels.values  # for production
            readings = person.drop(area_names, axis=1)
            if clip is not None:
                labels = labels[:clip]
                readings = readings[:clip]
            yield readings, labels  # , device

    return get_generator(train_df), get_generator(
        validation_df), get_generator(test_df), transformer

    # return train_x, train_y, validation_x, validation_y, test_x, test_y


def load_data_and_split(file_path, normalize=True, drop_device=True):
    if file_path.endswith('pkl'):
        data = pd.read_pickle(file_path)
    else:
        data = pd.read_csv(
            file_path, parse_dates=True, dayfirst=False, index_col=0)

    if drop_device and 'Device' in data.columns:
        data = data.drop('Device', axis=1)

    print("Columns:\n {}".format(data.columns))
    print("Data shape: {}\n".format(data.shape))

    train_val_split = '2018-09-11 10:00'
    val_test_split = '2018-09-12 12:00'

    train_x = data[:train_val_split].iloc[:, 0:-1].values.astype(np.float32)
    train_y = data[:train_val_split].iloc[:, -1:].values.astype(np.int32)

    validation_x = data[train_val_split:
                        val_test_split].iloc[:, 0:-1].values.astype(np.float32)
    validation_y = data[train_val_split:
                        val_test_split].iloc[:, -1:].values.astype(np.int32)

    test_x = data[val_test_split:].iloc[:, 0:-1].values.astype(np.float32)
    test_y = data[val_test_split:].iloc[:, -1:].values.astype(np.int32)

    transformer = StandardScaler()
    if normalize:
        train_x = transformer.fit_transform(train_x)
        validation_x = transformer.transform(validation_x)
        test_x = transformer.transform(test_x)

    print("Train shape: {}".format(train_x.shape))
    print("Validation shape: {}".format(validation_x.shape))
    print("Test shape: {}".format(test_x.shape))

    return train_x, train_y, validation_x, one_hot(
        validation_y), test_x, one_hot(test_y), transformer


def one_hot(y):
    if type(y) is pd.Series:
        y = y.values.reshape(-1, 1)
    num_classes = np.unique(y).shape[0]

    return np.squeeze(np.eye(num_classes)[y.reshape(-1).astype(int)]).astype(
        np.int32)


if __name__ == "__main__":

    path = os.getcwd()
    parent = '/'.join(path.split('/')[:-1])
    data_path = parent + "/data/trimmed-aggregated-training-data.csv"

    # train_x, train_y, validation_x, validation_y, test_x, test_y = load_data(
    #     parent + "/data/whole_week.pkl")

    # print(train_x.dtype)
    # print(train_y.dtype)
    g1, g2, g3, t = get_example_generator(data_path)

    for obs, label in g1:
        pass
    for obs, label in g2:
        pass
    for obs, label in g3:
        pass
        # print(obs)
        # input()
        # print(label)
        # input()
        # print(device)
        # input()
