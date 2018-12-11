import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import itertools


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


def add_timedelta(df, tiers=None):
    """
    Adds a one-hot encoding of the relative time difference between
    observations. Assumes the df is indexed on DateTime, and the index is
    ordered.

    """
    if tiers is None:
        tiers = [0, 30, 120, 300, 600]  #, 1200, 1800]
    # The time delta for the first value should be zero
    time_deltas = [0]
    previous_index = df.index[0]
    num_seconds = []
    for index in df.index[1:]:
        time_delta = index - previous_index
        num_seconds += [time_delta]
        tier = 0
        for v in tiers:
            if time_delta.seconds < v:
                break
            tier += 1
        previous_index = index
        time_deltas += [tier]

    # new_features = one_hot(np.asarray(time_deltas), num_classes=len(tiers) + 1)
    # new_columns = ["Within {} seconds".format(v) for v in tiers]
    # new_columns += ["More than {} seconds".format(tiers[-1])]
    # feature_df = pd.DataFrame(
    #     data=new_features, columns=new_columns, index=df.index)
    # df = pd.concat([df, feature_df], axis=1)

    return df, num_seconds


def add_time_of_day(df, chunkSize=3):
    """
    Adds a one-hot encoding of the time of day of each observation.
    Assumes the df is indexed on DateTime
    chunkSize: Specifies how many hours to consolidate.
    e.g. chunkSize=3 means that 00:00 -> 02:59 is consolidated
                                03:00 -> 05:59 is consolidated
    """
    chunkSize = int(chunkSize)
    hours = [v.hour // chunkSize for v in df.index]
    columns = np.sort(np.unique(hours)) * chunkSize
    new_features = one_hot(np.asarray(hours))
    feature_df = pd.DataFrame(
        data=new_features,
        columns=[
            "About {}:00".format('0' + str(h) if h < 10 else h)
            for h in columns
        ],
        index=df.index)
    df = pd.concat([df, feature_df], axis=1)

    return df


def pad_data(readings, labels, clip):
    # Compute the number of rows of padding to add
    padding = clip - len(readings)
    zeros = np.zeros((padding, readings.shape[1]))
    readings = np.concatenate((np.asarray(readings), zeros), axis=0)
    dummyLabels = np.zeros((padding, labels.shape[1]))
    labels = np.concatenate((labels, dummyLabels), axis=0)

    return readings, labels


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


def load_data_cool(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(
            file_path, dayfirst=False, parse_dates=True, index_col=0)
    elif file_path.endswith('.pkl'):
        df = pd.read_pickle(file_path)
    else:
        raise ValueError("Unknown filetype", file_path)

    df = df.sort_index()
    return df


def one_hot_areas(df):
    areas = df.iloc[:, -1]
    hot_areas = one_hot(areas)
    m, n = hot_areas.shape
    area_names = []
    for i in range(n):
        area_name = 'area{}'.format(i)
        area_names += [area_name]
        df[area_name] = hot_areas[:, i]

    df = df.drop('areas', axis=1)

    return df, area_names


def get_example_generator(
        file_path,
        clip=20,
        pad=True,
        time_resolution=1,
        time_differences=[30, 60, 120, 180, 240, 300, 600, 900, 1200],
        filter_staff=False):
    """
    file_path:        path to input data
    clip:             accept at most `clip` inputs from each passenger
    pad:              if True, pad the data with 0's to match sequence sizes
    time_resolution:  The resolution of the onehot encoding of time in the data.
                        Will add 24 // n columns to the data.
    time_differences: The time intervals used in the encoding of the relative
                        time difference between observations within a sequence.
                        Will add |n|+1 columns to the data.

    Returns three generators, and a transformer:
    train_gen:        contains all the data not used for testing
    val_gen:          the first half of the validation data
    test_gen:         the second half of the validation data
    transformer:      the transformer that was used to scale the data

    """
    validation_start = '2018-09-09 18:00'
    validation_end = '2018-09-10 18:00'

    df = load_data_cool(file_path)

    if filter_staff:
        staff_dict = get_staff_dict()
    else:
        staff_dict = {}

    df, area_names = one_hot_areas(df)

    # Drop the columns that should not be scaled
    unscalable_columns = ['Device'] + area_names
    df, transformer = scale_df(df, unscalable_columns, validation_start,
                               validation_end)
    df = add_time_of_day(df, chunkSize=time_resolution)

    # Train on everything outside of the validation interval
    train_df = df[:validation_start].append(df[validation_end:])
    validation_df = df[validation_start:validation_end]
    # Split validation into test and validation
    test_start = validation_df.shape[0] // 2
    test_df = validation_df.iloc[test_start:]
    validation_df = validation_df.iloc[:test_start]

    def get_generator(df):
        groups = df.groupby('Device')
        for device, group in groups:
            if device in staff_dict:
                continue
            # The sequence can cause errors if it is this small, and it is
            # useless anyways.
            # if len(group) < 3:
            #     continue
            # if clip is not None:
            #     group = group[:clip]
            person = group.drop('Device', axis=1)
            labels = person[area_names]
            readings = person.drop(area_names, axis=1)
            readings, time_deltas = add_timedelta(
                readings, tiers=time_differences)

            readings = np.asarray(readings)
            labels = np.asarray(labels)

            yield readings, time_deltas  # , device

    return get_generator(train_df), get_generator(
        validation_df), get_generator(test_df), transformer


def get_rnn_data(file_path, **kwargs):
    def unzip(gen):
        x, y = zip(*list(gen))
        return np.asarray(x), np.asarray(y)

    train, val, test, t = get_example_generator(file_path, **kwargs)
    return unzip(train), unzip(val), unzip(test), t


def remove_staff(df, key="Device"):
    staff_dict = get_staff_dict()


def get_staff_dict():
    af1_staff_path = os.path.dirname(
        os.path.abspath(__file__)) + "/../data/af1-measurements.csv"
    af1 = pd.read_csv(af1_staff_path, index_col=None)
    staff_dict = {}

    for i, data in af1.iterrows():
        ID, _, _, _, _, _, _, kind = data
        if kind == "staff":
            staff_dict[ID] = True

    return staff_dict


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


def one_hot(y, num_classes=None):
    if type(y) is pd.Series:
        y = y.values.reshape(-1, 1)
    if num_classes is None:
        num_classes = np.unique(y).shape[0]

    return np.squeeze(np.eye(num_classes)[y.reshape(-1).astype(int)]).astype(
        np.int32)


if __name__ == "__main__":

    data_path = os.path.dirname(os.path.abspath(
        __file__)) + "/../data/trimmed-aggregated-training-data.csv"

    # train_x, train_y, validation_x, validation_y, test_x, test_y = load_data(
    #     parent + "/data/whole_week.pkl")

    # print(train_x.dtype)
    # print(train_y.dtype)
    # train, val, test, t = get_example_generator(data_path)
    # get_example_generator(data_path, clip=21, pad=True, time_resolution=2,
    #         time_differences=[10, 20, 30])
    # return unzip(train), unzip(val), unzip(test), t
    def timedelta_to_seconds(dt):
        # days = dt.days
        # hours = 24 * days + dt.hours
        # minutes = 60*hours + dt.minutes
        # seconds = minutes * 60 + dt.seconds
        return dt / np.timedelta64(1, 's')

        # return seconds

    train, val, test, _ = get_rnn_data(
        data_path,
        clip=21,
        pad=True,
        time_resolution=2,
        time_differences=[10, 20, 30],
        filter_staff=False)

    print("Not filtered:\m")
    print(train[1])
    second_list = []
    for t in itertools.chain(train[1], val[1], test[1]):
        # t is a list of timedeltas
        t_seconds = map(timedelta_to_seconds, t) # t_seconds is a list of seconds
        second_list += t_seconds

    np.savetxt("with-staff-seconds.out", np.asarray(second_list))
