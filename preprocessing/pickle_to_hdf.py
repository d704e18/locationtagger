"""
This script was used to generate /data/af1-data.h5
I don't expect it will be used again
"""

import pandas as pd
from datetime import datetime
import os

project_root = os.path.dirname(
    os.path.abspath(__file__)) + '/' + os.pardir + '/'
data_dir = project_root + 'data/'

labels = pd.read_csv(data_dir + 'af1-labels.csv')
af1_sensors = pd.read_csv(data_dir + 'af1-sensors.csv')

valid_devices = set(labels["BLUETOOTHADDRESS"])
# Only include devices we have labels for
valid_sensors = set(
    af1_sensors['ID'])  # Only include readings from AF1 sensors
# Use a dictionary for fast lookup
valid_devices = dict(zip(valid_devices, range(len(valid_devices))))

files = [
    data_dir + f for f in os.listdir(data_dir) if '.pkl' in f and '2018' in f
]
files.sort()  # sort by date

path = data_dir + 'af1-data.h5'

# if os.path.exists(path):
# os.remove(path)

store = pd.HDFStore(path)  # make sure we don't append to some existing files
large_df = None

print('Writing files to hdf {}:'.format(path))


def swap_day_month(x):
    y = x.year
    mo = x.day
    d = x.month
    h = x.hour
    mi = x.minute
    s = x.second

    x = datetime(year=y, month=mo, day=d, hour=h, minute=mi, second=s)

    return x


for file in files:
    print(file)

for file in files:  # Skip first file as it is alreayd added
    df = pd.read_pickle(file)
    # groups = [values for (key, values) in df.groupby('DeviceID')]
    # for g in groups:
    groups = [(key, group) for (key, group) in df.groupby('DeviceID')
              if key in valid_devices]
    # groups = [(key, group) for (key, group) in groups

    print()
    print('Grouped df {}'.format(file))
    print()
    i = 0
    n = len(groups)
    for key, group in groups:
        group = group[group['SensorID'].isin(valid_sensors)]
        group.index = pd.DatetimeIndex(group.index.map(swap_day_month))
        # Filter out any sensor not in af1
        # store.append(str(key), group)
        if large_df is None:
            large_df = group
        else:
            large_df = large_df.append(group)
        i += 1
        print("Wrote group {} of {} ({})".format(i, n, key))
    df = None  # Garbo
    groups = None
    print('Wrote {}'.format(file))

store.close()
large_df.sort_index(inplace=True)
large_df.to_csv(data_dir + "large_df.csv")
