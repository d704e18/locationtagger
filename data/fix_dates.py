import os
import warnings
from datetime import datetime

import pandas as pd

warnings.filterwarnings('ignore')

store = pd.HDFStore("af1-data.h5")
newStore = pd.HDFStore("af1-fixed-dates.h5")
print("Getting keys..")
keys = set(pd.read_csv('af1-labels.csv')['BLUETOOTHADDRESS'])

# keys = store.keys()


def swap_day_month(x):
    y = x.year
    mo = x.day
    d = x.month
    h = x.hour
    mi = x.minute
    s = x.second

    x = datetime(year=y, month=mo, day=d, hour=h, minute=mi, second=s)

    return x


num_keys = len(keys)

for i, key in enumerate(keys):
    key = str(key)
    if key not in store:
        continue
    df = store[key]
    # print(df)
    df.index = pd.DatetimeIndex(df.index.map(swap_day_month))
    # print(df)
    # input()
    newStore[key] = df
    print("Fixed key {} of {}.".format(i + 1, num_keys))
