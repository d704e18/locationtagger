import pandas as pd

df = None
store = pd.HDFStore("spooktober.af1-data.h5")
keys = [k for k in store.keys()]

for key in keys:
    if df is None:
        df = store[key]
    else:
        df = df.append(store[key])

df.to_csv('lolfixed.csv')
