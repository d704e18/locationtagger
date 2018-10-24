import pandas as pd
import numpy as np

df = pd.read_csv(
    "aggregated-training-data.csv",
    parse_dates=True,
    dayfirst=False,
    index_col=0)

keys = [k for k in df.keys()]

for key in keys:
    values = df[key].values
    if (values == 0).all():
        df = df.drop(key, axis=1)

print(len(keys))
print(len([k for k in df.keys()]))

df.to_csv("trimmed-aggregated-training-data.csv")
