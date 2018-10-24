import pandas as pd
import os

files = [file for file in os.listdir() if '2018' in file]

unique_id_sets = []

for file in files:
    df = pd.read_pickle(file)
    ids = set(list(df['DeviceID']))
    unique_id_sets += [ids]
    df = None

joint_set = unique_id_sets[0]

for s in unique_id_sets:
    print(len(s))
    joint_set = set(list(joint_set) + list(s))

print(len(joint_set))

