import os
import pandas as pd

pickles = [f for f in os.listdir() if '.pkl' in f and '201809' in f]

for pickle in pickles:
    newname = pickle.split('.')[0]
    df = pd.read_pickle(pickle)
    df.to_csv(newname + '.csv')
    print('Finished', pickle)
