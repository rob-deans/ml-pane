import pandas as pd
import numpy as np


def get_all_data():

    df = pd.read_csv('FresnoClean.csv')

    max_ = df.max()
    min_ = df.min()

    def normalise(row):
        for i, coloumn in enumerate(df):
            row[coloumn] = np.interp(row[coloumn], [min_[i], max_[i]], [0, 1])
        return row

    df = df.apply(normalise, axis=1)
    data = []
    for r in df.iterrows():
        data.append(r[1].values.tolist())
    return np.array(data), max_, min_


def denormalise(val, maxi, mini):
    return (val * (maxi - mini)) + mini
