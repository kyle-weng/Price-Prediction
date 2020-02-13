"""Trying to get a better look at the data """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ecdf(data):
    """simple eCDF of array"""
    plt.plot(np.sort(data), np.linspace(0,1,len(data)))

def write(ids, p, name="pred.csv", d="pred/"):
    with open(d+name, "w") as f:
        f.write(pd.DataFrame({
            "id": ids,
            "Predicted":p
        }).to_csv(index=False))

def fillize(data):
    """Replace NaNs with zero, and adds indicator
    *only checks opened_position_qty"""
    nans = np.isnan(data["opened_position_qty"])
    return data.fillna(0).assign(nans=nans)

def deidize(data):
    """drop id column"""
    return data.drop("id", axis=1)

def relize(data):
    """replace bid[i] and ask[i] columns with
    bid[i]-bid[i-1] ans ask[i]-ask[i-1]"""
    rel = data.copy()
    for i in range(5,1,-1):
        rel[f"bid{i}"] = rel[f"bid{i-1}"] - rel[f"bid{i}"]
        rel[f"ask{i}"] -= rel[f"ask{i-1}"]

    rel["bid1"] = rel["last_price"] - rel["bid1"]
    rel["ask1"] -= rel["last_price"]
    return rel

def logize(data):
    """Replace the non-negative columns with log1p of them"""
    cols = sum([
        ["opened_position_qty", "closed_position_qty", "transacted_qty"],
        [f"bid{i}" for i in range(2,6)],
        [f"bid{i}vol" for i in range(1,6)],
        [f"ask{i}" for i in range(2,6)],
        [f"ask{i}vol" for i in range(1,6)],
    ], [])
    logged = data.copy()
    logged.loc[:,cols] = np.log1p(logged.loc[:,cols])
    return logged

import sklearn.model_selection
def splitize(data):
    """make train and validation set"""
    return sklearn.model_selection.train_test_split(data.drop("y",axis=1), data["y"])

import sklearn.preprocessing
def normall(datas):
    """standardize last_price, mid, and d_open_interest
    NOTE takes in list of dataframes to process
    """
    scaler_price = sklearn.preprocessing.StandardScaler()
    scaler_price.fit(datas[0][["last_price"]])
    scaler_interest = sklearn.preprocessing.StandardScaler()
    scaler_interest.fit(datas[0][["d_open_interest"]])

    normeds = []
    for data in datas:
        normed = data.copy()
        normed[["last_price","mid"]] = scaler_price.transform(normed[["last_price","mid"]])
        normed[["d_open_interest"]] = scaler_interest.transform(normed[["d_open_interest"]])
        normeds.append(normed)
    return normeds
