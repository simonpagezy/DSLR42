# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 19:05:33 2018

@author: Simon
"""

import pandas as pd
from describe import manual_describe, numeric_features
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('resources/dataset_train.csv', index_col = "Index")
result_df = manual_describe(dataset)


def preprocess(dataset):
    cols = numeric_features(dataset)
    for col in cols:
        dataset[col] = (dataset[col] - dataset[col].mean()) / dataset[col].std()
    return dataset[cols]


def get_grades(prep_dataset, house, topics):
    df = prep_dataset[prep_dataset["Hogwarts House"] == house][topics]
    return df

def plot_hist(dataset):
    for col in numeric_features(dataset):
        plt.hist(get_grades(dataset, "Gryffindor", col), bins=25, alpha=0.5, label = 'Gry', color = 'y')
        plt.legend(loc = 'upper right')
        plt.title(col)
        plt.show()
    
        
        
        





#for col in result_df.columns:
#     grouped = dataset.groupby("Hogwarts House", axis = 0)[col].mean()
#
#
#
#def create_df(dataset):
#    """
#    Create a matrix to stock the needed result. 
#    """
#    cols = numeric_features(dataset)
#    info = dataset["Hogwarts House"].unique()
#    result_df = pd.DataFrame(np.zeros(shape=(len(info), len(cols))), index = info, columns = cols)
#    return result_df
#
#def fill_df(dataset, result_df):
#    """
#    Fill in df with the needed results (mean per topic grouped by house)
#    Normalize the results
#    """
#    for col in result_df.columns:
#        grouped = dataset.groupby("Hogwarts House", axis=0)[col].mean()
#        center_val = (grouped.values - np.mean(grouped.values))/np.std(grouped.values)
#        result_df[col] = center_val
#    return result_df