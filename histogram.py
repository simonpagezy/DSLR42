# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 19:05:33 2018

@author: Simon
"""

import pandas as pd
from describe import numeric_features
import matplotlib.pyplot as plt
import numpy as np
import sys

def preprocess(dataset):
    cols = numeric_features(dataset)
    for col in cols:
        dataset[col] = (dataset[col] - dataset[col].mean()) / dataset[col].std()
    return dataset[cols]

def get_grades(dataset, prep_dataset, house, topics):
    df = prep_dataset[dataset["Hogwarts House"] == house][topics]
    df.dropna(inplace=True)
    return df

def plot_hist(dataset, prep_dataset):
    for col in prep_dataset.columns:
        plt.figure()
        plt.hist(get_grades(dataset, prep_dataset, "Gryffindor", col), bins=25, alpha=0.5, label = 'Gry', color = 'r')
        plt.hist(get_grades(dataset, prep_dataset, "Ravenclaw", col), bins=25, alpha=0.5, label = 'Rav', color = 'b')
        plt.hist(get_grades(dataset, prep_dataset, "Slytherin", col), bins=25, alpha=0.5, label = 'Sly', color = 'g')
        plt.hist(get_grades(dataset, prep_dataset, "Hufflepuff", col), bins=25, alpha=0.5, label = 'Huf', color = 'y')
        plt.legend(loc = 'upper right')
        plt.title(col)
        plt.show()
    
if __name__ == "__main__":
    dataset = pd.read_csv(sys.argv[1], index_col = "Index")
    prep_dataset = preprocess(dataset)
    plot_hist(dataset, prep_dataset)
    
    