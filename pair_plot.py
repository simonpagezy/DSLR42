# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 19:06:20 2018

@author: Simon
"""

import pandas as pd
from describe import numeric_features
import matplotlib.pyplot as plt
import numpy as np
import sys
import seaborn as sns
sns.set(style="ticks", color_codes=True)

def preprocess(dataset):
    """
    Preprocess the training dataset: keep the interesting features and drop na
    Inputs:
        - dataset (pd.dataframe) training dataset of Harry Potter
    """
    cols = numeric_features(dataset)
    cols.remove("Defense Against the Dark Arts") # same as Astronomy
    cols.remove("Arithmancy") # homogenous repartition between houses
    cols.remove("Care of Magical Creatures") # homogenous repartition between houses
    cols = ["Hogwarts House"] + cols
    dataset = dataset[cols]
    dataset = dataset.dropna()
    return dataset

def scatter_plot(prep_dataset):
    """
    Pair plot on the remaining features of the training set
    Input:
        -prep_dataset (pd.dataframe)
    """
    sns.pairplot(prep_dataset, hue="Hogwarts House", markers = ".", size=2)
    plt.show()
    

    
if __name__ == "__main__":
    dataset = pd.read_csv(sys.argv[1], index_col = "Index")
    prep_dataset = preprocess(dataset)
    scatter_plot(prep_dataset)
    



