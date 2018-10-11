# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 19:05:55 2018

@author: Simon
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys


def preprocess(dataset):
    """
    Preprocess the training dataset: keep the two similar features and drop na
    Inputs:
        - dataset (pd.dataframe) training dataset of Harry Potter
    """
    cols = ["Astronomy", "Defense Against the Dark Arts"]
    dataset = dataset[cols]
    dataset = dataset.dropna()
    return dataset


def scatter_plot(prep_dataset):
    """
    Plot the two similar features of the dataset.
    Input:
        -prep_dataset (pd.dataframe)
    """
    plt.figure()
    plt.scatter(prep_dataset['Astronomy'], prep_dataset['Defense Against the Dark Arts'], label = 'students')
    plt.legend()
    plt.title("correlated features")
    plt.xlabel("Astronomy")
    plt.ylabel("Defense Against the Dark Arts")
    plt.show()
    
if __name__ == "__main__":
    dataset = pd.read_csv(sys.argv[1], index_col = "Index")
    prep_dataset = preprocess(dataset)
    scatter_plot(prep_dataset)
