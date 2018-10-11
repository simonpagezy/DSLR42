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
    """
    Preprocess the training dataset: keep the numeric columns and scale
    features.
    Inputs:
        - dataset (pd.dataframe) training dataset of Harry Potter
    """
    cols = numeric_features(dataset)
    for col in cols:
        dataset[col] = (dataset[col] - dataset[col].mean()) / dataset[col].std()
    return dataset[cols]

def get_grades(dataset, prep_dataset, house, topic):
    """
    Get a datafram with the grades of the student from a specific house in a 
    specific topic.
    Inputs:
        - dataset (pd.dataframe) training dataset of Harry Potter
        - prep_dataset (pd.dataframe) preprocessed dataset of Harry Potter
        - house (str) name of the House you want to focus on
        - topic (str) name of the Topic you want to focus on
    """
    df = prep_dataset[dataset["Hogwarts House"] == house][topic]
    df.dropna(inplace=True)
    return df

def plot_hist(dataset, prep_dataset):
    """
    Plot one histogram per topic to see the repartition of grades of the houses
    Inputs:
        - dataset(pd.dataframe) training dataset of HP
        - prep_dataset(pd.dataframe) preprocessed dataset of HP
    """
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
    
    