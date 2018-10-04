# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 14:37:43 2018

@author: Simon
"""
import pandas as pd
import numpy as np
import math

# Functions
def numeric_features(dataset):
    """
    Input:
        dataset(pd.Dataframe)
    Output:
        (list of str) names of the numeric features
    """
    numeric_col = []
    for col_name in dataset.columns:
        try:
            float(dataset[col_name][0])    
            numeric_col.append(col_name)
        except ValueError:
            continue
    return numeric_col

def create_result_df(features, add_info):
    """
    Input:
        features(list of str)
        add_info(list of str) 
    Outputs:
        create a dataset with features in column and add_info in index
    """
    info = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    if add_info == list():
        info = info + add_info
    df = pd.DataFrame(np.zeros(shape=(len(info), len(features))), index = info, columns = features)
    return df

def manual_std(result_df, dataset):
    """
    Compute manualy the standard deviation
    """
    for col_name in result_df.columns:
        centered_square = 0
        for val in dataset[col_name]:
            if np.isnan(val):
                continue
            else:
                centered_square += (val - result_df[col_name]["Mean"])**2
        result_df[col_name]["Std"] = np.sqrt(centered_square/result_df[col_name]["Count"])
    return result_df

def manual_describe(dataset, add_info = None):
    """
    Enter a dataset and it will output information on the numerical features of
    your dataset.
    """
    result_df = create_result_df(numeric_features(dataset), add_info)
    for col_name in result_df.columns:
        n = 0
        total = 0
        _max = - np.inf
        _min = np.inf
        ordered_list = []
        for val in dataset[col_name]:
            if np.isnan(val):
                continue
            else:
                ordered_list.append(val)
                n += 1
                total = total + val
                if _max < val:
                    _max = val
                if _min > val:
                    _min = val
        ordered_list.sort()
        result_df[col_name]["Count"] = n
        result_df[col_name]["Mean"] = total/n
        result_df[col_name]["Max"] = _max
        result_df[col_name]["Min"] = _min
        result_df = manual_std(result_df, dataset)
        result_df[col_name]["25%"] = ordered_list[math.ceil(n/4) - 1]
        result_df[col_name]["50%"] = ordered_list[math.ceil(n/2) - 1] 
        result_df[col_name]["75%"] = ordered_list[math.ceil(3*n/4) - 1] 
    return result_df

# example
dataset = pd.read_csv("resources/dataset_train.csv", index_col = "Index")
result_df = manual_describe(dataset)



