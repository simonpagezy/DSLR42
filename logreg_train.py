# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 17:27:19 2018

@author: dimio
"""
import numpy as np
import pandas as pd
import sys

class LogisticRegressionOVR_train(object):
    def __init__(self, w=[], eta=5e-5, n_iter=30000):
        self.eta = eta
        self.n_iter = n_iter
        self.w = w
        
    def _scaling(self,X):
        for i in range(len(X)):
            X[i] = ( X[i] - X.mean())  / X.std()
        return X

        
    def _processing(self,hptrain):
        hptrain = hptrain.dropna()
        hp_features = np.array((hptrain.iloc[:,5:]))
        hp_labels = np.array(hptrain.loc[:,"Hogwarts House"])
        
        np.apply_along_axis(self._scaling, 0, hp_features)
        return hp_features, hp_labels

        
        
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    

    def fit(self, hptrain):
        X, y = self._processing(hptrain)
        X = np.insert(X, 0, 1, axis=1)
        m = X.shape[0]

        for i in np.unique(y):
            y_copy = np.where(y == i, 1, 0)
            w = np.ones(X.shape[1])

            for _ in range(self.n_iter):
                output = X.dot(w)
                errors = y_copy - self._sigmoid(output)
                gradient = np.dot(X.T, errors)
                w += self.eta * gradient

            self.w.append((w, i))
        return self.w
    
    
    
    def _predict_one(self, x):
        return max((x.dot(w), c) for w, c in self.w)[1]
    
    def predict(self, X):
        return [self._predict_one(i) for i in np.insert(X, 0, 1, axis=1)]
    
    
    
    def score(self,hptrain):
        X, y = self._processing(hptrain)
        return sum(self.predict(X) == y) / len(y)   
    



    
    
if __name__ == "__main__":
    hptrain = pd.read_csv(sys.argv[1], index_col = "Index")
    weights = LogisticRegressionOVR_train().fit(hptrain)
    np.save("weights", weights)
    print("Poids sauvegardés dans weights.npy, accuracy :", LogisticRegressionOVR_train().score(hptrain))