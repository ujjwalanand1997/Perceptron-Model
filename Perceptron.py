#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 20:12:04 2018

@author: ujjwal
"""

import numpy as np
class Perceptron(Object):
    def __init__(self,l_rate = 0.01, n_iter = 10):
        self.l_rate = l_rate
        self.n_iter = n_iter
        
    def fit(self,X,y):
        self.w_ = np.zeros(1+X.shape[1])
        self.errors = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.l_rate * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)