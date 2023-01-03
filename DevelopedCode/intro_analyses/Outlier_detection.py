#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 10:14:02 2022

@author: rmarion
"""
import sys
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
sys.path.append('..')
from utils import gaussian_smoothing

# Import data
file_path = '../../QuickStart/Data/Electricity/residential_all_hour_with_date_time.pkl'
x_date_time = pd.read_pickle(file_path)

# Smoothing
x_date_time = gaussian_smoothing(x_date_time)
x = np.array(x_date_time).T

# Take just the first 365 days
df_sorted = x_date_time.sort_values(by='date_time')
df = df_sorted["2009-07-15" : "2010-07-14"]
x = np.array(df).T

# Detect outliers with Isolation Forest
outliers_fraction = 0.01
clf = IsolationForest(random_state=0, contamination = outliers_fraction).fit(x)
pred = clf.predict(x)

# Plots
plt.plot(x[pred == 1].T, c = "b", alpha = 0.7)
plt.plot(x[pred == -1].T, c = "r", alpha = 0.7)
plt.show()
x_pca = PCA(n_components=2).fit_transform(x)
out = pred == -1
non_out = pred == 1
plt.scatter(x_pca[non_out, 0], x_pca[non_out, 1], c="r", label = "non outliers")
plt.scatter(x_pca[out, 0], x_pca[out, 1], c= "b", label = "outliers")
plt.legend()
plt.show()
scores = clf.decision_function(x)
plt.hist(scores)

# final dataset
x_clean = x[non_out, :]
