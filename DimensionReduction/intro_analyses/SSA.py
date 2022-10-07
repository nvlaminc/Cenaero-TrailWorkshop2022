#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 11:55:35 2022

@author: rmarion
"""


import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors

# Import data
file_path = '../../QuickStart/Data/Electricity/residential_all_hour_with_date_time.pkl'
x_date_time = pd.read_pickle(file_path)

# Smoothing
def gaussian_smoothing (df, std = 3, n_points = 25):
    df_smoothed = df.copy()
    df_smoothed = df_smoothed.rolling(n_points, min_periods = 1, center=True, win_type = "gaussian").mean(std = std)

    return df_smoothed

x_date_time = gaussian_smoothing(x_date_time)
x = np.array(x_date_time).T


# Take just the first 365 days
df_sorted = x_date_time.sort_values(by='date_time')
df = df_sorted["2009-07-15" : "2010-07-14"]
x = np.array(df).T

del df_sorted
del df


from pyts.decomposition import SingularSpectrumAnalysis
# Code maxes out RAM
# transformer = SingularSpectrumAnalysis(window_size=24)
# x_new = transformer.transform(x)
