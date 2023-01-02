#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 11:55:35 2022

@author: rmarion
"""
import sys
import pandas as pd
import numpy as np
sys.path.append('..')
from utils import gaussian_smoothing
from pyts.decomposition import SingularSpectrumAnalysis

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
del df_sorted
del df

# Code maxes out RAM (allocation error due to too large array)
transformer = SingularSpectrumAnalysis(window_size=24)
x_new = transformer.transform(x)
