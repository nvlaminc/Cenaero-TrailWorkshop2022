#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 14:26:04 2022

@author: rmarion
"""

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from somlearn import SOM
import susi
from susi.SOMPlots import plot_nbh_dist_weight_matrix, plot_umatrix
from itertools import permutations 
from sklearn.metrics import silhouette_score, calinski_harabasz_score

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


K_max = 20
grid_combos = np.array([[1]*(K_max-1), np.arange(2, K_max + 1)]).T

pca = PCA(n_components=10)
x_pca = pca.fit_transform(x)
results_df = pd.DataFrame(columns = ["n_rows", "n_cols", "K", "CH", "sil"])
results_df.n_rows = grid_combos[:, 0]
results_df.n_cols = grid_combos[:, 1]
results_df.K = results_df[["n_rows", "n_cols"]].product(axis = 1)

for row_index in range(results_df.shape[0]):
    
    grid_vals = results_df.loc[row_index, ["n_rows", "n_cols"]]
    som = susi.SOMClustering(n_rows=grid_vals[0],n_columns=grid_vals[1])
    som.fit(x_pca)
    cluster_coord = pd.DataFrame(np.array(som.get_clusters(x_pca)), columns = ["dim1", "dim2"])
    clusters = cluster_coord.groupby(['dim1', 'dim2']).grouper.group_info[0]

    CH = calinski_harabasz_score(x, clusters)
    sil = silhouette_score(x, clusters)
    results_df.loc[row_index, "CH"] = CH
    results_df.loc[row_index, "sil"] = sil
    print(row_index)
    

results_df.plot.line(x = "K", y = "CH")
results_df.plot.line(x = "K", y = "sil")
