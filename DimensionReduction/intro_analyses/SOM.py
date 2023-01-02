#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 14:26:04 2022

@author: rmarion
"""
import sys
import susi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
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
#Design SOM model
K_max = 20
grid_combos = np.array([[1]*(K_max-1), np.arange(2, K_max + 1)]).T
pca = PCA(n_components=10)
x_pca = pca.fit_transform(x)
results_df = pd.DataFrame(columns = ["n_rows", "n_cols", "K", "CH", "sil"])
results_df.n_rows = grid_combos[:, 0]
results_df.n_cols = grid_combos[:, 1]
results_df.K = results_df[["n_rows", "n_cols"]].product(axis = 1)
#Test several grid sizes with n_row = 1 to view K as a number of clusters
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
plt.show()
