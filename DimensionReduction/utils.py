import os 
import glob
import time
import darts
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.fft import fft
from scipy import signal
from tqdm import tqdm
from torchmetrics import MeanSquaredError
from pytorch_lightning.callbacks import EarlyStopping

#### Darts Modules #####
from darts.models import RNNModel
from darts.metrics import rmse, coefficient_of_variation

#### Scikilearn Modules ####
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, calinski_harabasz_score

import susi

plt.style.use('classic')

def gen_array(path):
    df = pd.read_pickle(path)
    ids = df['ID'].values
    # Extract only consumption values from dataframe
    df = df['consumption'].values
    # Duration
    n_t = (ids==np.unique(ids)[0]).sum()
    # Number of different IDS
    n_s = len(np.unique(ids))
    x = [df[i*n_t: (i+1)*n_t] for i in range(n_s)]
    return np.asarray(x)

def gen_dwt(sig, width= np.arange(1,25), time_basis = 24):
    dwt = []
    # mean per participant for each week
    s = sig[:, :7*time_basis*(sig.shape[-1]//(time_basis*7))].reshape(len(sig), -1, 7, time_basis).mean(1)
    for i in tqdm(range(len(sig))):
        # generate the discrete wavelet tranform + downsampling
        cwt = signal.cwt(s[i].flatten(), signal.ricker, width)[:, ::2]
        dwt.append(cwt)
    return np.asarray(dwt)

def piecewise_approx(sig, window=20, overlapping=True):
    sig_apca = []
    for t in range(len(sig)//window):
        for w in range(window):
            #sig_apca.append(sig[t*window:(t+1)*window].mean())
            sig_apca.append(np.median(sig[t*window:(t+1)*window]))
    return np.asarray(sig_apca)

def gen_apca(sig, window):
    apca = []
    for i in tqdm(range(len(sig))):
        apca.append(piecewise_approx(sig[i], window))
    return np.asarray(apca)

def plot_cluster(dataset, cluster, cluster_id=0, period='day', time_basis = 24):
    fig = plt.figure()
    population = (cluster == cluster_id).sum()
    if period == 'day':
        # Reshape per days
        x = dataset[cluster==cluster_id].reshape(population, -1, time_basis)
        for i in range(population):
            # plot the mean consumption per day for each element
            #of the population of the cluster.
            plt.plot(x[i].mean(0), c='black', linewidth=0.5, alpha=0.5)
        plt.plot(x.mean(1).mean(0))
        plt.suptitle(f'Daily Energy Consumption Profile of the Cluster {cluster_id+1}')
    elif period == 'week':
        # Reshape per weeks remove last days to have a multiple
        #of a week (i.e. 7 days).
        x = dataset[cluster==cluster_id][:, :7*time_basis*(dataset.shape[-1]//(time_basis*7))].reshape(population, -1, 7, time_basis)
        for i in range(population):
            plt.plot(x[i].mean(0).flatten(), c='black', linewidth=0.25, alpha=0.75)
        plt.plot(x.mean(1).mean(0).flatten())
        plt.suptitle(f'Weekly Energy Consumption Profile of the Cluster {cluster_id+1}')    
    ax = plt.gca()
    ax.set_ylim([0, 4])
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Electrical Consumption')
    plt.show()
    
def plot_clusters(dataset, cluster, time_basis = 24):
    n_cluster = len(np.unique(cluster))
    fig, ax = plt.subplots(n_cluster, 2, figsize=(10, 20))
    for j in range(n_cluster):
        sig = dataset[cluster==j].reshape((cluster==j).sum(), -1, time_basis)
        for i in range((cluster==j).sum()):
            ax[j][0].plot(sig[i].mean(0), c='black', linewidth=0.5, alpha=0.5)
        ax[j][0].grid()
        ax[j][0].plot(sig.mean(0).mean(0))
        ax[j][0].set_ylim([0, 5])
        sig = dataset[cluster==j][:, :7*time_basis*(dataset.shape[-1]//(time_basis*7))].reshape((cluster==j).sum(), -1, 7, time_basis)
        for i in range((cluster==j).sum()):
            ax[j][1].plot(sig[i].mean(0).flatten(), c='black', linewidth=0.5, alpha=0.5)
        ax[j][1].grid()
        ax[j][1].plot(sig.mean(0).mean(0).flatten())
        ax[j][1].set_ylim([0, 5])
    ax[j][0].set_xlabel('Time [h]')
    ax[j][1].set_xlabel('Time [h]')
    plt.suptitle(f'Visualisation of the {n_cluster} clusters profile', y=0.915)
    plt.show()
       
def rmse_var(x, x_hat):
    return np.sqrt( np.mean((x - x_hat)**2)) / np.mean(x)

def mse_cluster(sig, cluster):
    mse = 0
    for i in tqdm(range(len(sig))):
        
        mse += rmse_var(sig[i], sig[cluster==cluster[i]].mean(0))
    return mse/len(sig)

def seasons_reshape(x):
    l_period = 24*365//4
    b_id = 1758
    season = np.zeros((x.shape[0], 4, l_period))
    season[:, 0, l_period-b_id:] = 0.5*(x[:, :b_id] + x[:, 8328-b_id:8328])
    season[:, 0, :l_period-b_id] = x[:, 6138:6138+l_period-b_id]
    season[:, 1] = 0.5*(x[:, b_id:b_id+l_period] + x[:, b_id+4*l_period:b_id+5*l_period])
    season[:, 2] =  x[:, b_id+l_period:b_id+2*l_period]
    season[:, 3] =  x[:, b_id+2*l_period:b_id+3*l_period]
    season = season[:, :, :7*24*13].reshape((season.shape[0], season.shape[1], -1, 7, 24))
    return season.mean(2).reshape(season.shape[0], -1)

def gaussian_smoothing (df, std = 3, n_points = 25):
    df_smoothed = df.copy()
    df_smoothed = df_smoothed.rolling(n_points, min_periods = 1, center=True, win_type = "gaussian").mean(std = std)
    return df_smoothed

def compute_intersect(series_a, series_b, intersect=True):
    series_a_common = series_a.slice_intersect(series_b) if intersect else series_a
    series_b_common = series_b.slice_intersect(series_a) if intersect else series_b
    return series_a_common, series_b_common
    
def biased_error(y, y_):
    return 100*np.mean(y_ - y)/np.mean(y)

def gen_cluster(x, labels, ids, method):
    clusters = []
    for c in range(len(np.unique(labels))):
        print(f'Population of cluster {c+1} for {method} is {(labels==c).sum()}')
        tmp = {}
        tmp['labels'] = c
        tmp['centroid'] = x[labels == c].mean(0)
        tmp['ids'] = ids[labels == c]
        clusters.append(tmp)
    return clusters

def gen_train_val_series(c, cluster_list, x_date_time, do_plot=False):
    df = pd.DataFrame()
    df['consumption'] = cluster_list[c]['centroid']
    df['date_time'] = pd.date_range(start="2009-07-15", end="2010-07-14 23:00:00", freq="60T", name='date_time')
    df = df.set_index('date_time')
    train_serie = darts.TimeSeries.from_dataframe(df)
    if do_plot :
        train_serie.plot()

    conso = x_date_time.sort_values(by='date_time')["2010-07-14" : "2010-07-21"][cluster_list[c]['ids']].values.mean(1)
    df = pd.DataFrame()
    df['date_time'] = pd.date_range(start="2010-07-14", end="2010-07-21 23:00:00", freq="60T", name='date_time')
    df['consumption'] = conso
    df = df.set_index('date_time')
    val_series = darts.TimeSeries.from_dataframe(df)
    if do_plot:
        val_series.plot()
    return train_serie, val_series
