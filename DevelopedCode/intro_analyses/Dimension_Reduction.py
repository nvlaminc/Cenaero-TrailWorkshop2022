import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
sys.path.append('..')
from utils import *
from scipy.fft import fft
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA

if __name__ == "__main__":

    ###Initialize variables
    time_basis = 48
    n_cluster = 4

    ###Load data
    """
    #To use new data format by hour (time_basis = 24 in plot_cluster)
    file_path = '../../QuickStart/Data/Electricity/residential_all_hour_with_date_time.pkl'
    x_date_time = pd.read_pickle(file_path)
    ids = x_date_time.columns.values
    df = x_date_time.sort_values(by='date_time')["2009-07-15" : "2010-12-31"]
    x = np.array(df).T
    """
    #To use old data format by 30min (time_basis = 48 in plot_cluster)
    file_path = '../../QuickStart/Data/Electricity/residential_all.pkl'
    df = pd.read_pickle(file_path)
    val = df['consumption'].values
    ids = df['ID'].values
    val_reshape = []
    for i in tqdm(range(3639)):
        val_reshape.append(val[i*25728: (i+1)*25728])
    x = np.array(val_reshape)

    ### Feature Extraction ###

    # 0. No Feature Extraction
    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(x)
    # Energy Consumption Profile for cluster_id = 0 and periods of 1 day and 1 week
    plot_cluster(x, kmeans.predict(x), cluster_id=0, period='day', time_basis = 48)
    plot_cluster(x, kmeans.predict(x), cluster_id=0, period='week', time_basis = 48)
    #visualisation in 2D T-SNE manifold for ease
    tsne = TSNE(n_components=2, n_jobs=-1).fit_transform(x)
    fig = plt.figure('T-SNE visualisation of KMeans Clustering')
    fig = plt.suptitle('Visualisation of KMeans Clustering on Original Signals')
    plt.scatter(tsne[:, 0], tsne[:, 1], c=kmeans.predict(x), edgecolors = "face")
    plt.show()

    # 1. Principal Component Analysis
    # Investigate the number of component to consider by checking at the explained
    #variance ratio
    pca = PCA(n_components=1000)
    pca.fit(x)
    fig = plt.figure('pca')
    plt.plot(1-pca.explained_variance_ratio_[:25])
    plt.grid()
    plt.suptitle('Cumulated Explained Variance Ratio per component') 
    plt.xlabel('# Components')
    plt.ylabel('1 - Explained Variance Ratio')
    plt.show()
    pca = PCA(n_components=10)
    x_pca = pca.fit_transform(x)
    kmeans_pca = KMeans(n_clusters=n_cluster, random_state=0).fit(x_pca)
    #visualisation of the clustered space in the two first dimensions
    fig = plt.figure('KMeans Clustering on PCA Representation')
    fig = plt.suptitle('Visualisation of KMeans Clustering on PCA Representation') 
    plt.scatter(x_pca[:, 0], x_pca[:, 1], c=kmeans_pca.predict(x_pca), edgecolors = "face")
    plt.show()
    # Energy Consumption Profile for cluster_id = 0 and periods of 1 day and 1 week
    plot_cluster(x, kmeans_pca.predict(x_pca), cluster_id=0, period='day', time_basis = 48)
    plot_cluster(x, kmeans_pca.predict(x_pca), cluster_id=0, period='week', time_basis = 48)    
    
    
    # 2. Multidimensional Scaling (MDS)
    mds = MDS(n_components=14, n_jobs=-1, verbose=2)
    x_mds = mds.fit_transform(x)
    kmeans_mds = KMeans(n_clusters=n_cluster, random_state=0).fit(x_mds)
    #visualisation of the clustered space in the two first dimension
    fig = plt.figure('KMeans Clustering on MDS Representation')
    plt.scatter(x_mds[:, 0], x_mds[:, 1], c=kmeans_mds.predict(x_mds), edgecolors = "face")
    plt.suptitle('Visualisation of KMeans Clustering on MDS Representation')
    plt.show()
    # Energy Consumption Profile for cluster_id = 0 and periods of 1 day and 1 week
    plot_cluster(x, kmeans_mds.predict(x_mds), cluster_id=0, period='day', time_basis = 48)
    plot_cluster(x, kmeans_mds.predict(x_mds), cluster_id=0, period='week', time_basis = 48) 
    
    # 3. Fast Fourier Transform
    x_fft = np.abs(fft(x, n=100))
    kmeans_fft = KMeans(n_clusters=n_cluster, random_state=0).fit(x_fft)
    #visualisation of the signal in the Frequency domain
    fig = plt.figure('Frequency')
    plt.suptitle('Visualisation of the dataset in the Frequency domain')
    plt.plot(np.transpose(x_fft), c='black', linewidth=0.5, alpha=0.5)
    plt.plot(x_fft.mean(0))
    plt.grid()
    plt.show()
    # Energy Consumption Profile for cluster_id = 0 and periods of 1 day and 1 week
    plot_cluster(x, kmeans_fft.predict(x_fft), cluster_id=0, period='day', time_basis = 48)
    plot_cluster(x, kmeans_fft.predict(x_fft), cluster_id=0, period='week', time_basis = 48)    

    # 4. Discrete Wavelet Transform 
    dwt = gen_dwt(x, time_basis = 48)
    x_dwt = dwt.reshape(len(dwt), -1)
    kmeans_dwt = KMeans(n_clusters=n_cluster, random_state=0).fit(x_dwt)
    #visualation of the mean DWT vs. random DWT
    fig = plt.figure()
    plt.imshow(dwt.mean(0), cmap='binary')
    plt.suptitle('Mean Discrete Wavelet Transform')
    fig = plt.figure()
    rdm_id = np.random.randint(0, len(x))
    plt.imshow(dwt[rdm_id], cmap='binary')
    plt.suptitle(f'Discrete Wavelet Transform of {rdm_id} ID')
    # Energy Consumption Profile for cluster_id = 0 and periods of 1 day and 1 week
    plot_cluster(x, kmeans_dwt.predict(x_dwt), cluster_id=0, period='day', time_basis = 48)
    plot_cluster(x, kmeans_dwt.predict(x_dwt), cluster_id=0, period='week', time_basis = 48)
    
    # 5.Piecewise Constant Approximation --- Tests not complete (slow) !! 
    window = 20
    x_apca = gen_apca(x, window=window)
    x_apca = x_apca[:, ::window]
    kmeans_apca = KMeans(n_clusters=n_cluster, random_state=0).fit(x_apca)
    #visualisation of the APCA 
    fig = plt.figure()
    rdm_id = np.random.randint(0, len(x))
    plt.suptitle(f'Piecewise Constant Approximation of Signal {rdm_id}')
    plt.grid()
    plt.plot(x[rdm_id, 100:2500], label='Original Signal')
    plt.plot(x_apca[rdm_id, 100:2500], label='APCA Reduced Signal')
    plt.legend()
    plt.show()
    # Energy Consumption Profile for cluster_id = 0 and periods of 1 day and 1 week
    plot_cluster(x, kmeans_apca.predict(x_apca), cluster_id=0, period='day', time_basis = 48)
    plot_cluster(x, kmeans_apca.predict(x_apca), cluster_id=0, period='week', time_basis = 48)    
