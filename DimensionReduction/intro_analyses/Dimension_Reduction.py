import sys
sys.path.append('..')
from utils import *

if __name__ == "__main__":
    pickle_path = '../../QuickStart/Data/Electricity/residential_all.pkl'
    x = gen_array(file_path)
    
    n_cluster = 14

    ### Feature Extraction ###
    
    # 0. No Feature Extraction
    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(x)
    #visualisation in 2D T-SNE manifold fo ease
    tsne = TSNE(n_components=2, n_jobs=-1).fit_transform(x)
    fig = plt.suptitle('Visualisation of KMeans Clustering on Original Signals')
    plt.scatter(tsne[:, 0], tsne[:, 1], c=kmeans.predict(x))
    plt.show()
    
    # Energy Consumption Profile for cluster_id = 0 and a period of 1 day
    plot_cluster(x, kmeans.predict(x), cluster_id=0, period='day')
    
    
    
    
    
    # 1. Principal Component Analysis
    # Investigate the number of component to consider by checking at the exlpained
    #covariance ration
    pca = PCA(n_components=100)
    pca.fit(val_reshape)
    fig = plt.figure('pca')
    plt.plot(1-pca.explained_variance_ratio_[:25])
    plt.grid()
    plt.suptitle('Cumulated Explained Covariance Ratio per components') 
    plt.xlabel('# Components')
    plt.ylabel('1 - Explained Variance Ratio')
    plt.show()
    
    pca = PCA(n_components=10)
    x_pca = pca.fit_transform(x)
    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(x_pca)
    #visualisation of the clustered space in the two first dimension
    fig = plt.figure('Visualisation of KMeans Clustering on PCA Representation')
    plt.scatter(x_pca[:, 0], x_pca[:, 1], c=kmeans.predict(x_pca))
    plt.show()
    
    # Energy Consumption Profile for cluster_id = 0 and a period of 1 day
    plot_cluster(x, kmeans.predict(x), cluster_id=0, period='day')
    
    
    
    
    
    # 2. Multidimensional Scaling (MDS)
    mds = MDS(n_components=10, n_jobs=-1, verbose=2)
    x_mds = mds.fit_transform(x)
    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(x_mds)
    #visualisation of the clustered space in the two first dimension
    #fig = plt.figure('Visualisation of KMeans Clustering on MDS Representation')
    #plt.scatter(x_mds[:, 0], x_mds[:, 1], c=kmeans.predict(x_mds))
    #plt.show()
    
    # Energy Consumption Profile for cluster_id = 0 and a period of 1 day
    #plot_cluster(x, kmeans.predict(x), cluster_id=0, period='day')
    
    
    
    
    
    # 3. Fast Fourier Transform
    x_fft = np.abs(fft(val_reshape, n=100))
    #visulation of the signal in the Frequency domain
    fig = plt.figure('Frequency')
    plt.suptitle('Visualisation of the dataset in the Frequency domain')
    plt.plot(np.transpose(x_fft), c='black', linewidth=0.05, alpha=0.85)
    plt.plot(x_fft.mean(0))
    plt.grid()
    plt.show()
    
    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(x_fft)
    
    
    
    
    
    # 4. Discrete Wavelet Transform 
    from scipy import signal

    dwt = gen_dwt(x)
    #visualation of the mean DWT vs. random DWT
    fig = plt.figure()
    plt.imshow(dwt.mean(0), cmap='binary')
    plt.suptitle('Mean Discrete Wavelet Transform')
    fig = plt.figure()
    rdm_id = np.random.randint(0, len(x))
    plt.imshow(dwt[rdm_id], cmap='binary')
    plt.suptitle(f'Discrete Wavelet Transform of {rdm_id} ID')
    
    x_dwt = dwt.reshape(len(dwt), -1)
    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(x_dwt)
    #plot_cluster(x, kmeans.predict(x), cluster_id=0, period='week')
    
    
    
    
    
    # 5.Piecewise Constant Approximation --- To Complete !! 
    
    window = 20
    x_apca = gen_apca(x, window=window)
    #visualisation of the APCA 
    fig = plt.figure()
    rdm_id = np.random.randint(0, len(x))
    plt.suptitle(f'Piecewise Constant Approximation of Signal {rdm_id}')
    plt.grid()
    plt.plot(x[rdm_id, 100:2500], label='Original Signal')
    plt.plot(x_apca[rdm_id, 100:2500], label='APCA Reduced Signal')
    plt.legend()
    plt.show()
    
    x_apca = x_apca[:, ::window]
    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(x_apca)
    plot_cluster(x, kmeans.predict(x), cluster_id=0, period='week')