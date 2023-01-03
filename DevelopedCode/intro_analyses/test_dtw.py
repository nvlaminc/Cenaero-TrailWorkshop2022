import time
import os, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans

if __name__ == "__main__":
    #Read input args
    parser = argparse.ArgumentParser() 
    parser.add_argument("--data_file", type=str, default="../../QuickStart/Data/Electricity/residential_all_hour_with_date_time.pkl", help="the path to the data") 
    parser.add_argument("--results_folder", type=str, default="./results", help="the path to the output folder") 
    args = parser.parse_args()
    os.makedirs(args.results_folder, exist_ok=True)
    #Load data
    file_path= args.data_file
    x_date_time = pd.read_pickle(file_path)
    ids = x_date_time.columns.values
    df = x_date_time.sort_values(by='date_time')["2009-07-15" : "2010-12-31"]
    val_reshape = np.array(df).T
    #Apply Kmeans with DTW for a mean week
    val_reshape = np.reshape(val_reshape[:, :7*24*76], (3639, -1, 7, 24))
    val_reshape = val_reshape.mean(1)
    val_reshape = np.reshape(val_reshape, (3639, 7*24))
    x = val_reshape
    start_time = time.time()
    #kmeans = TimeSeriesKMeans(n_clusters = 14, verbose=True, n_jobs = -1, metric="dtw", metric_params={"global_constraint":"itakura", "itakura_max_slope":2.}, max_iter=300)
    #kmeans = TimeSeriesKMeans(n_clusters = 14, verbose=True, n_jobs = -1, metric="dtw", metric_params={"global_constraint" : "sakoe_chiba", "sakoe_chiba_radius" : 3}, max_iter=300)
    kmeans = TimeSeriesKMeans(n_clusters = 14, verbose=True, n_jobs = -1, metric="dtw", max_iter=300, random_state=0)
    kmeans.fit(x)
    elapsed_time = time.time() - start_time
    fig = plt.figure()
    j = 0
    n_belongs = (kmeans.predict(x)==j).sum()
    tmp = val_reshape[kmeans.predict(x)==j].reshape(n_belongs, -1, 24)
    for i in range(n_belongs):
        plt.plot(tmp[i].mean(0), c='black', linewidth=0.5, alpha=0.5)
    plt.plot(val_reshape[kmeans.predict(x)==j].reshape(n_belongs, -1, 24).mean(1).mean(0))
    fig = plt.figure()
    j = 0
    n_belongs = (kmeans.predict(x)==j).sum()
    tmp = val_reshape[kmeans.predict(x)==j][:, :7*24*76].reshape(n_belongs, -1, 7, 24)
    for i in range(n_belongs):
        plt.plot(tmp[i].mean(0).flatten(), c='black', linewidth=0.5, alpha=0.5)
    plt.plot(tmp.mean(1).mean(0).flatten())
    plt.show()
    print(elapsed_time)
    print("end of the script")