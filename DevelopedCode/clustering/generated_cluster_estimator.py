import sys
sys.path.append('..')
from utils import *

###################
#### Load Data ####
###################

file_path = '../../QuickStart/Data/Electricity/residential_all_hour_with_date_time.pkl'
x_date_time = pd.read_pickle(file_path)
ids = x_date_time.columns.values

## Smoothing ##
x_date_time = gaussian_smoothing(x_date_time)

## Keep One Year ##
df = x_date_time.sort_values(by='date_time')["2009-07-15" : "2010-07-14"]
x = np.array(df).T

## Outliers Removal ##
id_remove = []
# Maximum- and variance-based outlier removal
for i in np.argsort(-x.max(1))[:10]:
    id_remove.append(i)
for i in np.argsort(-x.std(1))[:10]:
    id_remove.append(i)
id_remove = np.asarray(id_remove)
# Isolation Forest outlier removal
clf = IsolationForest(random_state=0, contamination = 0.01, n_jobs=-1).fit(x)
pred = clf.predict(x)
#Concatenate and delete the outliers
id_remove = np.concatenate((id_remove, np.squeeze(np.argwhere((pred==-1)))))
id_remove = np.unique(id_remove)
print(f'Shape before removing outlier {x.shape}.')
x = np.delete(x, id_remove, 0)
ids = np.delete(ids, id_remove, 0)
print(f'Shape after removing outlier {x.shape}.')

########################
###### Clustering ######
########################

n_components = 10
n_clusters = 4
do_plot = False

x_pca = PCA(n_components=n_components).fit_transform(x)

##############################
#### 1. KMeans Clustering ####
##############################

print(f'KMeans clustering with {n_clusters} clusters.')
t = time.time()
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(x_pca)
print(f'Time to cluster the dataset {time.time()-t:.2f} s.')
if do_plot:
    plot_clusters(x, kmeans.predict(x_pca))
kmeans_clusters = gen_cluster(x=x, ids=ids, labels=kmeans.labels_, method='kmeans')

################################
#### 2. Spectral Clustering ####
################################

print(f'Spectral clustering with {n_clusters} clusters.')
t = time.time()
spectral = SpectralClustering(n_clusters=n_clusters, random_state=0, affinity = "nearest_neighbors").fit(x_pca)
print(f'Time to cluster the dataset {time.time()-t:.2f} s.')
if do_plot:
    plot_clusters(x, spectral.labels_)
spectral_clusters = gen_cluster(x=x, ids=ids, labels=spectral.labels_, method='spectral clustering')


#################################
#### 3. Self-Organizing Maps ####
#################################

print(f'SOM clustering with {n_clusters} clusters.')
t = time.time()
som = susi.SOMClustering(n_rows=1,n_columns=n_clusters)
som.fit(x_pca)
cluster_coord = pd.DataFrame(np.array(som.get_clusters(x_pca)), columns = ["dim1", "dim2"])
labels_ = cluster_coord.groupby(['dim1', 'dim2']).grouper.group_info[0]
print(f'Time to cluster the dataset {time.time()-t:.2f} s.')
if do_plot:
    plot_clusters(x, labels_)
som_clusters = gen_cluster(x=x, ids=ids, labels=labels_, method='som clustering')

##############################################
#### Generate Models Trained on Centroids ####
##############################################

#1. KMeans

print('Generating Models for KMeans computed centroids...')

for c in range(len(kmeans_clusters)):
    train, val = gen_train_val_series(c, kmeans_clusters, x_date_time)
    directory = 'cluster/kmeans/'
    os.makedirs(directory, exist_ok = True)
    model_name = f"RNN_cluster_{c}"
    my_stopper = EarlyStopping(
        monitor="val_MeanSquaredError",  # "val_loss",
        patience= 7,
        min_delta=0.0025,
        mode='min',)
    kwargs = {"accelerator": "gpu", "gpus": [0], "auto_select_gpus": True, "callbacks": [my_stopper]}
    model = RNNModel(model = "LSTM" , input_chunk_length=7*24, training_length=7*24, random_state = 42, n_epochs = 20, 
                    save_checkpoints = True, work_dir = directory, n_rnn_layers = 3, hidden_dim = 25, model_name = model_name, 
                    log_tensorboard = True, force_reset = True, torch_metrics = MeanSquaredError(), pl_trainer_kwargs = kwargs)
    model.fit(series=train, val_series=val)
    pred_serie = model.predict(n=len(val))
    pred, true = compute_intersect(pred_serie, val)
    metrics = [rmse(true, pred), coefficient_of_variation(true, pred), biased_error(true.values(), pred.values())]
    np.save(os.path.join(directory, model_name, 'logs', 'metrics'), metrics)
np.save(os.path.join(directory, 'cluster'), kmeans_clusters)

#2. Spectral 

print('Generating Models for Spectral computed centroids...')

for c in range(len(spectral_clusters)):
    train, val = gen_train_val_series(c, spectral_clusters, x_date_time)
    directory = 'cluster/spectral/'
    os.makedirs(directory, exist_ok = True)
    model_name = f"RNN_cluster_{c}"
    my_stopper = EarlyStopping(
        monitor="val_MeanSquaredError",  # "val_loss",
        patience= 7,
        min_delta=0.0025,
        mode='min',)
    kwargs = {"accelerator": "gpu", "gpus": [0], "auto_select_gpus": True, "callbacks": [my_stopper]}
    model = RNNModel(model = "LSTM" , input_chunk_length=7*24, training_length=7*24, random_state = 42, n_epochs = 20, 
                    save_checkpoints = True, work_dir = directory, n_rnn_layers = 3, hidden_dim = 25, model_name = model_name, 
                    log_tensorboard = True, force_reset = True, torch_metrics = MeanSquaredError(), pl_trainer_kwargs = kwargs)
    model.fit(series=train, val_series=val)
    pred_serie = model.predict(n=len(val))
    pred, true = compute_intersect(pred_serie, val)
    metrics = [rmse(true, pred), coefficient_of_variation(true, pred), biased_error(true.values(), pred.values())]
    np.save(os.path.join(directory, model_name, 'logs', 'metrics'), metrics)
np.save(os.path.join(directory, 'cluster'), spectral_clusters)

#3. SOM

print('Generating Models for SOM computed centroids...')

for c in range(len(som_clusters)):
    train, val = gen_train_val_series(c, som_clusters, x_date_time)
    directory = 'cluster/som/'
    os.makedirs(directory, exist_ok = True)
    model_name = f"RNN_cluster_{c}"
    my_stopper = EarlyStopping(
        monitor="val_MeanSquaredError",  # "val_loss",
        patience= 7,
        min_delta=0.0025,
        mode='min',)
    kwargs = {"accelerator": "gpu", "gpus": [0], "auto_select_gpus": True, "callbacks": [my_stopper]}
    model = RNNModel(model = "LSTM" , input_chunk_length=7*24, training_length=7*24, random_state = 42, n_epochs = 20, 
                    save_checkpoints = True, work_dir = directory, n_rnn_layers = 3, hidden_dim = 25, model_name = model_name, 
                    log_tensorboard = True, force_reset = True, torch_metrics = MeanSquaredError(), pl_trainer_kwargs = kwargs)
    model.fit(series=train, val_series=val)
    pred_serie = model.predict(n=len(val))
    pred, true = compute_intersect(pred_serie, val)
    metrics = [rmse(true, pred), coefficient_of_variation(true, pred), biased_error(true.values(), pred.values())]
    np.save(os.path.join(directory, model_name, 'logs', 'metrics'), metrics)
np.save(os.path.join(directory, 'cluster'), som_clusters)
