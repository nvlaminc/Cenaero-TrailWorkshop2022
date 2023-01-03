import sys
sys.path.append('..')
from utils import *

#Read input args
parser = argparse.ArgumentParser()
parser.add_argument("--ID", type=int, default="1002", help="ID of the resident for which the consumption will be forecast")
parser.add_argument("--CLUSTERING", type=str, default="som")
parser.add_argument("--GPU", type=int, default="0")
args = parser.parse_args()
#Init variables
file_path= f"../../QuickStart/Data/Electricity/residential_{args.ID}.pkl"
df = pd.read_pickle(file_path)
df["ID"] = df["ID"].astype("category")
df["time_code"] = df["time_code"].astype("uint16")
df = df.set_index("date_time")
df_elec = df["consumption"].resample("60min", label='right', closed='right').sum().to_frame()
#Create train/val series
df_train = df_elec["2009-07-15" : "2010-07-14"]
df_val   = df_elec["2010-07-14" : "2010-07-21"]
train_serie = darts.TimeSeries.from_dataframe(df_train, value_cols="consumption")
val_serie = darts.TimeSeries.from_dataframe(df_val, value_cols="consumption")
#Init the forecasting model
my_stopper = EarlyStopping(
        monitor="val_loss",  # "val_MeanSquaredError",
        patience= 5,
        min_delta=0.005,
        mode='min',)
kwargs = {"accelerator": "gpu", "gpus": [args.GPU], "auto_select_gpus": True, "callbacks": [my_stopper]}
directory  = './results'
model_name = f'{args.CLUSTERING}_{args.ID}'
os.makedirs(os.path.join(directory, model_name), exist_ok = True)
cluster_list = np.load(os.path.join('../clustering/cluster', args.CLUSTERING, 'cluster.npy'), allow_pickle=True)
for i in range(len(cluster_list)):
    if args.ID in cluster_list[i]['ids']:
        label = cluster_list[i]['labels']    
model = RNNModel.load_from_checkpoint(model_name = f'RNN_pretrained_{label}/',  work_dir = f'../clustering/cluster/{args.CLUSTERING}', best=True)
del model.trainer_params['callbacks'] 
model.trainer_params['callbacks'] = [my_stopper]
model.trainer_params['gpus'] = [args.GPU]
model.fit(series=train_serie, val_series=val_serie, epochs=50) 
pred_serie = model.predict(n=len(val_serie))
pred, true = compute_intersect(pred_serie, val_serie)
#pay attention that model.epochs_trained gives the number of epochs since the original checkpoint
metrics = [rmse(true, pred), coefficient_of_variation(true, pred), biased_error(true.values(), pred.values()), model.epochs_trained]
np.save(os.path.join(directory, model_name, 'metrics'), metrics)