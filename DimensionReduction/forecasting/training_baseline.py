import sys
sys.path.append('..')
from utils import *

#Read input args
parser = argparse.ArgumentParser()
parser.add_argument("--ID", type=int, default="1002", help="ID of the resident for which the consumption will be forecast")
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
        monitor="val_MeanSquaredError",  # "val_loss",
        patience= 5,
        min_delta=0.0025,
        mode='min',)
kwargs = {"accelerator": "gpu", "gpus": [0], "auto_select_gpus": True, "callbacks": [my_stopper]}
directory  = './results'
model_name = f'baseline_{args.ID}'
os.makedirs(os.path.join(directory, model_name), exist_ok = True)
model = RNNModel(model = "LSTM" , input_chunk_length=7*24, training_length=7*24, random_state = 42, n_epochs = 20, 
                save_checkpoints = True, work_dir = directory, n_rnn_layers = 3, hidden_dim = 25, model_name = model_name, 
                log_tensorboard = True, force_reset = True, torch_metrics = MeanSquaredError(), pl_trainer_kwargs = kwargs)
model.fit(series=train_serie, val_series=val_serie)
pred_serie = model.predict(n=len(val_serie))
pred, true = compute_intersect(pred_serie, val_serie)
metrics = [rmse(true, pred), coefficient_of_variation(true, pred), biased_error(true.values(), pred.values()), model.epochs_trained]
np.save(os.path.join(directory, model_name, 'metrics'), metrics)