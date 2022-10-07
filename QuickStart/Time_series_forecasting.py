import os
import darts
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from darts.models import *
from darts.utils.statistics import plot_acf, check_seasonality
from sklearn.linear_model import LinearRegression
from models import TimeForecastingModel
from pytorch_lightning.callbacks import EarlyStopping
from torchmetrics import MeanSquaredError

if __name__ == "__main__":
    #Read input args
    parser = argparse.ArgumentParser()
    parser.add_argument("--ID", type=int, default="1002", help="ID of the resident for which the consumption will be forecast") 
    parser.add_argument("--data_folder", type=str, default="./results", help="the path to the dataset folder") 
    parser.add_argument("--results_folder", type=str, default="./results", help="the path to the output folder") 
    args = parser.parse_args()
    os.makedirs(args.data_folder, exist_ok=True)
    os.makedirs(args.results_folder, exist_ok=True)
    #Load data
    file_path= os.path.join(args.data_folder, f"residential_{args.ID}.pkl")
    df = pd.read_pickle(file_path)
    df["ID"] = df["ID"].astype("category")
    df["time_code"] = df["time_code"].astype("uint16")
    df = df.set_index("date_time")
    print(df.info())
    df_elec = df["consumption"].resample("60min", label='right', closed='right').sum().to_frame()
    print(df_elec.info())
    #Import time series into darts
    series = darts.TimeSeries.from_dataframe(df_elec, value_cols="consumption")
    series.plot(new_plot = True)
    #Split into train, val datasets
    train, val = series.split_before(pd.Timestamp("20100714"))
    train.plot(new_plot=True, label="training")
    val.plot(label="validation")
    #Inspect Seasonality
    plot_acf(train, m = 24, max_lag = 24, alpha=0.05)
    for m in range(2, 25):
        is_seasonal, period = check_seasonality(train,  m = m, max_lag = 25, alpha = 0.05)
        if is_seasonal:
            print("There is seasonality of order {}.".format(period))

    #Naive Seasonal Model
    naive_model = TimeForecastingModel(NaiveSeasonal, K = 7*24) 
    pred, pred_rmse = naive_model.historical_forecast(series = series, start = pd.Timestamp("20100715"), horizon = 7*24, stride = 30*24,
                                retrain = True, train_length = 7*24*365)
    #Linear Regression model
    linear_model = TimeForecastingModel(RegressionModel, lags = 7*24, output_chunk_length = 7*24, model = LinearRegression())
    pred, pred_rmse = linear_model.historical_forecast(series = series, start = pd.Timestamp("20100715"), horizon = 7*24, stride = 30*24,
                            retrain = True, train_length = 7*24*365)
    #Prophet
    prophet_model = TimeForecastingModel(Prophet)
    pred, pred_rmse = prophet_model.historical_forecast(series = series, start = pd.Timestamp("20100715"), horizon = 7*24, stride = 30*24,
                        retrain = True, train_length = 7*24*365)
    #LSTM
    torch_metrics = MeanSquaredError()
    
    my_stopper = EarlyStopping(
    monitor="train_MeanSquaredError",  # "val_loss",
    patience= 3,
    min_delta=0.005,
    mode='min',
    )
    
    kwargs = {"accelerator": "gpu", "gpus": -1, "auto_select_gpus": True, "callbacks": [my_stopper]}
    rnn_model = TimeForecastingModel(RNNModel, model = "LSTM" , input_chunk_length=7*24, training_length=7*24, random_state = 42, 
                                n_epochs = 20, save_checkpoints = True, work_dir = args.results_folder, n_rnn_layers = 3,
                                hidden_dim = 25, model_name = "RNN_test_29082022", log_tensorboard = True, force_reset = True, 
                                torch_metrics = torch_metrics, pl_trainer_kwargs = kwargs)

    pred, pred_rmse = rnn_model.historical_forecast(series = series, start = pd.Timestamp("20100715"), horizon = 7*24, stride = 30*24,
                        retrain = True, train_length = 7*24*365)
    plt.show()