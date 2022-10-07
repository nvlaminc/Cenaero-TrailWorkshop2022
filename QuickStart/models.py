import time
import numpy as np
from darts.models import *
from darts.metrics import rmse

class TimeForecastingModel(object):
    def __init__(self, model_cls, **kwargs):
        self._model_cls = model_cls
        self._model = model_cls(**kwargs)

    def fit(self, train_data, **kwargs):
        start_time = time.time()
        self._model.fit(train_data, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"Training time for {self._model_cls.__name__} model is: {elapsed_time:.4f}s")

    def predict(self, val_data, **kwargs):
        start_time = time.time()
        pred = self._model.predict( n = len(val_data), **kwargs)
        elapsed_time = time.time() - start_time
        print(f"Test time for {self._model_cls.__name__} model is: {elapsed_time:.4f}s")
        pred_rmse = rmse(val_data, pred)
        val_data.plot(new_plot=True,label="Actual data")
        pred.plot(label=f"{self._model_cls.__name__} forecast")
        print(f"RMSE error for the {self._model_cls.__name__} model (): {pred_rmse:.4f} kWh")
        return pred, pred_rmse

    def historical_forecast(self, series, start, horizon, stride, **kwargs):
        rmse_list = list()
        start_time = time.time()
        pred = self._model.historical_forecasts(series, start = start, forecast_horizon = horizon, 
                                                verbose = True, stride = stride, last_points_only = False, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"Test time for {self._model_cls.__name__} model is: {elapsed_time}s")
        series.plot(new_plot=True, label="Data")
        for fcast_serie in pred:
            fcast_serie.plot(label = f"Backtest {self._model_cls.__name__}")
            pred_rmse = rmse(series, fcast_serie)
            rmse_list.append(rmse(series, fcast_serie))
            print(f"RMSE error for the {self._model_cls.__name__} model (): {pred_rmse:.4f} kWh")
        print(f"Mean RMSE error for the {self._model_cls.__name__} model (): {np.mean(rmse_list):.4f} kWh")
        return pred, rmse_list
