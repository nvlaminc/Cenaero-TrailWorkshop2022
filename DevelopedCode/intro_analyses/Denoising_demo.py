#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 11:16:23 2022

@author: rmarion
"""
import sys
sys.path.append('..')
from utils import *
import denoising

file_path = '../../QuickStart/Data/Electricity/residential_all_hour_with_date_time.pkl'
x_date_time = pd.read_pickle(file_path)
ids = x_date_time.columns.values

# Example series
df = x_date_time.sort_values(by='date_time')
df_ex = df.iloc[:, [0]]
df_ex.columns = ["consumption"]
series_darts = darts.TimeSeries.from_series(df_ex)

n_plot = 24*7 # plot one week
time_stamp_end = series_darts.get_timestamp_at_point(n_plot)

# Original series
series_darts.drop_after(time_stamp_end).plot(new_plot = True)
plt.title(r"Original")
plt.show()

## Kalman Filter ##

KF = darts.models.filtering.kalman_filter.KalmanFilter(dim_x=1, kf=None)
KF.fit(series_darts)
series_smoothed = KF.filter(series_darts)

series_darts.drop_after(time_stamp_end).plot(label = "original")
series_smoothed.drop_after(time_stamp_end).plot(new_plot=False,c = "r", label = "smoothed")
plt.title(r"Kalman Filter")
plt.show()


## FFT Filter ##
n_components = 5 # cut off for noise filtering (not sure what good values would be?)
df_smoothed = df_ex.copy()
df_smoothed.loc[:, "consumption"] = denoising.fft_denoiser(df_smoothed.consumption.copy(), n_components = n_components)
series_smoothed =  darts.TimeSeries.from_dataframe(df_smoothed.copy(), value_cols="consumption")

series_darts.drop_after(time_stamp_end).plot(label = "original")
series_smoothed.drop_after(time_stamp_end).plot(new_plot=False,c = "r", label = "smoothed")
plt.title(r"FFT Denoising")
plt.show()

## Rolling Average ##
df_smoothed = df_ex.copy()
df_smoothed = df_smoothed.rolling(25, min_periods = 1, center=True).mean()
series_smoothed =  darts.TimeSeries.from_dataframe(df_smoothed.copy(), value_cols="consumption")

series_darts.drop_after(time_stamp_end).plot(label = "original")
series_smoothed.drop_after(time_stamp_end).plot(new_plot=False,c = "r", label = "smoothed")
plt.title(r"Basic mean")
plt.show()

## Gaussian Filter (std = 3) ##

std = 3

window = signal.windows.gaussian(25, std=std)
plt.figure()
plt.plot(window)
plt.title(r"Gaussian window ($\sigma$=3)")
plt.show()

df_smoothed = df_ex.copy()
df_smoothed = df_smoothed.rolling(25, min_periods = 1, center=True, win_type = "gaussian").mean(std = std)
series_smoothed =  darts.TimeSeries.from_dataframe(df_smoothed.copy(), value_cols="consumption")


series_darts.drop_after(time_stamp_end).plot(label = "original")
series_smoothed.drop_after(time_stamp_end).plot(new_plot=False,c = "r", label = "smoothed")
plt.title(r"Gaussian window ($\sigma$=3)")
plt.show()

## Gaussian Filter (std = 7) ##
std = 7

window = signal.windows.gaussian(25, std=std)
plt.figure()
plt.plot(window)
plt.title(r"Gaussian window ($\sigma$=7)")
plt.show()

df_smoothed = df_ex.copy()
df_smoothed = df_smoothed.rolling(25, min_periods = 1, center=True, win_type = "gaussian").mean(std = std)
series_smoothed =  darts.TimeSeries.from_dataframe(df_smoothed.copy(), value_cols="consumption")

series_darts.drop_after(time_stamp_end).plot(label = "original")
series_smoothed.drop_after(time_stamp_end).plot(new_plot=False,c = "r", label = "smoothed")
plt.title(r"Gaussian window ($\sigma$=7)")
plt.show()