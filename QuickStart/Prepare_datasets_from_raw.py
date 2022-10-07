# -*- coding: utf-8 -*-
"""
Script for transforming data from Data_raw folder into the data in Data folder
"""

import glob 
import os
import pandas as pd

# custom functions
import utils

# Directories for raw data and processed data
prefix_raw = "Data_raw/"
prefix_data = "Data/"

##########################
#### ELECTRICITY DATA ####
##########################

data_type = "Electricity"
file_path_raw_data = prefix_raw + data_type + "/"
file_path_data = prefix_data + data_type + "/"


## Meta data (meter ids, building type, etc) ##


file_name_meta_data = file_path_raw_data + 'SME and Residential allocations.xlsx'
df = pd.read_excel(file_name_meta_data)
meta_data = df.iloc[:, 0:4]
# Residential: Code = 1
res_IDs = list(meta_data.loc[meta_data.Code == 1, "ID"]) # IDs for residential 
# Daylight savings time (DST) day codes
dst_days = {"beg": [452],
            "end": [298, 669]}
# Daylight savings time (DST) time codes before and after end of DST
dst_times = {"before": [3, 4],
            "after": [5, 6]}

## Consumption data ##

file_names_consum_data = sorted(glob.glob(os.path.join(file_path_raw_data, "*.txt")))
nb_files = len(file_names_consum_data)

df_list = list()
for file_index in range(nb_files):
    
    print('\n processing file %d of %d' % (file_index + 1, nb_files))
    
    file_name_consum_data = file_names_consum_data[file_index]
    
    df = pd.read_csv(file_name_consum_data, sep = " ", header = None)
    df.columns = ["ID", "day_time_code", "consumption"]

    df_sub = utils.format_raw_data(df_old = df, res_IDs = res_IDs, dst_days = dst_days, dst_times = dst_times)
    df_list.append(df_sub)

df_all = pd.concat(df_list).reset_index(drop = True) 

del df_list

df_all = df_all.sort_values(['ID', 'date_time'], ascending=[True, True]).reset_index(drop = True)
file_name_out = file_path_data + "residential_all.pkl"
utils.save_data(file_name_out, df_all)

df_ID = df_all.groupby(df_all.ID).get_group(1003)
# Consumption data by consumer
list_ID_hour_all = []
#IDs_list = df_all.ID.to_list()
for ID, df_ID in df_all.groupby(df_all.ID):
    file_name_out = file_path_data + f"residential_{ID}.pkl"
    utils.save_data(file_name_out, df_ID)
    df_ID= df_ID.set_index("date_time")
    df_ID_hour = df_ID["consumption"].resample("60min", label='right', closed='right').sum()
    df_ID_hour.rename(ID, inplace = True)
    #df_ID_hour = pd.DataFrame(df_ID["consumption"].resample("60min", label='right', closed='right').sum(), columns = ["consumption"])
    file_name_out = file_path_data + f"residential_{ID}_hour.pkl"
    utils.save_data(file_name_out, df_ID_hour)
    list_ID_hour_all.append(df_ID_hour)
    
df_ID_hour_all = pd.concat(list_ID_hour_all, axis = 1, keys = [s.name for s in list_ID_hour_all])
file_name_out = file_path_data + "residential_all_hour_with_date_time.pkl"
utils.save_data(file_name_out, df_ID_hour_all)

# =============================================================================
# df = df_ID_hour_all.sort_values(by='date_time')
# gp = df.groupby(df.index.to_period('D'))
# ID_gp = list(gp.groups.keys())[2]
# gp.get_group(ID_gp)
# =============================================================================
