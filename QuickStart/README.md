# About

* This repository includes code for processing energy consumption data accessed via 
the Irish Social Science Data Archive - www.ucd.ie/issda: Electricity Customer Behaviour Trial - Study Number (SN): 0012-00. The data are not accessible here, but may be requested directly from the ISSDA 
via a request form (https://www.ucd.ie/issda/data/commissionforenergyregulationcer/).

* This repository also provides code for time series forecasting on the processed data based on the DARTS framework. 

# Instructions

### Dependencies

See ```requirements.txt```

### Running the scripts

The following files from the ISSDA must be added before running the script
```Prepare_datasets_from_raw.py```:

* Data_raw/Electricity: add the files "File1.txt", ..., "File6.txt" and "SME and Residential allocations.xlsx"

Once you got the files from ```Prepare_datasets_from_raw.py```, you can launch ```Time_series_forecasting.py```:

```
python Time_series_forecasting.py --ID 1239 --data_folder "/home/nvlaminc/Documents/Projects/Ariac/building/TRAIL_workshop_2022/Energy_consumption_forecasting/Data/Electricity" --results_folder "/home/nvlaminc/Documents/Projects/Ariac/building/results/CER_Smart_Metering_Project"
```

# Author(s)

* All the data processing code was written by Rebecca Marion (https://github.com/rebeccamarion/Energy_consumption_forecasting).

* All the time series forecasting code was written by Noémie Vlaminck (Cenaero).

# References

* Commission for Energy Regulation (CER). (2012). CER Smart Metering Project - Electricity Customer Behaviour Trial, 2009-2010 [dataset]. 1st Edition. Irish Social Science Data Archive. SN: 0012-00. https://www.ucd.ie/issda/data/commissionforenergyregulationcer/

* DARTS - Time Series Made Easy in Python. https://github.com/unit8co/darts
