# PyTorch implementation of LSTM Model for Multi-time-horizon Solar Forecasting

## How to Run
### Conda environment for running the code 
  A conda environment file is provided for convenience. Assuming you have Anaconda python distribution available on your computer, you can create a new conda environment with the necessary packages using the following command:

`conda env create -f conda_env.yaml -n "multi_time_horizon"`
  
### Running the code
  1. Clone (or download) the repository: 
  
  `git clone https://github.com/sakshi-mishra/LSTM_Solar_Forecasting.git`
  

### Training/Testing Data

The training and testing data needs to be downloaded from the [NOAA FTP server](ftp://aftp.cmdl.noaa.gov/data/radiation/surfrad/) for the locations/sites. You can use GNU wget to automate the download process. The scripts assume that the data is in the *data* folder as per the structure outlined in the [data_dir_struct.txt](data_dir_struct.txt) file.

