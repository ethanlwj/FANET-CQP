import pandas as pd 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.optim
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from math import floor, ceil


#===============================================================================================#
# Dataset formatting
#===============================================================================================#

def split_dataset(y, n_history, n_prediction, stride):
    '''
    Create a windowed dataset from the input time-series

    : param y               : input series to be windowed
    : param n_history       : number of past observations as input to the model
    : param n_prediction    : prediction horizon of the model 
    : param stride          : spacing between the windows
    
    : return X, Y           : arrays of windowed input and output series for the model
                              X has shape (n_history, n_samples, n_features)
                              Y has shape (n_prediction, n_samples, n_features)
    '''

    L = y.shape[0]
    n_samples = int((L - n_history - n_prediction) / stride + 1)
    X = np.zeros([n_history, n_samples, 1])
    Y = np.zeros([n_prediction, n_samples, 1])


    for i in range(n_samples):
        start_x = stride*i
        end_x = start_x + n_history
        start_y = end_x
        end_y = end_x + n_prediction 

        X[:, i, 0] = y[start_x:end_x]
        Y[:, i, 0] = y[start_y:end_y]
        
    return X, Y

def split_dataset_df(df, n_history, n_prediction, stride, throughput_only=False):

    n_features = len(df.columns) - 1
    L = len(df)
    n_samples = int((L - n_history - n_prediction) / stride + 1)

    X = np.zeros([n_samples, n_history, n_features])
    if not throughput_only:
        Y = np.zeros([n_samples, n_prediction, n_features])
    else:
        Y = np.zeros([n_samples, n_prediction, 1])

    try:
        df = df.drop('Time', axis=1)
    except:
        pass

    for i in range(n_samples):
        start_x = stride*i
        end_x = start_x + n_history
        start_y = end_x
        end_y = end_x + n_prediction 

        for n, feature in enumerate(df.columns):
            '''
            Throughput_t
            Distance
            Delay
            SINR_bin
            RSSI_bin
            '''

            X[i,:,n] = df[feature][start_x:end_x]
            if not throughput_only:
                Y[i,:,n] = df[feature][start_y:end_y]
            else:
                if feature == "Throughput_t":
                    Y[i,:,0] = df[feature][start_y:end_y]

        
    return X, Y

def train_test_split_custom(t, y, train_ratio):
    '''
    Splits the time and feature array to train and test datasets according to the given ratio

    : param 
    '''

    # Remove the first 2 data because the network has not started yet
    t = t[2:]
    y = y[2:]

    len_t = t.shape[0]
    len_y = y.shape[0]

    if len_t != len_y:
        raise Exception("Number of time samples are not the same as number of feature samples!")

    split_idx = int(train_ratio*len_t)
    
    t_train = t[:split_idx]
    t_test = t[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]

    return t_train, t_test, y_train, y_test

def numpy_to_torch(X_train, Y_train, X_test, Y_test):
    '''
    convert numpy array to PyTorch tensor
    
    : param X_train:                           windowed training input data (input window size, # examples, # features); np.array
    : param Ytrain:                           windowed training target data (output window size, # examples, # features); np.array
    : param Xtest:                            windowed test input data (input window size, # examples, # features); np.array
    : param Ytest:                            windowed test target data (output window size, # examples, # features); np.array
    : return X_train_torch, Y_train_torch,
    :        X_test_torch, Y_test_torch:      all input np.arrays converted to PyTorch tensors 
    '''
    
    X_train_torch = torch.from_numpy(X_train).type(torch.Tensor)
    Y_train_torch = torch.from_numpy(Y_train).type(torch.Tensor)

    X_test_torch = torch.from_numpy(X_test).type(torch.Tensor)
    Y_test_torch = torch.from_numpy(Y_test).type(torch.Tensor)
    
    return X_train_torch, Y_train_torch, X_test_torch, Y_test_torch

def load_data(path, plot=False):
    network_dict = {}
    i=0
    legend_list = [] 

    for file in os.listdir(path):
        csv_path = path + '\\' +  file
        print(csv_path)
        df_network = pd.read_csv(csv_path)
        network_dict[file.split('.csv')[0]] = df_network
        if plot:
            if i<10:
                plt.plot(df_network['Time'],df_network['Throughput_t'])
                legend_list.append('Scenario ' + str(i+1))
                i += 1

    if plot:
        plt.xlabel('Time (s)')
        plt.ylabel('Throughput (bps)')
        plt.legend(legend_list)
        plt.show()

    return network_dict

def bin_data(df, feature_to_bin, max_value=None):
    binned_feature_name = feature_to_bin.split('_')[0] + '_bin'

    min_value = df[feature_to_bin].min()
    if max_value == None:
        max_value = df[feature_to_bin].max()

    if feature_to_bin == "SINR_dBm":
        bins = [-100, -80]
        bins = np.append(bins,list(np.linspace(-20,max_value,10)))
    elif feature_to_bin == "RSSI_dBm":
        bins = [-190, -180]
        bins = np.append(bins, np.linspace(-140, max_value, 10))   

    df[binned_feature_name] = pd.cut(df[feature_to_bin], labels=False, bins=bins, include_lowest=True)

    return df, max_value

def bin_data_new(df, feature_to_bin, max_value=None):
    binned_feature_name = feature_to_bin.split('_')[0] + '_bin'

    min_value = df[feature_to_bin].min()
    if max_value == None:
        max_value = df[feature_to_bin].max()

    if feature_to_bin == "SINR_dBm":
        bins = [-10, 0]
        bins = np.append(bins,list(np.linspace(0,max_value,10)))
    elif feature_to_bin == "RSSI_dBm":
        bins = [-90, -80]
        bins = np.append(bins, np.linspace(-140, max_value, 10))   

    df[binned_feature_name] = pd.cut(df[feature_to_bin], labels=False, bins=bins, include_lowest=True)
    print(df.head())

    return df, max_value

def scale_feature(df, features_to_scale, scaler):
    df_to_scale = df[features_to_scale]
    df = df.drop(features_to_scale, axis=1)
    if scaler == None:
        scaler = MinMaxScaler()
        scaler.fit(df_to_scale)
    
    df_scaled = scaler.transform(df_to_scale)

    for j,feature in enumerate(features_to_scale):
        df[feature] = df_scaled[:,j]

    return df, scaler

def preprocess_multivar(df, features_to_drop, features_to_scale, seq_len, target_len, scaler, max_sinr=None, max_rssi=None, stride=1):
    index_to_drop = [0,1]
    for idx in index_to_drop:
        if idx in df.index.values.tolist():
            df=df.drop(idx, axis=0)
    df_scenario = df.copy()
    df_scenario['Throughput_t'] = df_scenario['Throughput_t']/1e6
    df_scenario_trim = df_scenario.drop(features_to_drop, axis=1)

    # Rearranging the columns so that throughput will be the first feature
    cols = df_scenario_trim.columns.tolist()
    index = cols.index("Throughput_t")
    cols = cols[index:] + cols[0:index]
    df_scenario_trim = df_scenario_trim[cols] 
    # print(df_scenario_trim)

    # Binning the SINR_dBm
    # df_scenario_trim, _ = bin_data_new(df_scenario_trim, "SINR_dBm", max_sinr)
    
    # Binning the RSSI_dBm
    # df_scenario_trim, _ = bin_data_new(df_scenario_trim, "RSSI_dBm", max_rssi)

    df_binned = df_scenario_trim.drop(['SINR_dBm','RSSI_dBm'], axis=1)
    df_binned['SINR_bin'] = df_scenario_trim['SINR_dBm']
    df_binned['RSSI_bin'] = df_scenario_trim['RSSI_dBm']
    
    # Min max scaling 
    df_scaled, scaler = scale_feature(df_binned, features_to_scale, scaler)

    # Splitting dataset into validation and testing sets 
    X, y = split_dataset_df(df_scaled, seq_len, target_len, stride=stride)
        
    X_torch = torch.from_numpy(X).type(torch.Tensor)
    Y_torch = torch.from_numpy(y).type(torch.Tensor)

    return X_torch, Y_torch, scaler


class BandwidthDataset(Dataset):
    def __init__(self, X, Y) -> None:
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[1]

    def __getitem__(self, index):
        return (self.X[:,index,:], self.Y[:,index,:])

class CWTImageDataset(Dataset):
    def __init__(self, X, Y) -> None:
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        image = self.X[index, :, :]
        label = self.Y[index]
        return image, label

class TransformerBandwidthDataset(Dataset):
    '''Bandwidth dataset formatted for transformer'''
    def __init__(self, X, Y) -> None:
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[1]

    def __getitem__(self, index):
        seq_len = self.X.shape[0]
        target_len = self.Y.shape[0]
        time_idx = torch.arange(0, seq_len+target_len).type(torch.float).unsqueeze(0)

        fx = torch.cat([self.X[:,index,:], self.Y[:,index,:]], 0)

        sample = (time_idx, fx)

        return sample

    def _generate_square_subsequent_mask(self,t0,tn):
        mask = torch.zeros(t0+tn,t0+tn)
        for i in range(0,t0):
            mask[i,t0:] = 1 
        for i in range(t0,t0+tn):
            mask[i,i+1:] = 1
        mask = mask.float().masked_fill(mask == 1, float('-inf'))#.masked_fill(mask == 1, float(0.0))
        return mask

class TransformerAdversarialDataset(Dataset):
    def __init__(self, X, Y, Z) -> None:
        self.X = X
        self.Y = Y
        self.Z = Z

    def __len__(self):
        return self.X.shape[1]

    def __getitem__(self, index):
        seq_len = self.X.shape[0]
        target_len = self.Y.shape[0]
        time_idx = torch.arange(0, seq_len+target_len).type(torch.float).unsqueeze(0)

        fx = torch.cat([self.X[:,index,:], self.Y[:,index,:]], 0)
        label = self.Z[index]

        sample = (time_idx, fx, label)

        return sample

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

