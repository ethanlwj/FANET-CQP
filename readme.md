# Univariate Throughput Forecasting model for Flying Ad Hoc Networks (FANETs)

## Description

This repo contains the source code, data and example inferencing figures for a univariate throughput forecasting model designed for FANETs. The datasets present in the "Data" folder are proprietary data that are collected in an outdoor open field environment, with one UAV communicating with a Ground Control Station (GCS). In the experiments, the UAV move according to the Gauss Markov (GM) or Random Waypoint Mobility Model (RWPM). The UAV communicates with the GCS at a fixed rate of 5Mbps for 10 minutes. Throughput data are collected at 1s intervals at the GCS for a total of 1200 data points per experiment. 

The collected data are then used to train, test and evaluate a univariate throughput forecasting model for FANETs based on the seq2seq model. 

## Installation
Run the following to install the necessary packages for running the codes in this repo:
```pip3 install -r requirements.txt```

## Usage
1. cd into the src directory
```cd src```

2. Run train.py to start training the model. The train.py script will also evaluate the model using a separate dataset. Model pth file will be saved in `../Model` while figures will be saved in `../Figures`.
```python3 train.py --args1 <value1> --args2 <value2> ...```

The complete argument list is:
- `target_len`: Length of the prediction window. Default=3
- `seq_len`: Length of the history window. Default=10
- `batch_size`: Batch size used to train the model. Default=32
- `epochs`: Number of epochs to train the model. Default=200
- `hidden_size`: Sizes of all hidden layers. Default=1024
- `enc_layer`: Number of enc_layers. Default=6
- `dec_layer`: Number of decoder layers. Default=6
- `lr`: Learning rate used to train the model. Default=0.00001
- `training_method`: Training method for the Seq2seq model. Options="Recursive", "Teacher forcing", "Mixed teacher forcing". Default=Recursive
- `teacher_forcing_rate`: Teacher forcing rate when "Recursive" is not used as the training method. Default=0.5
- `loss_function`: Loss function is used to train the model. Options="MSE", "RMSE". Default=MSE
- `cell_type`: Cell type for the seq2seq model. Options="GRU", "LSTM". Default=GRU
- `multivariate`: Enable or disable multivariate mode for seq2seq model. Default=False
- `dropout`: Enable or disable dropout layers in the seq2seq model. Default=False
- `bidirectional`: Enable or disable bidrectional mode for the cells in the seq2seq model. Default=False

3. The script will automatically save the trained model into the `Model` directory once training is completed. 


