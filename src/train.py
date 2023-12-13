import torch
from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import os
from data_formatting import split_dataset, train_test_split_custom, numpy_to_torch, BandwidthDataset
from seq2seq import Seq2Seq
import argparse

plt.rcParams.update({'font.size': 30})
plt.style.use('ggplot')

def import_data(path):
    network_dict = {}

    for file in os.listdir(path):
        csv_path = path + '/' +  file
        print("Importing data from ... " + csv_path)
        df_network = pd.read_csv(csv_path, names = ['Time', 'Throughput_t'])
        network_dict[file.split('.csv')[0]] = df_network

    return network_dict

def format_data(dict, batch_size, seq_len, target_len):
    test_dl_d = {}

    for i, (key, value) in enumerate(dict.items()):
        t = value['Time']
        y = value['Throughput_t']

        # GM+Waypoint is used to evaluate the model separately
        if key == "GM+waypoint":
            mobilities = ["GM", "RWPM"]
            for i, mob in enumerate(mobilities): # First part is GM, second part is RWPM
                y_sel = y[i*400:i*400+150]
                X_test, Y_test = split_dataset(y_sel, n_history=seq_len, n_prediction=target_len, stride=target_len)
                X_torch = torch.from_numpy(X_test).type(torch.Tensor)
                Y_torch = torch.from_numpy(Y_test).type(torch.Tensor)

                test_dt = BandwidthDataset(X_torch, Y_torch)
                test_dl = DataLoader(test_dt, batch_size=1, drop_last=True)

                test_dl_d[mob] = test_dl

            continue

        # Train:Validation:Test = 0.7:0.15:0.15
        t_train, t_test, y_train, y_test = train_test_split_custom(t, y, 0.7)

        t_val, t_test, y_val, y_test = train_test_split_custom(t_test, y_test, 0.5)

        X_train,Y_train = split_dataset(y_train, n_history=seq_len, n_prediction=target_len, stride=1)
        X_test, Y_test = split_dataset(y_test, n_history=seq_len, n_prediction=target_len, stride=target_len)
        X_val, Y_val = split_dataset(y_val, n_history=seq_len, n_prediction=target_len, stride=1)

        # Gets the tensor type of the train, val and test datasets 
        X_torch_train, Y_torch_train, X_torch_test, Y_torch_test = numpy_to_torch(X_train, Y_train, X_test, Y_test)
        X_torch_val = torch.from_numpy(X_val).type(torch.Tensor)
        Y_torch_val = torch.from_numpy(Y_val).type(torch.Tensor)

        if i == 0:
            X_train_cat = X_torch_train
            X_test_cat = X_torch_test
            X_val_cat = X_torch_val
            Y_train_cat = Y_torch_train
            Y_test_cat = Y_torch_test
            Y_val_cat = Y_torch_val
        else:
            if key != "GM+waypoint":
                X_train_cat = torch.cat((X_train_cat, X_torch_train), dim=1)
                X_test_cat = torch.cat((X_test_cat, X_torch_test), dim=1)
                X_val_cat = torch.cat((X_val_cat, X_torch_val), dim=1)
                Y_train_cat = torch.cat((Y_train_cat, Y_torch_train), dim=1)
                Y_test_cat = torch.cat((Y_test_cat, Y_torch_test), dim=1)
                Y_val_cat = torch.cat((Y_val_cat, Y_torch_val), dim=1)

    
    train_dataset = BandwidthDataset(X_train_cat, Y_train_cat)
    test_dataset = BandwidthDataset(X_test_cat, Y_test_cat)
    val_dataset = BandwidthDataset(X_val_cat, Y_val_cat)

    # Creates data loader for each of the dataset split
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, drop_last=True)

    return train_loader, test_loader, val_loader, test_dl_d

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

def train(target_len, seq_len, batch_size, train_loader, test_loader, val_loader, epochs, hidden_size, enc_layer, dec_layer, lr, training_method, teacher_forcing_rate, loss, cell, multivariate,
          dropout, bidirectional, input_size=1):
    # Ensure reproducibility
    torch.manual_seed(100)

    PLOT = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    print("Training with " + training_method + " method")
    model = Seq2Seq(input_size=input_size, hidden_size=hidden_size, n_prediction=target_len, n_history = seq_len, num_encoder_layers=enc_layer, num_decoder_layers=dec_layer, device=device, loss_function=loss, cell_type=cell, multivariate=multivariate, dropout=dropout, bidirectional=bidirectional).to(device)
    model.apply(reset_weights)
    loss, _ = model.train_model(train_loader=train_loader, val_loader=val_loader, n_epochs=epochs, batch_size=batch_size, learning_rate=lr,training_method=training_method, teacher_forcing_rate=teacher_forcing_rate,plot=PLOT)
    
    _, time_elapsed, Y_pred_seq2seq, Y_actual, _ = model.test(test_loader)
    r2_temp = r2_score(Y_actual.squeeze(),Y_pred_seq2seq.squeeze())
    mse_temp = mean_squared_error(Y_actual.squeeze(),Y_pred_seq2seq.squeeze())
    rmse_temp = np.sqrt(mse_temp)

    print('R2 score : {:.2f}'.format(r2_temp))
    print("Average MSE for " + training_method + ": {:.2f}".format(mse_temp))
    print("Average RMSE : {:.2f}".format(rmse_temp))
    print("Average inferencing time : {}".format(time_elapsed))
    print('\n')

    print("============================= Training completed =============================")

    return model

def evaluate(model, test_dl_d, save_path):
    print("============================= Evaluating... =============================")
    for key,test_dl in test_dl_d.items():
        _, time_elapsed, Y_pred_seq2seq, Y_actual, _ = model.test(test_dl)

        r2_temp = r2_score(Y_actual.squeeze(),Y_pred_seq2seq.squeeze())
        mse_temp = mean_squared_error(Y_actual.squeeze(),Y_pred_seq2seq.squeeze())
        rmse_temp = np.sqrt(mse_temp)

        print("Evaluating model against new data from {}".format(key))
        print('R2 score : {:.2f}'.format(r2_temp))
        print("Average MSE for " + training_method + ": {:.2f}".format(mse_temp))
        print("Average RMSE : {:.2f}".format(rmse_temp))
        print("Average inferencing time : {}".format(time_elapsed))
        print('\n')

        plt.figure(figsize=[10,8])
        plt.plot(range(len(Y_actual.flatten()[:150])),Y_actual.flatten()[:150],c='red',linestyle='--')
        plt.plot(range(len(Y_pred_seq2seq.flatten()[:150])),Y_pred_seq2seq.flatten()[:150],c='orange')

        plt.xlabel('Timesteps')
        plt.ylabel('Throughput (Mbps)')
        plt.legend(['Ground Truth', 'Seq2seq'])

        fig_path = os.path.join(save_path, 'Figures', '{}.png'.format(key))

        plt.savefig(fig_path)

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description='Start simulation script on/off')
        parser.add_argument('--target_len',
                            type=int,
                            default=3,
                            help='Length of the prediction window. Default=3')
        parser.add_argument('--seq_len',
                            type=int,
                            default=10,
                            help='Length of the history window. Default=10')
        parser.add_argument('--batch_size',
                            type=int,
                            default=32,
                            help='Batch size used to train the model. Default=32')
        parser.add_argument('--epochs',
                            type=int,
                            default=200,
                            help='Number of epochs to train the model. Default=200')
        parser.add_argument('--hidden_size',
                            type=int,
                            default=1024,
                            help='Sizes of all hidden layers. Default=1024')
        parser.add_argument('--enc_layer',
                            type=int,
                            default=6,
                            help='Number of enc_layers. Default=6')
        parser.add_argument('--dec_layer',
                            type=int,
                            default=6,
                            help='Number of decoder layers. Default=6')
        parser.add_argument('--lr',
                            type=float,
                            default=0.00001,
                            help='Learning rate used to train the model. Default=0.00001')
        parser.add_argument('--training_method',
                            type=str,
                            default="Recursive",
                            help='Training method for the Seq2seq model. Options="Recursive", "Teacher forcing", "Mixed teacher forcing". Default=Recursive')
        parser.add_argument('--teacher_forcing_rate',
                            type=float,
                            default=0.5,
                            help='Teacher forcing rate when "Recursive" is not used as the training method. Default=0.5')
        parser.add_argument('--loss_function',
                            type=str,
                            default="MSE",
                            help='Loss function is used to train the model. Options="MSE", "RMSE". Default=MSE')
        parser.add_argument('--cell_type',
                            type=str,
                            default="GRU",
                            help='Cell type for the seq2seq model. Options="GRU", "LSTM". Default=GRU')
        parser.add_argument('--multivariate',
                            type=bool,
                            default=False,
                            help='Enable or disable multivariate mode for seq2seq model. Default=False')
        parser.add_argument('--dropout',
                            type=bool,
                            default=False,
                            help='Enable or disable dropout layers in the seq2seq model. Default=False')
        parser.add_argument('--bidirectional',
                            type=bool,
                            default=False,
                            help='Enable or disable bidrectional mode for the cells in the seq2seq model. Default=False')
        args = parser.parse_args()
        target_len = int(args.target_len)
        seq_len = int(args.seq_len)
        batch_size = int(args.batch_size)
        epochs = int(args.epochs)
        hidden_size = int(args.hidden_size)
        enc_layer = int(args.enc_layer)
        dec_layer = int(args.dec_layer)
        lr = float(args.lr)
        training_method = str(args.training_method)
        teacher_forcing_rate = float(args.teacher_forcing_rate)
        loss_function = str(args.loss_function)
        cell_type = str(args.cell_type)
        multivariate = bool(args.multivariate)
        dropout = bool(args.dropout)
        bidirectional = bool(args.bidirectional)

        pwd = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
        data_dir = os.path.join(pwd, "Data")

        network_dict = import_data(data_dir)

        train_loader, test_loader, val_loader, test_dl_d = format_data(network_dict, batch_size, seq_len, target_len)

        oss = "Training the model with the following parameters:\n--batch_size={} \n--target_len={} \n--seq_len={} \n--epochs={} \n--hidden_size={} \n--enc_layer={} \n--dec_layer={} \n--lr={} \n--training_method={} \n--loss_function={} \n--cell_type={} \n--dropout={} \n--bidirectional={}".format(batch_size, target_len, seq_len, epochs, hidden_size, enc_layer, dec_layer, lr, training_method, loss_function, cell_type, dropout, bidirectional)
        print(oss)

        model = train(target_len=target_len, seq_len=seq_len, batch_size=batch_size, train_loader=train_loader, test_loader=test_loader, val_loader=val_loader, epochs=epochs, hidden_size=hidden_size, 
                    enc_layer=enc_layer, dec_layer=dec_layer, lr=lr, training_method=training_method, teacher_forcing_rate=teacher_forcing_rate, loss=loss_function, cell=cell_type, 
                    multivariate=multivariate, dropout=dropout, bidirectional=bidirectional)

        evaluate(model, test_dl_d, pwd)

        print("Saving the model...")
        model_path = os.path.join(pwd, "Model", "seq2seq.pth")
        torch.save(model.state_dict(), model_path)

    except KeyboardInterrupt:
        print("Ctrl-C -> Exit")






