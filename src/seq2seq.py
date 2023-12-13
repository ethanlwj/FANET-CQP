import torch 
import torch.nn as nn
import torch.optim
from tqdm import trange
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import math

from enc_dec import LSTMDecoder, LSTMEncoder, GRUDecoder, GRUEncoder, BahdanauDecoderGRU, LuongDecoderGRU, AttnDecoderGRU

def rmse_loss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

class Seq2Seq(nn.Module):
    ''' 
    > Sequence-to-sequence model with the encoder and decoder models\n
    > Also contains the functions to train, test and predict using the model    
    '''

    def __init__(self, input_size, n_prediction, n_history, hidden_size, num_encoder_layers, num_decoder_layers, 
                 device, loss_function, cell_type, multivariate, dropout, bidirectional):
        super(Seq2Seq, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.device = device
        self.cell_type = cell_type
        self.history_length = n_history
        self.multivariate = multivariate
        self.dropout_enabled = dropout
        self.prediction_length = n_prediction
        self.bidirectional = bidirectional
        

        if cell_type == 'LSTM':
            self.encoder = LSTMEncoder(input_size, hidden_size, num_encoder_layers)
            self.decoder = LSTMDecoder(input_size, hidden_size, num_decoder_layers)
        elif cell_type == 'GRU':
            self.encoder = GRUEncoder(input_size, hidden_size, num_encoder_layers, bidirectional)
            self.decoder = GRUDecoder(input_size, hidden_size, num_decoder_layers)
        elif cell_type == 'attention_GRU':
            self.encoder = GRUEncoder(input_size, hidden_size, num_encoder_layers, bidirectional)
            self.decoder = BahdanauDecoderGRU(input_size, hidden_size, num_decoder_layers, n_history, self.multivariate, dropout)

        if multivariate:
            self.fc = nn.Linear(input_size, 1)
            self.relu = nn.ReLU()
            self.weights = nn.Parameter(torch.FloatTensor(1, 1), requires_grad=True)

        if dropout:
            self.dropout = nn.Dropout(p=0.2)

        if loss_function == 'MSE':
            self.loss_fn = nn.MSELoss()
            self.loss_fn_name = 'MSE'
        elif loss_function == 'RMSE':
            self.loss_fn = rmse_loss
            self.loss_fn_name = 'RMSE'

    def train_model(self, train_loader, val_loader, n_epochs, batch_size, learning_rate, training_method, teacher_forcing_rate, plot, fig_path=None, k=10):
        '''
        > param train_loader:           Dataloader containing the training data (input and target)
        > param n_epochs:               Number of epochs
        > param batch_size:             Number of samples in each batch for training
        > parma learning_rate:          Optimizer lr
        > param training_method:        Specifies the training method used to train the seq2seq model can be any of
                                        Recursive, Teacher forcing, Mixed teacher forcing
        > param teacher_forcing_rate:   The probability that teacher forcing is used to predict the next data in the sequence
        > param plot:                   Boolean value to determine if the training loss should be plotted or not
        
        > return losses:                Training loss of each epoch
        
        '''
        # Array to store the loss value in each epoch 
        train_losses = []
        val_losses = []

        # Initialize optimizer and loss function
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, mode='min', patience=20) # adaptive LR

        n_batches = len(train_loader)

        self.training_method = training_method
        self.teacher_forcing_ratio = teacher_forcing_rate
        
        with trange(n_epochs) as tr:
            for it in tr:
                self.train()
                batch_loss = 0
                for batch, (X,Y) in enumerate(train_loader):
                    X, Y = X.to(self.device), Y.to(self.device)
                    '''
                    X and Y has the shape (batch size, seq_len, num_features)
                    '''
                    v = k/(k+math.exp(it/k))
                    forecast, _ = self._forward_pass(X, Y, batch_size, 'Train', v)

                    # Calculate batch loss
                    loss = self.loss_fn(forecast,Y)
                    batch_loss += loss.item()

                    # Backpropagation
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                batch_loss /= n_batches
                train_losses.append(batch_loss)
                val_loss, _, _, _, _ = self.test(val_loader)
                val_losses.append(val_loss)

                self.scheduler.step(val_loss)

                # progress bar 
                tr.set_postfix(loss="{0:.3f}, val_loss = {1:.3f}".format(batch_loss,val_loss))

        if plot:
            plt.figure(figsize=[8,6])
            plt.plot(range(n_epochs), train_losses, c='red')
            plt.plot(range(n_epochs), val_losses, c='blue')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('train_loss={}, val_loss={}'.format(batch_loss, val_loss))
            plt.legend(['Train loss', 'Validation loss'])

            if fig_path != None:
                plt.savefig(fig_path)
            else:
                plt.show()
                    
        return train_losses, val_losses

    def predict(self, X, Y):
        
        '''
        : param input_tensor:      input data (seq_len, input_size); PyTorch tensor 
        : param target_len:        number of target values to predict 
        : return np_outputs:       np.array containing predicted values; prediction done recursively 
        '''
        
        start = time.time()
        # encode input_tensor
        input_shape = X.shape
        if len(input_shape) == 2:
            X = X.unsqueeze(0) # Add in batch size of 1 if prediction is done on single sample

        forecast, attn_weights = self._forward_pass(X, Y, 1, 'Inference')

        end = time.time()
        time_elapsed = end - start
            
        np_forecast = forecast.cpu().detach().numpy()
        
        return np_forecast, time_elapsed, attn_weights

    def test(self, test_loader):
        
        '''
        test the lstm encoder-decoder model
        : param test_loader:        testing data in Dataloader format 
        : param n_prediction:       number of prediction horizon
        :
        : return test_loss:         average loss from the testing
        '''
        
        test_loss = 0
        n_batches = len(test_loader)
        time_list = np.zeros((n_batches,1))
        Y_pred = np.zeros((n_batches, self.prediction_length, self.input_size))
        Y_actual = np.zeros((n_batches, self.prediction_length, self.input_size))
        attn_weights = np.zeros((n_batches, self.prediction_length, self.history_length))
        self.eval()
        with torch.no_grad():
            for batch, (X,Y) in enumerate(test_loader):
                X, Y = X.to(self.device), Y.to(self.device)
                y_pred, time_elapsed, attn_weight_curr = self.predict(X, Y)
                Y_pred[batch, :, :] = y_pred
                Y_actual[batch, :, :] = Y.cpu().detach().numpy()
                attn_weights[batch, :, :] = attn_weight_curr
                time_list[batch,0] = time_elapsed
                y_pred = torch.from_numpy(y_pred).type(torch.Tensor).to(self.device)
                test_loss += self.loss_fn(y_pred,Y).item()


        test_loss /= n_batches
        average_time = np.mean(time_list)

        return test_loss, average_time, Y_pred, Y_actual, attn_weights

    def save_model(self, save_path):
        torch.save(self.state_dict(), save_path)
        print("Saved PyTorch Model State to " + save_path)

    def _get_corrected_output(self, x):
        input_shape = x.shape
        if len(input_shape) == 3:
            batch_size = input_shape[0]
        else:
            batch_size = 1

        if batch_size > 1:
            correction_factor = self.relu(self.fc(x[:,:,1:]))
            forecast = self.relu(x[:,:,0].unsqueeze(-1) + correction_factor)
        else:
            correction_factor = self.relu(self.fc(x[:,1:]))
            forecast = self.relu(x[:,0].unsqueeze(-1) + correction_factor)

        return forecast

    def _forward_pass(self, X, Y, batch_size, type, v=None):

        assert type in ['Train', 'Inference'], "Type can only be \'Train\' or \'Inference\'"

        # Initialize hidden state
        if self.cell_type in ['GRU', 'attention_GRU']:
            encoder_hidden = self.encoder.init_hidden(batch_size)[0].to(self.device)
        else:
            encoder_hidden = None

        # Pass input through encoder 
        encoder_out, encoder_hidden = self.encoder(X, encoder_hidden)

        # If Encoder is bidirectional, add the forward and backward features before passing it 
        # to the decoder
        if self.bidirectional:
            encoder_out = encoder_out[:,:,:self.hidden_size] + encoder_out[:,:,self.hidden_size:]

        # Initialize numpy array to store attention weights used for prediction 
        attn_weights = np.zeros((self.prediction_length, self.history_length))

        # Pass encoder hidden state to decoder
        decoder_input = X[:,-1,:]
        if self.cell_type in ['GRU', 'attention_GRU']:
            decoder_hidden = encoder_hidden[-self.num_decoder_layers:, :, :]
        else:
            decoder_hidden = (encoder_hidden[0][-self.num_decoder_layers:, :, :], encoder_hidden[1][-self.num_decoder_layers:, :, :])

        if type == 'Train':
            # Initialize the output tensor
            output = torch.zeros(batch_size, self.prediction_length, X.shape[2]).to(self.device)

            if self.cell_type == "attention_GRU":
                if self.training_method == "Recursive":
                    for t in range(self.prediction_length):
                        decoder_out, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_out)
                        output[:,t,:] = decoder_out
                        decoder_input = decoder_out

                elif self.training_method == "Teacher Forcing":
                    for t in range(self.prediction_length):
                        decoder_out, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_out)
                        output[:,t,:] = decoder_out
                        decoder_input = Y[:,t,:]

                elif self.training_method == "Mixed Teacher Forcing":
                    for t in range(self.prediction_length):
                        decoder_out, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_out)
                        output[:,t,:] = decoder_out
                        temp = random.random() 
                        if (temp<v): # Teacher Forcing
                            decoder_input = Y[:,t,:] 
                        else: # Recursive
                            decoder_input = decoder_out
            else:
                if self.training_method == "Recursive":
                    for t in range(self.prediction_length):
                        decoder_out, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                        output[:,t,:] = decoder_out
                        decoder_input = decoder_out

                elif self.training_method == "Teacher Forcing":
                    for t in range(self.prediction_length):
                        decoder_out, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                        output[:,t,:] = decoder_out
                        decoder_input = Y[:,t,:]

                elif self.training_method == "Mixed Teacher Forcing":
                    for t in range(self.prediction_length):
                        decoder_out, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                        output[:,t,:] = decoder_out
                        temp = random.random() 
                        if (temp<v): # Teacher Forcing
                            decoder_input = Y[:,t,:] 
                        else: # Recursive
                            decoder_input = decoder_out

        elif type == 'Inference':
            # initialize tensor for predictions
            output = torch.zeros(self.prediction_length, X.shape[2]).to(self.device)

            if self.cell_type == 'attention_GRU':
                for t in range(self.prediction_length):
                    decoder_output, decoder_hidden, attn_weight_t = self.decoder(decoder_input, decoder_hidden, encoder_out)
                    attn_weights[t,:] = attn_weight_t.cpu().detach().numpy()
                    output[t] = decoder_output.squeeze(0)
                    decoder_input = decoder_output
            else:
                for t in range(self.prediction_length):
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    output[t] = decoder_output.squeeze(0)
                    decoder_input = decoder_output

        forecast = output 
        
        return forecast, attn_weights

