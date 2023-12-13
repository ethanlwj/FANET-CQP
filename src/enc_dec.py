import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

class LSTMEncoder(nn.Module):
    ''' Encoder model with LSTM as basic units '''

    def __init__(self, input_size, hidden_size, num_layers):
        '''
        Initializes the encoder model

        : param input_size      : Number of features in the input X
        : param hidden_size     : Number of features in the hidden state h
        : param num_layers      : Number of recurrent layers. Value > 1 will stack the LSTMs together 
        '''
        super(LSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first=True)

    def forward(self, X, encoder_hidden_state):
        '''
        forward pass of the LSTM encoder model

        : param X                       : input to the encoder with shape (batch_size, seq_len, input_size)

        : return lstm_out, hidden_state : lstm_out is of shape (batch_size, seq_len, hidden_size) and contains the 
                                          the output features from the last layer of the LSTM for each t.
                                          hidden_state contains the final hidden state and cell state for each sequence 
                                          in the batch. 
        '''

        if encoder_hidden_state == None:
            encoder_hidden_state = self.init_hidden(32)
        
        # View avoids explicit data copy, allowing fast and memory efficient operations. Changes to data in the view
        # will be reflected in the base tensor as well
        lstm_out, self.hidden_state = self.lstm(X.view(X.shape[0], X.shape[1], X.shape[2]))

        return lstm_out, self.hidden_state

    def init_hidden(self, batch_size):
        
        '''
        Initialize hidden state

        : param batch_size:    x_input.shape[1]

        : return:              zeroed hidden state and cell state 
        '''
        
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))

class LSTMDecoder(nn.Module):
    ''' Decoder model with LSTM as the basic units '''

    def __init__(self, input_size, hidden_size, num_layers):
        '''
        Initializes the decoder model

        : param input_size      : number of features in the input X
        : param hidden_size     : number of features in the hidden state 
        : param num_layers      : number of stacked LSTM layers

        '''
        super(LSTMDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)
        self.relu = nn.ReLU()

    def forward(self, X, encoder_hidden_state):
        '''
        Forward pass for the decoder model. First, gets the last hidden state in the final layer for all t.
        Then, pass the final hidden state to a fully connected layer to generate final output of decoder.

        : param X                       : input to the decoder with size (batch_size, input_size) 2D tensor
        : param encoder_ hidden_state   : hidden state obtained from the encoder 

        : return out, hidden_state      : out is the final output of the decoder of size. 
                                          hidden_state contains the final hidden state and cell state for each sequence 
                                          in the batch. 
        '''

        if len(X.shape) == 2:
            X = X.unsqueeze(1)

        lstm_out, self.hidden_state = self.lstm(X, encoder_hidden_state)

        out = lstm_out[:,-1,:]
        out = self.linear(out)
        out = self.relu(out)

        return out, self.hidden_state

class GRUEncoder(nn.Module):
    ''' Encoder model with LSTM as basic units '''

    def __init__(self, input_size, hidden_size, num_layers, bidirectional):
        '''
        Initializes the encoder model

        : param input_size      : Number of features in the input X
        : param hidden_size     : Number of features in the hidden state h
        : param num_layers      : Number of recurrent layers. Value > 1 will stack the LSTMs together 
        '''
        super(GRUEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.gru = nn.GRU(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first=True, bidirectional=bidirectional)

    def forward(self, X, encoder_hidden_state):
        '''
        forward pass of the LSTM encoder model

        : param X                       : input to the encoder with shape (batch_size, seq_len, input_size)

        : return lstm_out, hidden_state : lstm_out is of shape (batch_size, seq_len, hidden_size) and contains the 
                                          the output features from the last layer of the LSTM for each t.
                                          hidden_state contains the final hidden state and cell state for each sequence 
                                          in the batch. 
        '''

        # View avoids explicit data copy, allowing fast and memory efficient operations. Changes to data in the view
        # will be reflected in the base tensor as well
        if encoder_hidden_state == None:
            encoder_hidden_state = self.init_hidden(32)


        gru_out, self.hidden_state = self.gru(X.view(X.shape[0], X.shape[1], X.shape[2]), encoder_hidden_state)

        return gru_out, self.hidden_state

    def init_hidden(self, batch_size):
        
        '''
        Initialize hidden state

        : param batch_size:    x_input.shape[1]

        : return:              zeroed hidden state and cell state 
        '''

        if self.bidirectional:
            hidden_state = (torch.zeros(2*self.num_layers, batch_size, self.hidden_size),
                torch.zeros(2*self.num_layers, batch_size, self.hidden_size))

        else:
            hidden_state = (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))
        
        return hidden_state

class GRUDecoder(nn.Module):
    ''' Decoder model with LSTM as the basic units '''

    def __init__(self, input_size, hidden_size, num_layers):
        '''
        Initializes the decoder model

        : param input_size      : number of features in the input X
        : param hidden_size     : number of features in the hidden state 
        : param num_layers      : number of stacked LSTM layers

        '''
        super(GRUDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)
        self.relu = nn.ReLU()

    def forward(self, X, encoder_hidden_state):
        '''
        Forward pass for the decoder model. First, gets the last hidden state in the final layer for all t.
        Then, pass the final hidden state to a fully connected layer to generate final output of decoder.

        : param X                       : input to the decoder with size (batch_size, input_size) 2D tensor
        : param encoder_ hidden_state   : hidden state obtained from the encoder 

        : return out, hidden_state      : out is the final output of the decoder of size. 
                                          hidden_state contains the final hidden state and cell state for each sequence 
                                          in the batch. 
        '''

        # 
        if len(X.shape) == 2:
            X = X.unsqueeze(1)

        # print(X.shape)
        gru_out, self.hidden_state = self.gru(X, encoder_hidden_state)

        out = gru_out[:,-1,:]
        out = self.linear(out)
        out = self.relu(out)

        return out, self.hidden_state

class AttnDecoderGRU(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers,history_length, multivariate):
        super(AttnDecoderGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.history_length = history_length
        self.multivariate = multivariate
        
        self.attention = nn.Linear(self.hidden_size+self.input_size, self.history_length)
        self.attn_combine = nn.Linear(self.hidden_size+self.input_size, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, 1)
        self.relu = nn.ReLU()

        self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=num_layers, batch_first=True)
        
    def forward(self, X, hidden_state, encoder_outputs):
        '''
        X (batch_size,1)
        prev_hidden_state (num_layers, batch_size, hidden_size)
        encoder_output (num_layers, batch_size, hidden_size)
        '''
        batch_size = X.shape[0]

        prev_hidden_state = hidden_state[0, :, :]
        attn_input = torch.cat((X, prev_hidden_state), dim=1)

        attn_weights = F.softmax(self.attention(attn_input), dim=-1)
        # (batch_size, history_length)

        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs) # (batch_size, 1, hidden_size)
        # attn_applied = attn_applied.reshape(batch_size, batch_size, attn_applied.shape[2])

        if len(X.shape) == 2:
            X = X.unsqueeze(1)

        attn_combine = torch.cat((X, attn_applied), 2)

        attn_combine = self.relu(self.attn_combine(attn_combine))
        
        gru_out, decoder_hidden = self.gru(attn_combine, hidden_state)

        out = gru_out[:,-1,:]
        out = self.relu(self.fc(out))

        return out, decoder_hidden, attn_weights

class AutoCorrelationDecoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, history_length, multivariate, dropout, repeat=40):
        super(BahdanauDecoderGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.history_length = history_length
        self.multivariate = multivariate
        self.dropout_enabled = dropout
        self.repeat = repeat

        self.fc_hidden = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc_encoder = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.weight = nn.Parameter(torch.FloatTensor(1, hidden_size), requires_grad=True)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(input_size=self.hidden_size+self.input_size*self.repeat, hidden_size=self.hidden_size, num_layers=num_layers, batch_first=True)

        if multivariate:
            self.fc1 = nn.Linear(self.hidden_size, 128)
            self.fc2 = nn.Linear(128, input_size)        
        else:
            self.fc_final = nn.Linear(self.hidden_size, input_size)

        if dropout:
            self.dropout = nn.Dropout(p=0.2)

    def forward(self, X, hidden_state, encoder_outputs):
        pass 

class BahdanauDecoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, history_length, multivariate, dropout, repeat=40):
        super(BahdanauDecoderGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.history_length = history_length
        self.multivariate = multivariate
        self.dropout_enabled = dropout
        self.repeat = repeat

        self.fc_hidden = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc_encoder = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.weight = nn.Parameter(torch.FloatTensor(1, hidden_size), requires_grad=True)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(input_size=self.hidden_size+self.input_size*self.repeat, hidden_size=self.hidden_size, num_layers=num_layers, batch_first=True)

        if multivariate:
            self.fc1 = nn.Linear(self.hidden_size, 128)
            self.fc2 = nn.Linear(128, input_size)        
        else:
            self.fc_final = nn.Linear(self.hidden_size, input_size)

        if dropout:
            self.dropout = nn.Dropout(p=0.2)

        # if transpose_conv:
        #     self.conv1 = nn.Conv1d()

    def forward(self, X, hidden_state, encoder_outputs):
        '''
        Input sizes
        X - (batch_size, 1)
        hidden_state - (1, batch_size, hidden_size)
        encoder_outputs - (batch_size, seq_len, hidden_size)
        '''
        hidden_state_reshape = torch.transpose(hidden_state, 0, 1)

        # Calculate alignment score
        tmp = torch.tanh(self.fc_hidden(hidden_state_reshape) + self.fc_encoder(encoder_outputs)) # (N, seq_len, H_enc)
        tmp_reshape = torch.transpose(tmp, 1, 2)
        alignment_scores = torch.matmul(self.weight, tmp_reshape) # (32,1,20)

        # Softmaxing alignment scores to get attention weights
        attn_weights = F.softmax(alignment_scores, dim=2)

        # Multiplying attention weights with encoder outputs to get context vector
        context_vec = torch.bmm(attn_weights, encoder_outputs) # (32, 1, 256)

        # Concatenating the context vector with the input 
        X_repeat = X.unsqueeze(1).repeat(1,1,self.repeat)
        attn_vec = torch.cat((context_vec, X_repeat), dim=2)

        # Passing the concatenated vector to the GRU layer
        gru_out, h_dec = self.gru(attn_vec, hidden_state)

        # Passing the output of the GRU layer to a Linear layer for forecasting
        output = gru_out[:,-1,:]

        if self.multivariate:
            if self.dropout_enabled:
                output = self.relu(self.dropout(self.fc1(output)))
                output = self.relu(self.dropout(self.fc2(output)))
            else:
                output = self.relu(self.fc1(output))
                output = self.relu(self.fc2(output))
        else:
            if self.dropout_enabled:
                output = self.relu(self.dropout(self.fc_final(output)))
            else:
                output = self.relu(self.fc_final(output))

        return output, h_dec, attn_weights

class MultiFlowDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, history_length, multivariate, dropout, repeat=40):
        super(MultiFlowDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.history_length = history_length
        self.multivariate = multivariate
        self.dropout_enabled = dropout
        self.repeat = repeat

        self.fc_hidden = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc_encoder = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.weight = nn.Parameter(torch.FloatTensor(1, hidden_size), requires_grad=True)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(input_size=self.hidden_size+self.input_size*self.repeat, hidden_size=self.hidden_size, num_layers=num_layers, batch_first=True)

        self.fc1 = nn.Linear(self.hidden_size, 1)
        self.fc2 = nn.Linear(self.hidden_size, 1)
        self.fc3 = nn.Linear(self.hidden_size, 1)
        self.fc4 = nn.Linear(self.hidden_size, 1)
        self.fc5 = nn.Linear(self.hidden_size, 1)
        self.fc6 = nn.Linear(self.hidden_size, 1)

        if dropout:
            self.dropout = nn.Dropout(p=0.2)

        # if transpose_conv:
        #     self.conv1 = nn.Conv1d()

    def forward(self, X, hidden_state, encoder_outputs):
        '''
        Input sizes
        X - (batch_size, 1)
        hidden_state - (1, batch_size, hidden_size)
        encoder_outputs - (batch_size, seq_len, hidden_size)
        '''
        hidden_state_reshape = torch.transpose(hidden_state, 0, 1)

        # Calculate alignment score
        tmp = torch.tanh(self.fc_hidden(hidden_state_reshape) + self.fc_encoder(encoder_outputs)) # (N, seq_len, H_enc)
        tmp_reshape = torch.transpose(tmp, 1, 2)
        alignment_scores = torch.matmul(self.weight, tmp_reshape) # (32,1,20)

        # Softmaxing alignment scores to get attention weights
        attn_weights = F.softmax(alignment_scores, dim=2)

        # Multiplying attention weights with encoder outputs to get context vector
        context_vec = torch.bmm(attn_weights, encoder_outputs) # (32, 1, 256)

        # Concatenating the context vector with the input 
        X_repeat = X.unsqueeze(1).repeat(1,1,self.repeat)
        attn_vec = torch.cat((context_vec, X_repeat), dim=2)

        # Passing the concatenated vector to the GRU layer
        gru_out, h_dec = self.gru(attn_vec, hidden_state)

        # Passing the output of the GRU layer to a Linear layer for forecasting
        output = gru_out[:,-1,:]

        flow1_out = self.relu(self.fc1(output))
        flow2_out = self.relu(self.fc2(output))
        flow3_out = self.relu(self.fc3(output))
        flow4_out = self.relu(self.fc4(output))
        flow5_out = self.relu(self.fc5(output))
        flow6_out = self.relu(self.fc6(output))

        output = (flow1_out, flow2_out, flow3_out, flow4_out, flow5_out, flow6_out)

        return output, h_dec, attn_weights

class LuongDecoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, history_length, multivariate):
        super(LuongDecoderGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.history_length = history_length
        self.multivariate = multivariate

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc_attention = nn.Linear(hidden_size, hidden_size, bias=False)
        self.weight = nn.Parameter(torch.FloatTensor(1, hidden_size))
        self.fc_final = nn.Linear(hidden_size*2, input_size)
        self.relu = nn.ReLU()

    def forward(self, X, hidden_state, encoder_outputs):
        if len(X.shape) == 2:
            X = X.unsqueeze(1)

        # Passing the input and previous hidden state through the GRU layer to generate current hidden state
        gru_out, h_dec = self.gru(X, hidden_state)

        gru_out_reshape = torch.transpose(gru_out, 0, 1) # gru_out -> (32, 1, 256)

        # Calculate alignment scores (concat method)
        # tmp = torch.tanh(self.fc_attention(gru_out_reshape + encoder_outputs)) # (32, 20, 256)
        # tmp_reshape = torch.transpose(tmp, 1, 2) # (32, 256, 20)
        # alignment_scores = torch.matmul(self.weight, tmp_reshape) # (1,256) * (32, 256, 20) = (32,1,20)

        # General method
        tmp = self.fc_attention(gru_out) # (32, 1, 256)
        tmp_reshape = torch.transpose(tmp, 1, 2) # (32, 256, 1)
        alignment_scores = torch.matmul(encoder_outputs, tmp_reshape) # (32, 20, 256) * (32, 256, 1) = (32, 20, 1)
        alignment_scores = torch.transpose(alignment_scores, 1, 2) # (32, 1, 20)

        # Softmaxing the alignment scores to get attention weights
        attn_weights = F.softmax(alignment_scores, dim=-1)

        # Multiplying attention weights with encoder outputs to get context vector
        context_vec = torch.bmm(attn_weights, encoder_outputs)

        # Concatenating the context vector with the gru ouput 
        attn_vec = torch.cat((context_vec, gru_out), dim=2)

        # Passing the concatenated vector to linear layer to generate prediction
        output = self.relu(self.fc_final(attn_vec[:,0,:]))

        return output, h_dec, attn_weights

