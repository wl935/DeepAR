import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    Decoder of the Encoder-Decoder structure
    """
    def __init__(self,seq_len:int,covariate_size:int,hidden_size:int):
        super(Decoder,self).__init__()
        self.seq_len = seq_len
        self.covariate_size = covariate_size
        self.hidden_size = hidden_size
        self.LSTM = nn.LSTM(input_size=covariate_size+1,hidden_size=hidden_size)
        self.linear_mu = nn.Linear(in_features=hidden_size, out_features=1)
        self.linear_sigma = nn.Linear(in_features=hidden_size,out_features=1)
        self.soft_plus = nn.Softplus()
        self.dropout = nn.Dropout(0.3)

    def forward(self,input, h_0, c_0):
        # decoder's input hidden input at time t_0 is the last hidden states of the encoder
        # later, the hidden and state should be the returned value from previous step
        outputs, (h_n,c_n) = self.LSTM(input, (h_0, c_0)) # h_n : [1, batch_size,hidden_size]
                                                # c_n : [1, batch_size, hidden_size]
                                                # outputs: [seq_len, batch_size, hidden_size]
        outputs = self.dropout(outputs)
        mus = self.linear_mu(outputs) #(seq_len, batch_size, 1)
        sigmas = self.soft_plus(self.linear_sigma(outputs)) # (seq_len,batch,1)
        return h_n, c_n, mus, sigmas