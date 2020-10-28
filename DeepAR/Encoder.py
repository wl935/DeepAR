import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder of the Encoder-Decoder structure.
    """
    def __init__(self, seq_len: int, covariate_size: int, hidden_size: int, device):
        super(Encoder,self).__init__()
        self.device = device
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.LSTM = nn.LSTM(input_size= covariate_size+1,hidden_size=hidden_size)
        self.linear_mu = nn.Linear(in_features=hidden_size,out_features=1 ) # to predict mu
        self.linear_sigma = nn.Linear(in_features=hidden_size, out_features=1) # to predict sigma
        self.soft_plus =nn.Softplus()
        self.dropout = nn.Dropout(0.3)

    def forward(self, input):
        # for the LSTM or recurrent net work, the input is expected to be shaped as
        # [seq_len, batch_size, input_len], where for our case, the input_len is covariate_size +1
        seq_len = input.shape[0]
        batch_size = input.shape[1]
        input_shape = input.shape[2]

        h_0 = torch.zeros((1,batch_size,self.hidden_size),device= self.device)
        c_0 = torch.zeros((1,batch_size,self.hidden_size),device= self.device)
        h_0 = h_0.double()
        c_0 = c_0.double()
        #h_0 = h_0.to(self.device)
        #c_0 = c_0.to(self.device)
        outputs,(h_n,c_n) = self.LSTM(input,(h_0,c_0))  # outputs : [seq_len,batch_size,hidden_size]
                                          # h_n : [1, batch_size, hidden_size]
                                          # c_n : [1, batch_size, hidden_size]
        outputs = self.dropout(outputs)
        if torch.sum(torch.isnan(h_n)) > 1:
            print(f"h_n:{h_n}")
        mus = self.linear_mu(outputs) # shape [seq_len, batch_size,1]
        sigmas = self.soft_plus(self.linear_sigma(outputs)) # shape [seq_len, batch_size, 1]
        return h_n, c_n, mus, sigmas
