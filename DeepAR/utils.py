import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from Encoder import Encoder
from Decoder import Decoder

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def calc_loss(X, Y_ahead, Y, encoder, decoder,device):
  loss = torch.tensor([0.0],device=device)
  X = X.double()
  Y = Y.double()
  X = X.permute(1, 0, 2) # shape change to (seq_len, batch_size, input_size)

  X = X.to(device)
  Y = Y.to(device)
  Y_ahead = Y_ahead.to(device)
  encoder = encoder.to(device)
  decoder = decoder.to(device)

  enc_h_n,enc_c_n, enc_mus, enc_sigmas = encoder(X)
  Y_ahead = Y_ahead.permute(1,0,2)
  Y = Y.permute(1,0)
  h,c,mus,sigmas = decoder(Y_ahead,enc_h_n,enc_c_n) # Y :(seq_len,batch_size, input_size) where input size = covariate_size+1, \
                                              # enc_h_n : (1,batch_size, hidden_size)
                                              # enc_c_n : (1,batch_size, hodden_size)
                                              # mus : (seq_len, batch_size,1)
                                              # sigmas : (seq_len,batch_size,1)
  real_vals = Y # the first element of input_size is the real value at that point
  real_vals = real_vals.unsqueeze(2) # real_vals of size [seq_len, batch_size,1]
  real_vals = real_vals.to(device)

  neg_square_dist = -1 * torch.square(real_vals - mus)
  two_sigma_square = 2 * torch.square(sigmas)
  exp_part = torch.exp(neg_square_dist/two_sigma_square)
  two_pi_sigma_square = 2 * 3.1415926 * torch.square(sigmas)

  likelihood = torch.div(exp_part,torch.sqrt(two_pi_sigma_square))
  log_likelihood = torch.log(likelihood)
  neg_log_likelihood = -1 * log_likelihood

  cur_total_loss = torch.sum(neg_log_likelihood)

  loss += cur_total_loss
  return loss


def train_fn(encoder, decoder, dataset, lr, batch_size, num_epochs, device):

  encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
  decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)

  data_iter = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=4)
  l_sum = 0.0
  for i in range(num_epochs):
    epoch_loss_sum = 0.0
    total_sample = 0
    for (X,Y_ahead,Y) in data_iter:
      encoder_optimizer.zero_grad()
      decoder_optimizer.zero_grad()
      l = calc_loss(X,Y_ahead,Y,encoder,decoder,device)
      l.backward()
      encoder_optimizer.step()
      decoder_optimizer.step()
      epoch_loss_sum += l.item()
      total_sample += X.shape[0]
      epoch_mean_loss = epoch_loss_sum/total_sample
    print(f"epoch{i} loss:{epoch_mean_loss}")
    l_sum += epoch_mean_loss
    if (i+1)% 5 == 0:
      print(f"epoch_num{i+1}, current loss: {l_sum/5}")
      l_sum = 0.0


def read_df(file_name: str, sep: str, index_col: int, parse_dates: bool,
            decimal: str, config: dict):
    """
    
    :param file_name: file name of the input data frame
    :param sep: seperation char
    :param index_col: index of the index column of data frame
    :param parse_dates: whether to parse dates or not
    :param decimal: decimal char
    :param config: config the
    :return: data frame after processing, scale
    """
    total_df = pd.read_csv(file_name,sep=sep, index_col=index_col, parse_dates=parse_dates, decimal=decimal)
    full_cols = [config['target_col']] + config['covariate_cols']
    target_df = total_df[full_cols].copy()
    target_df.loc[:, 'hour'] = target_df.index.hour
    target_df.loc[:, 'dayofweek'] = target_df.index.dayofweek
    for i in range(1, target_df.shape[1]):
      target_df.iloc[:,i] = (target_df.iloc[:,i] - np.mean(target_df.iloc[:,i]))/ np.std(target_df.iloc[:,i])
    scale = target_df[config['target_col']].sum() / target_df.shape[0]
    target_df[config['target_col']] = target_df[config['target_col']]/scale
    prediction_len = config['prediction_len']
    train_df = target_df.iloc[-5000:-prediction_len, :]
    test_df = target_df.iloc[-prediction_len:, :]
    return train_df, test_df, scale


def inference(context_df: pd.DataFrame,
              device: torch.device,
              encoder: Encoder,
              decoder: Decoder,
              train_df: pd.DataFrame,
              test_df: pd.DataFrame,
              prediction_len: int ):

    context_numpy = np.array(context_df)
    context_tensor = torch.tensor(context_numpy) #[seq_len, input_size]
    context_tensor = context_tensor.unsqueeze(0) #[1, seq_len, input_size]
    context_tensor = context_tensor.permute(1, 0, 2)
    context_tensor = context_tensor.to(device)
    with torch.no_grad():
      enc_h_n, enc_c_n, _, _ = encoder(context_tensor)
      previous_target = train_df.iloc[-1, 0]
      h_n, c_n = enc_h_n, enc_c_n
      mus_list = []
      sigmas_list = []
      for i in range(prediction_len):
        cur_covariate = test_df.iloc[i, 1:].values.tolist()
        cur_input = [previous_target] + cur_covariate
        print(cur_input)
        cur_array = np.array(cur_input)
        print(cur_array)
        cur_tensor = torch.tensor(cur_array)  # [1, input_size]
        cur_tensor = cur_tensor.unsqueeze(0)  # [1,1,input_size]
        cur_tensor = cur_tensor.unsqueeze(1)
        cur_tensor = cur_tensor.to(device)
        print(cur_tensor.shape)
        h_n, c_n, mus, sigmas = decoder(cur_tensor, h_n, c_n)  # mus: [1,1,1]
        cur_sigma = sigmas.squeeze().cpu().detach().item()
        cur_mu = mus.squeeze().cpu().detach().item()
        previous_target = cur_mu
        print(f'cur_mu: {cur_mu}')
        print(f"cur_sigma: {cur_sigma}")
        mus_list.append(cur_mu)
        sigmas_list.append(cur_sigma)
    return mus_list, sigmas_list


class deepar_dataset(Dataset):
  def __init__(self, input_df: pd.DataFrame, context_len, prediction_len):
    self.input_df = input_df
    self.context_len = context_len
    self.prediction_len = prediction_len
    self.total_len = context_len + prediction_len

  def __len__(self):
    return self.input_df.shape[0] - self.total_len - 1

  def __getitem__(self, idx):
    context = self.input_df.iloc[idx:idx + self.context_len, :]
    prediction = self.input_df.iloc[idx + self.context_len: idx + self.total_len, 0]  # only contains target itself
    prediction_step_ahead = self.input_df.iloc[idx + self.context_len - 1: idx + self.total_len - 1, :]
    x = torch.tensor(np.array(context))
    x = x.double()
    y = torch.tensor(np.array(prediction))
    y = y.double()
    y_ahead = torch.tensor(np.array(prediction_step_ahead))
    y_ahead = y_ahead.double()
    return x, y_ahead, y
