import numpy as np
import pandas as pd
from Encoder import Encoder
from Decoder import Decoder
from utils import train_fn,deepar_dataset,read_df,inference
import torch
import matplotlib.pyplot as plt

config_dict = {
  'context_len': 100,
  'prediction_len': 20,
  'covariate_size': 9,
  'hidden_size': 20,
  'sep': ';',
  'index_col': 0,
  'parse_dates': True,
  'decimal': ',',
  'file_name': 'LD2011_2014.txt',
  'target_col': 'MT_005',
  'covariate_cols': ['MT_001', 'MT_002', 'MT_003',
                     'MT_004', 'MT_006', 'MT_007', 'MT_008'],
}


if __name__=='__main__':
    context_len = config_dict['context_len']
    prediction_len = config_dict['prediction_len']
    file_name = config_dict['file_name']
    sep = config_dict['sep']
    index_col = config_dict['index_col']
    parse_dates = config_dict['parse_dates']
    decimal = config_dict['decimal']

    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    encoder = Encoder(seq_len=context_len,
                      covariate_size=9,
                      hidden_size=20,
                      device=device)
    
    decoder = Decoder(seq_len=prediction_len,
                      covariate_size=9,
                      hidden_size=20)
    encoder = encoder.double()
    decoder = decoder.double()
    train_df, test_df, scale = read_df(file_name=file_name, sep= sep,
                                       index_col=index_col, parse_dates=parse_dates,
                                       decimal=decimal,config=config_dict)
    #context_df = train_df.iloc[-5*context_len:].copy()
    #context_df.iloc[:, 0] = train_df.iloc[-5*context_len-1:-1, 0].values
    context_df = train_df.iloc[-context_len:].copy()
    context_df.iloc[:,0] = train_df.iloc[-context_len-1:-1,0].values

    our_dataset = deepar_dataset(train_df, context_len=context_len, prediction_len=prediction_len)
    train_fn(encoder, decoder, our_dataset, lr=1e-3, batch_size=128, num_epochs=200, device=device)
    mus_list, sigmas_list = inference(context_df, device, encoder, decoder, train_df, test_df, prediction_len)
    mu_df = pd.DataFrame(index=test_df.index, data= {'mus': mus_list})
    mu_df = mu_df * scale
    sigma_df = pd.DataFrame(index=test_df.index, data= {'sigmas':sigmas_list})
    #sigma_df = sigma_df * np.sqrt(scale)
    print(f"scale is: {scale}")
    sigma_df = sigma_df * scale
    mu_df['upper_bound'] = mu_df['mus'] + 3 * sigma_df['sigmas']
    mu_df['lower_bound'] = mu_df['mus'] - 3 * sigma_df['sigmas']
    test_df = test_df * scale
    fig = plt.figure()
    plt.plot(test_df.iloc[:, 0])
    plt.plot(mu_df)
    plt.savefig('result2.png')