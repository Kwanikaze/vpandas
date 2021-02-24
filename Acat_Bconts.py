import utils.process as process
import utils.checks as checks
import models.model as model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
from scipy.stats import multivariate_normal
import utils.params as params

#dict of hyperparameters
args = params.Params('./hyperparameters/binaryAB.json')

df_raw = process.read_csv('https://raw.githubusercontent.com/Kwanikaze/vpandas/master/data/data_A_conts_B_Cat.csv')
input_dims = {'A': 1,'B': 3} 
attributes = list(df_raw.columns) #assumes each attribute has a single column
real_vars = ['A']
cat_vars = [x for x in attributes if x not in real_vars]

df, df_OHE = process.preprocessing(df_raw, attributes,args, real_vars, cat_vars, duplications=100)

#df = process.duplicate_dataframe(df_raw, attributes, duplications=100)
#df = df[attributes].sample(frac=1, random_state=args.random_seed)
#df = process.unif_noise_to_real_columns(df, real_vars)
#df_OHE = process.one_hot_encode_columns(df, cat_vars)

train_df, train_df_OHE, val_df, val_df_OHE, test_df, test_df_OHE = process.split(df,df_OHE,[0.7,0.85])
num_samples = int(train_df.shape[0])

#  use gpu if available
use_gpu = False
device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
VAE_MRF = model.VariationalAutoencoder_MRF(train_df_OHE, val_df_OHE, attributes,input_dims, num_samples,args,cat_vars)
VAE_MRF = VAE_MRF.to(device)
VAE_MRF.train_marginals(args)

model.trainVAE_MRF(VAE_MRF,attributes,train_df_OHE)

checks.graphLatentSpace(VAE_MRF,train_df,train_df_OHE,attributes,num_samples,args)


x_list = np.array([0.01,0.11,0.22,0.33,0.44,0.55,0.66,0.77,0.88,0.99])
x_list  = Variable(torch.from_numpy(x_list))
x_list = x_list.to(device)
print("Print prediction results for A only:")
for x in x_list:
    x = torch.unsqueeze(x,0)
    print("\tInput: {} \t Output: {}".format(x.cpu().detach().numpy(), np.round(VAE_MRF.forward_single_attribute(x=x.float(), attribute='A')[0].cpu().detach().numpy(), decimals=2)))

x_test = np.eye(input_dims["B"])[np.arange(input_dims["B"])]  # Test data (one-hot encoded)
x_test = Variable(torch.from_numpy(x_test))
x_test = x_test.to(device)
print("Print prediction results for B only:")
for x in x_test:
    print(x)
    print("\tInput: {} \t Output: {}".format(x.cpu().detach().numpy(), np.round(VAE_MRF.forward_single_attribute(x=x.float(), attribute='B')[0].cpu().detach().numpy(), decimals=2)))

x_evidence_dict = {'A': x_list[1]} 
x_evidence_dict['A'] = torch.unsqueeze(x_evidence_dict['A'],0)
xB_query = VAE_MRF.query_single_attribute(x_evidence_dict, query_attribute = 'B', evidence_attributes = ['A'], query_repetitions=10000)

x_evidence_dict = {'A': x_list[5]} 
x_evidence_dict['A'] = torch.unsqueeze(x_evidence_dict['A'],0)
xB_query = VAE_MRF.query_single_attribute(x_evidence_dict, query_attribute = 'B', evidence_attributes = ['A'], query_repetitions=10000)

x_evidence_dict = {'A': x_list[9]} 
x_evidence_dict['A'] = torch.unsqueeze(x_evidence_dict['A'],0)
xB_query = VAE_MRF.query_single_attribute(x_evidence_dict, query_attribute = 'B', evidence_attributes = ['A'], query_repetitions=10000)


#ToDO
#CS230 cmd line
#Use GPU
#Hyperparam search
#Early Stopping
#Joint Training

#Bernoulli Likelihood


