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
args = params.Params('./hyperparameters/trinaryABC.json')

df_raw = process.read_csv('https://raw.githubusercontent.com/Kwanikaze/vpandas/master/data/data_3_ABC.csv')
input_dims = {'A': 3,'B': 3,'C': 3} #dicts ordered
attributes = list(df_raw.columns) #assumes each attribute has a single column
real_vars = []
cat_vars = [x for x in attributes if x not in real_vars]

df, df_OHE,mms_dict = process.preprocess(df_raw,args, real_vars, cat_vars, duplications=100) #mms is min_max_scalar

train_df, train_df_OHE, val_df, val_df_OHE, test_df, test_df_OHE = process.split(df,df_OHE,[0.7,0.85])
num_samples = int(train_df.shape[0])

#  use gpu if available
use_gpu = False
device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
VAE_MRF = model.VariationalAutoencoder_MRF(train_df_OHE, val_df_OHE, attributes,input_dims, num_samples,args,real_vars,cat_vars,mms_dict)
VAE_MRF = VAE_MRF.to(device)
VAE_MRF.train_marginals(args)

model.trainVAE_MRF(VAE_MRF,attributes,train_df_OHE)

checks.graphLatentSpace(VAE_MRF,train_df,train_df_OHE,attributes,num_samples,args,cat_vars)

x_test = np.eye(input_dims["B"])[np.arange(input_dims["B"])]  # Test data (one-hot encoded)
noise = np.random.uniform(low=0.0,high=1.0,size=x_test.shape)
x_test = Variable(torch.from_numpy(x_test + noise))
x_test = x_test.to(device)

print("Print prediction results for A only:")
for x in x_test:
    print("\tInput: {} \t Output: {}".format(x.cpu().detach().numpy(), VAE_MRF.forward_single_attribute(x=x.float(), attribute='A')))

print("Print prediction results for B only:")
for x in x_test:
    print("\tInput: {} \t Output: {}".format(x.cpu().detach().numpy(), VAE_MRF.forward_single_attribute(x=x.float(), attribute='B')))

print("Print prediction results for C only:")
for x in x_test:
    print("\tInput: {} \t Output: {}".format(x.cpu().detach().numpy(), VAE_MRF.forward_single_attribute(x=x.float(), attribute='C')))

x_evidence_dict = {'A': x_test[0],'B': x_test[0]}
xB_query = VAE_MRF.query_single_attribute(x_evidence_dict, query_attribute = 'C', evidence_attributes = ['A','B'], query_repetitions=10000)

x_evidence_dict = {'A': x_test[1],'B': x_test[1]}
xB_query = VAE_MRF.query_single_attribute(x_evidence_dict, query_attribute = 'C', evidence_attributes = ['A','B'], query_repetitions=10000)

x_evidence_dict = {'A': x_test[2],'B': x_test[2]}
xB_query = VAE_MRF.query_single_attribute(x_evidence_dict, query_attribute = 'C', evidence_attributes = ['A','B'], query_repetitions=10000)

x_evidence_dict = {'A': x_test[0],'B': x_test[1]}
xB_query = VAE_MRF.query_single_attribute(x_evidence_dict, query_attribute = 'C', evidence_attributes = ['A','B'], query_repetitions=10000)

