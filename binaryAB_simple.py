import utils.process as process
import utils.checks as checks
import utils.trainer as trainer
#import models.model as model
import models.joint_model as model
#import models.joint_model_linear as model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
from scipy.stats import multivariate_normal
import utils.params as params
import sys

args = params.Params('./hyperparameters/binaryAB.json')
df_raw =  process.read_csv('https://raw.githubusercontent.com/Kwanikaze/vpandas/master/data/data_2.csv')
print("Raw Data")
print(df_raw)
prob = df_raw.groupby(['A','B']).size().div(len(df_raw))
print("Joint P(A,B)")
print(prob)
print("Conditional P(B|A)")
Aprob =  df_raw.groupby('A').size().div(len(df_raw))
#print(Aprob)
probBgivenA = df_raw.groupby(['A', 'B']).size().div(len(df_raw)).div(Aprob, axis=0, level='A')
print(probBgivenA)

print("Conditional P(A|B)")
Bprob =  df_raw.groupby('B').size().div(len(df_raw))
#print(Aprob)
probAgivenB = df_raw.groupby(['B', 'A']).size().div(len(df_raw)).div(Bprob, axis=0, level='B')
print(probAgivenB)


input_dims = {'A': 2,'B': 2}
attributes = list(df_raw.columns)
real_vars = []
cat_vars = [x for x in attributes if x not in real_vars]

df, df_OHE,mms_dict = process.preprocess(df_raw, args, real_vars, cat_vars, duplications=20)
train_df, train_df_OHE, val_df, val_df_OHE, test_df, test_df_OHE = process.split(df,df_OHE,[0.7,0.85])

#Simple, linear encoder, linear decoder, no activation functions
use_gpu = False
device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
VAE_MRF = model.VariationalAutoencoder_MRF(train_df_OHE, val_df_OHE, attributes,input_dims, args,real_vars,cat_vars,mms_dict)
VAE_MRF = VAE_MRF.to(device)
trainer.trainVAE_MRF(VAE_MRF,args)

checks.graphLatentSpace(VAE_MRF,train_df,train_df_OHE,attributes,args,cat_vars)

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

#x_test = np.eye(input_dims["B"])[np.arange(input_dims["B"])]  # Test data (one-hot encoded)
#x_test = Variable(torch.from_numpy(x_test)).to(device)

x_evidence_dict = {'A': x_test[0]} 
xB_query = VAE_MRF.query_single_attribute(x_evidence_dict, query_attribute = 'B', evidence_attributes = ['A'], query_repetitions=10000)

x_evidence_dict = {'A': x_test[1]}
xB_query = VAE_MRF.query_single_attribute(x_evidence_dict, query_attribute = 'B', evidence_attributes = ['A'], query_repetitions=10000)

x_evidence_dict = {'B': x_test[0]} 
xB_query = VAE_MRF.query_single_attribute(x_evidence_dict, query_attribute = 'A', evidence_attributes = ['B'], query_repetitions=10000)

x_evidence_dict = {'B': x_test[1]}
xB_query = VAE_MRF.query_single_attribute(x_evidence_dict, query_attribute = 'A', evidence_attributes = ['B'], query_repetitions=10000)