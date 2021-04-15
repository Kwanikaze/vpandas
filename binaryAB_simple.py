import utils.process as process
import utils.checks as checks
import utils.trainer as trainer
#import models.model as model
import models.joint_model as model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
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

input_dims = {'A': 2,'B': 2}
attributes = list(df_raw.columns)
real_vars = []
cat_vars = [x for x in attributes if x not in real_vars]

df, df_OHE,mms_dict = process.preprocess(df_raw, args, real_vars, cat_vars, duplications=200)
train_df, train_df_OHE, val_df, val_df_OHE, test_df, test_df_OHE = process.split(df,df_OHE,[0.7,0.85])

#Simple, nonlinear

use_gpu = False
device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
VAE_MRF = model.VariationalAutoencoder_MRF(train_df_OHE, val_df_OHE, attributes,input_dims, args,real_vars,cat_vars,mms_dict)
VAE_MRF = VAE_MRF.to(device)
trainer.trainVAE_MRF(VAE_MRF,attributes,args)