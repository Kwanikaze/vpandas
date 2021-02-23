import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from numpy import genfromtxt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def read_csv(data_dir):
  df = pd.read_csv(data_dir)
  df.dropna() #drop rows with at least one element missing
  return df

def label_encode_columns(df,columns):
  labelencoder = LabelEncoder()
  for col in columns:
    df[col] = labelencoder.fit_transform(df[col])
  return df

def one_hot_encode_columns(df,columns_to_OHE):
  df[columns_to_OHE]= df[columns_to_OHE].astype(int)
  for col in columns_to_OHE:
    col_OHE = pd.get_dummies(prefix = col,data= df[col])
    #Generate Unif[0,1] noise with the same dimension as col_OHE
    noise = np.random.uniform(low=0.0,high=1.0,size=col_OHE.shape)
    col_OHE = col_OHE + noise
    df = df.join(col_OHE)
  df.drop(columns_to_OHE,axis=1,inplace=True)
  return df

#def add_uniform_noise(df,columns):

def duplicate_dataframe(df,columns,duplications):
  df = pd.DataFrame(np.tile(df, (duplications, 1)))
  df.columns=columns
  return df


#def build_OHE_train_test(file_name,df,data_dir,duplications):

  #return train_df, test_df


