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

def one_hot_encode_columns(df,columns):
  for col in columns:
    col_OHE = pd.get_dummies(prefix = col,data= df[col])
    df = pd.concat([df,col_OHE],axis=1)
  df.drop(columns,axis=1,inplace=True)
  return df

def duplicate_dataframe(df,columns,duplications):
  df = pd.DataFrame(np.tile(df, (duplications, 1)))
  df.columns=columns
  return df


