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

# Create OHE dataset for specified attributes given a global df
def OHE_sample(sample_df, features_to_OHE: list):
  for feature in features_to_OHE:
    feature_OHE = pd.get_dummies(prefix = feature,data= sample_df[feature])
    sample_df = pd.concat([sample_df,feature_OHE],axis=1)
  sample_df.drop(features_to_OHE,axis=1,inplace=True)
  print(sample_df)
  return sample_df

#Hardcode 2x2 P(A,B)
# Load global relation from github
from numpy import genfromtxt
data_2 = genfromtxt('https://raw.githubusercontent.com/Kwanikaze/vpandas/master/data_2.csv', delimiter=',',skip_header=1)
data_2_1000 = np.tile(data_2, (100, 1))

#print(data_2.shape)
#print(data_2_1000.shape)
df = pd.DataFrame(np.tile(data_2, (100, 1)))
df.columns=['A','B']
df=df.astype(int)
#print(df)
#df.to_csv('data_2_1000rows.csv',index=False)


#df = pd.read_csv("data_2_1000rows.csv") # 3columns A,B,C that each contain values 0 to 1, block diagonal
print(df.shape)

#Create two datasets containing AB and BC
num_samples = 500
sample1_df = df[['A','B']].sample(n=num_samples, random_state=4)
print(sample1_df.shape)
print(sample1_df.head())
#sample2_df = df[['B','C']].sample(n=num_samples, random_state=3)
#print(sample2_df.head())

# Make A,B,C inputs all 8 bits
#Could add noise so not exactly OHE: 0.01...0.9...0.01
sample1_OHE = OHE_sample(sample1_df,['A','B'])
#sample2_OHE = OHE_sample(sample2_df,['B','C'])

# Could onvert pandas dataframes to list of lists of lists
# [ [[OHE A1],[OHE B1]], [[OHE A2],[OHE B2]], ...  ]
