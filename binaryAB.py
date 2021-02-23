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
args = params.Params('./hyperparameters/binaryAB_sigmoid.json')

df = process.read_csv('https://raw.githubusercontent.com/Kwanikaze/vpandas/master/data/data_3.csv')
input_dims = {'A': 3,'B': 3} #dicts ordered
data2 = False

attributes = list(df.columns) #assumes each attribute has a single column
df = process.duplicate_dataframe(df, attributes, duplications=100)

num_samples = int(df.shape[0])
sample1_df = df[attributes].sample(n=num_samples, random_state=args.random_seed)
sample1_df_OHE = process.one_hot_encode_columns(sample1_df, attributes)
#print(sample1_df)

#  use gpu if available
use_gpu = False
device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
VAE_MRF = model.VariationalAutoencoder_MRF(sample1_df_OHE, attributes,input_dims, num_samples,args)
VAE_MRF = VAE_MRF.to(device)
VAE_MRF.train_marginals()

model.trainVAE_MRF(VAE_MRF,attributes,sample1_df_OHE)

checks.graphLatentSpace(VAE_MRF,sample1_df,sample1_df_OHE,attributes,num_samples,args)

x_test = np.eye(input_dims["A"])[np.arange(input_dims["A"])]  # Test data (one-hot encoded)
if data2:
  x_test = np.repeat(x_test, [5,5],axis=0)
x_test = Variable(torch.from_numpy(x_test))
x_test = x_test.to(device)

print("Print prediction results for A only:")
for x in x_test:
    print("\tInput: {} \t Output: {}".format(x.cpu().detach().numpy(), np.round(VAE_MRF.forward_single_attribute(x=x.float(), attribute='A')[0].cpu().detach().numpy(),decimals=2)))

print("Print prediction results for B only:")
for x in x_test:
    print("\tInput: {} \t Output: {}".format(x.cpu().detach().numpy(), np.round(VAE_MRF.forward_single_attribute(x=x.float(), attribute='B')[0].cpu().detach().numpy(),decimals=2)))


x_evidence_dict = {'A': x_test[0]} #Evidence is A=0
xB_query = VAE_MRF.query_single_attribute(x_evidence_dict, query_attribute = 'B', evidence_attributes = ['A'], query_repetitions=10000)

x_evidence_dict = {'A': x_test[1]}
xB_query = VAE_MRF.query_single_attribute(x_evidence_dict, query_attribute = 'B', evidence_attributes = ['A'], query_repetitions=10000)

x_evidence_dict = {'A': x_test[2]}
xB_query = VAE_MRF.query_single_attribute(x_evidence_dict, query_attribute = 'B', evidence_attributes = ['A'], query_repetitions=10000)

x_evidence_dict = {'B': x_test[0]}
xB_query = VAE_MRF.query_single_attribute(x_evidence_dict, query_attribute = 'A', evidence_attributes = ['B'], query_repetitions=10000)

x_evidence_dict = {'B': x_test[1]}
xB_query = VAE_MRF.query_single_attribute(x_evidence_dict, query_attribute = 'A', evidence_attributes = ['B'], query_repetitions=10000)

x_evidence_dict = {'B': x_test[2]}
xB_query = VAE_MRF.query_single_attribute(x_evidence_dict, query_attribute = 'A', evidence_attributes = ['B'], query_repetitions=10000)


#ToDO
#Compare to ppandas
#Query A|B
#Query C|A,B
#10 cardinality A,B
#One categorical, one discrete
#Hyperparameter Tuning by adding Gaussian noise

#Bernoulli Likelihood


