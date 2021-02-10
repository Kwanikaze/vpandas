import utils.process as process
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

df = process.read_csv('https://raw.githubusercontent.com/Kwanikaze/vpandas/master/data/data_2.csv')
attributes = list(df.columns) #assumes each attribute has a single column
df = process.duplicate_dataframe(df, attributes, duplications=100)

df= df.astype(int)
num_samples = 500
input_dims = {'A': 2,'B': 2}
sample1_df = df[['A','B']].sample(n=num_samples, random_state=args.random_seed)
sample1_df = process.one_hot_encode_columns(sample1_df, ['A','B'])
print(sample1_df)

#  use gpu if available
use_gpu = False
device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
VAE_MRF = model.VariationalAutoencoder_MRF(sample1_df, attributes,input_dims, num_samples,args)
VAE_MRF = VAE_MRF.to(device)
VAE_MRF.train_marginals()

model.trainVAE_MRF(VAE_MRF,attributes,sample1_df)




x_test = np.eye(input_dims["A"])[np.arange(input_dims["A"])]  # Test data (one-hot encoded)
duplicates = 5
x_test = np.repeat(x_test, [duplicates,duplicates],axis=0)
x_test = Variable(torch.from_numpy(x_test))
x_test = x_test.to(device)

print("Print prediction results for A only:")
for x in x_test:
    print("\tInput: {} \t Output: {}".format(x.cpu().detach().numpy(), np.round(VAE_MRF.forward_single_attribute(x=x.float(), attribute='A')[0].cpu().detach().numpy(),decimals=2)))

print("Print prediction results for B only:")
for x in x_test:
    print("\tInput: {} \t Output: {}".format(x.cpu().detach().numpy(), np.round(VAE_MRF.forward_single_attribute(x=x.float(), attribute='B')[0].cpu().detach().numpy(),decimals=2)))


#____
xA = sample1_df.filter(like='A', axis=1).values
xB = sample1_df.filter(like='B', axis=1).values
xA = Variable(torch.from_numpy(xA))
xB = Variable(torch.from_numpy(xB))

zA = VAE_MRF.latent(xA.float(), attribute='A',add_variance=True)
np_zA = zA.cpu().detach().numpy().reshape(num_samples,args.latent_dims)

zB = VAE_MRF.latent(xB.float(), attribute='B',add_variance=True)
np_zB = zB.cpu().detach().numpy().reshape(num_samples,args.latent_dims)


if args.latent_dims==1:
  plt.plot(np_zA, 'o', color='black',label="zA");
  plt.plot(np_zB, 's', color='red',label="zB");

  plt.title("Latent Encodings z from A and B Marginal Encoders")
elif args.latent_dims ==2:
  #plt.plot(np_zA[:,0], np_zA[:,1],'o', color='black');
  plt.plot(np_zA[:,0],np_zA[:,1], 'o', color='black',label="zA");
  plt.plot(np_zB[:,0],np_zB[:,1], 's', color='red',label="zB");
elif args.latent_dims ==3:
  from mpl_toolkits.mplot3d import Axes3D
  fig = plt.figure()
  ax = Axes3D(fig)
  #t = np.arange(1000)
  ax.scatter(np_zA[:,0], np_zA[:,1], np_zA[:,2])
plt.title("Latent Encodings z from A and B Marginal Encoders")
plt.legend()

#___

#x_test = np.eye(input_dims['A'])[np.arange(input_dims['A'])]
xA_evidence = x_test[0] #Evidence is A=0
#xA_evidence = xA_evidence.repeat(2,1)
print('A evidence input')
print(xA_evidence) #need to resize/ view for single sample, or make evidence a batch repeated

xB_query = VAE_MRF.query_single_attribute(x_evidence=xA_evidence, evidence_attribute = 'A', query_repetitions=10000)
print('B query output, first 5 rows:')
print(np.round(xB_query[0:5].cpu().detach().numpy(),decimals=2))

#Averaging all xB_query
print('xB_query mean of each column:')
print(torch.mean(xB_query,0).detach().numpy())

#Taking max of each row in xB_query and counting times each element is max
print('xB_query count of when each column is max:')
_,indices_max =xB_query.max(dim=1) 
#print(indices_max.numpy())
unique, counts = np.unique(indices_max.numpy(), return_counts=True)
print(dict(zip(unique, counts)))


print("hello")