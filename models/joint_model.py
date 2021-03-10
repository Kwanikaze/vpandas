from models import marginalVAE
import utils.checks as checks
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy.stats import multivariate_normal as mvn

def trainVAE_MRF(VAE_MRF,attributes,train_df_OHE, args):
    
    VAE_MRF.train() #set model mode to train
    VAE_MRF.update_args(args)       
    optimizer = torch.optim.Adam(params = VAE_MRF.parameters(), lr = VAE_MRF.learning_rate)
    #print(list(VAE_MRF.parameters()))
    x_train_dict = {a: Variable(torch.from_numpy(VAE_MRF.train_df_OHE.filter(like=a,axis=1).values)) for a in attributes}
    x_val_dict = {a: Variable(torch.from_numpy(VAE_MRF.val_df_OHE.filter(like=a, axis=1).values)) for a in attributes}
    
    val_loss = 0

    early_stopping = EarlyStopping(patience=args.patience, verbose=False)
    for epoch in range(VAE_MRF.num_epochs):

    #VAE_MRF.joint_training(args)
    print("\nTraining MRF finished!")


class VariationalAutoencoder_MRF(nn.Module):
    def __init__(self, train_df_OHE,val_df_OHE,attributes, input_dims, args,real_vars,cat_vars,mms_dict):
        super().__init__()
        self.train_df_OHE = train_df_OHE
        self.val_df_OHE = val_df_OHE
        self.latent_dims = args.latent_dims #later dict so latent_dim different for each attribute 
        self.input_dims = input_dims # dict # possible outcomes for each categorical attribute
        self.num_samples =  int(train_df_OHE.shape[0])
        self.attributes = attributes
        self.layers = nn.ModuleDict()
        for a in self.attributes:
            self.layers['fc1' + str(a)] = nn.Linear(input_dims[a], self.latent_dims)
            self.layers['fc_mu' + str(a)] = nn.Linear(self.latent_dims, self.latent_dims)
            self.layers['fc_logvar' + str(a)] = nn.Linear(self.latent_dims, self.latent_dims)
            self.layers['fc_out' + str(a)] = nn.Linear(self.latent_dims,input_dims[a])
        
        self.real_vars = real_vars
        self.cat_vars = cat_vars
        self.learning_rate = args.learning_rate
        #for a in self.attributes: 
        #    cat_var=False
        #    if a in self.cat_vars:
        #        cat_var = True
        #    self.marginalVAEs[a] = marginalVAE.marginalVAE(self.input_dims[a], self.num_samples, args, cat_var)
        #    self.marginalVAEs[a].train()
        self.mms_dict = mms_dict #MinMaxScalar for real vars
        self.num_runs = args.num_runs # Sensitive to initial parameters, take best performing model out of runs
        self.graph_samples = args.graph_samples
        self.covarianceAB = torch.nn.Parameter(torch.randn(size=(self.latent_dims,self.latent_dims)),requires_grad=True)
    
    def update_args(self, args):
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.activation = args.activation
        self.anneal_factor = args.anneal_factor
        self.num_epochs = args.num_epochs
        self.variational_beta = args.variational_beta
    
    #def joint_training(self, args):
        # Feed in A to reconstruct A and predict B
        # Backprop on A loss and B loss and KL divergence with Beta scaled by number of reconstructions
        #loss = loss1+loss2+loss3
        #loss.backward()
        #optimizer.step()
        #attribute to hold out from input to Encoder