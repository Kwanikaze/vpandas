from models import marginalVAE
import utils.checks as checks
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy.stats import multivariate_normal as mvn

class VariationalAutoencoder_MRF(nn.Module):
    def __init__(self,train_df_OHE,val_df_OHE,attributes, input_dims, args,real_vars,cat_vars,mms_dict):
        super().__init__()
        self.train_df_OHE = train_df_OHE
        self.val_df_OHE = val_df_OHE
        self.latent_dims = args.latent_dims #later dict so latent_dim different for each attribute 
        self.input_dims = input_dims # dict # possible outcomes for each categorical attribute
        self.num_samples =  int(train_df_OHE.shape[0])
        self.attributes = attributes
        self.marginalVAEs = {} #Dictionary of Marginal VAEs for each attribute
        self.real_vars = real_vars
        self.cat_vars = cat_vars
        self.mms_dict = mms_dict #MinMaxScalar for real vars
        self.num_runs = args.num_runs # Sensitive to initial parameters, take best performing model out of runs
        self.mu_emp = 0 # vector of mu's for all attributes
        self.covar_emp = 0 # covariance matrix for all attributes
        self.mu_dict = {} # dict of tensors of mu's for each attribute
        self.covar_dict = {} # dict of tensors of variance for each pair of attributes
        self.emperical = args.emperical
        self.graph_samples = args.graph_samples
        if args.emperical == "False":
          self.covarianceAB = torch.nn.Parameter(torch.randn(size=(self.latent_dims,self.latent_dims)),requires_grad=True)
        #Need to make covarianceAB a parameter, requires_grad=True
  

    #Stage 1 - Train Marginal VAEs and then freeze parameters
    def train_marginals(self,args):
      learning_rates = [1e-1,1e-2,1e-3]
      batch_sizes = [128,256,512]
      activations = ['sigmoid','relu','leaky_relu']
      #epochs = [200,400,600,800]
      anneal_factors = [200,400,600,800]
      for a in self.attributes:
        print("\nTraining marginal VAE for " + a + " started!")
        cat_var=False
        if a in self.cat_vars:
          cat_var = True
        if args.hypertune == "True":
          best_val_loss_min = 1e6
          for lr in learning_rates:
            for activ in activations:
              for bs in batch_sizes:
                #for ep in epochs:
                for af in anneal_factors:
                  args.learning_rate = lr
                  args.activation = activ
                  args.batch_size = bs
                  #args.num_epochs = ep
                  args.anneal_factor = af
                  attribute_VAE = marginalVAE.marginalVAE(self.input_dims[a], self.latent_dims, args, cat_var)
                  early_VAE,val_loss_min = marginalVAE.trainVAE(attribute_VAE, self.train_df_OHE, self.val_df_OHE, a, args)
                  if val_loss_min < best_val_loss_min:
                    self.marginalVAEs[a] = early_VAE
                    best_val_loss_min = val_loss_min
                    #print("Current Best Validation Loss")
                    #print(best_val_loss_min)
          print("Best Validation Loss and Parameters")
          print(best_val_loss_min)
          print("learning rate: {}".format(self.marginalVAEs[a].learning_rate))
          print("activation function: " + self.marginalVAEs[a].activation)
          print("batch size: {}".format(self.marginalVAEs[a].batch_size))
          #print("num epochs: {}".format(self.marginalVAEs[a].num_epochs))
          print("anneal factor: {}".format(self.marginalVAEs[a].anneal_factor))
        else:
          attribute_VAE = marginalVAE.marginalVAE(self.input_dims[a], self.latent_dims, args, cat_var)
          self.marginalVAEs[a],_ = marginalVAE.trainVAE(attribute_VAE, self.train_df_OHE, self.val_df_OHE, a, args)
        print("\nTraining marginal VAE for " + a + " finished!")
        for param in self.marginalVAEs[a].parameters():
          param.requires_grad = False
      print('Parameters for Marginal VAEs fixed')

    def emp_covariance(self,x_dict): #x_dict['A'].shape is num_samples, input_dims
      z_dict = {a: self.latent(x_dict[a].float(), attribute=a, add_variance=True)  for a in self.attributes} #num_samples,latent_dims
      np_z_dict = {a: z_dict[a].cpu().detach().numpy().reshape(self.num_samples,self.latent_dims) for a in self.attributes}  #num_samples,latent_dims
      z_obs = np.concatenate(tuple(np_z_dict.values()),axis=1) #(num_samples,num_attrs*latent_dims)
      self.mu_emp = np.mean(z_obs,axis=0) #mean of each column
      self.covar_emp = np.cov(z_obs,rowvar=False)
      ld = self.latent_dims
      #dict of tensors of mu's for each attribute
      dim_counter = 0
      for a in self.attributes:
        self.mu_dict[a] = self.mu_emp[dim_counter : dim_counter + ld]
        self.mu_dict[a] = torch.unsqueeze(torch.tensor(self.mu_dict[a]).float(),1) #each entry: latent_dims,1
        dim_counter += ld
      
      i = 0 # row counter
      for a in self.attributes:
        j = 0 # column counter
        for b in self.attributes:
          self.covar_dict[a+b] = self.covar_emp[i:i+ld,j:j+ld]
          self.covar_dict[a+b] = torch.tensor(self.covar_dict[a+b]).float() #each entry: latent dim, latent dim
          j += ld
        i += ld
      '''
      print("Mu's of Z")
      print(self.mu_emp)
      for a in self.attributes:
        print(a)
        print(self.mu_dict[a])
        print(self.mu_dict[a].shape)

      print("Covariance Matrix of Z")
      print(self.covar_emp)
      for a in self.attributes:
        for b in self.attributes:
          print(a+b)
          print(self.covar_dict[a+b])
      '''

    def learned_covariance(self,x_dict):

      return

    # Conditional of Multivariate Gaussian
    def conditional(self, z_evidence_dict, evidence_attributes, query_attribute,query_repetitions):      
        relevant_attributes = self.attributes.copy() #keeps order according to input_dims
        for a in self.attributes:
          if (a not in evidence_attributes) and (a != query_attribute):
            relevant_attributes.remove(a)
        
        evidence_attributes = relevant_attributes.copy()
        evidence_attributes.remove(query_attribute)
        
        evidence_tensors = []  #Unpack z_evidence_dict into single tensor 
        for a in self.attributes:
          if a in z_evidence_dict.keys():
            evidence_tensors.append(z_evidence_dict[a]) #latent_dim,1
        z = torch.cat(evidence_tensors,dim=0) #latent_dims*evidence_vars,1

        #notation: https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
        q = self.latent_dims
        N_minus_q = q*len(evidence_attributes)
        
        mu1 = self.mu_dict[query_attribute]
        
        mu2 = torch.empty(N_minus_q,1)
        i=0
        for e in evidence_attributes:
          mu2[i:i+q, 0:1] = self.mu_dict[e]
          i += q
        '''
        print("mu2 shape")
        print(mu2.shape)
        print(mu2)
        '''
        sigma11 = self.covar_dict[query_attribute+query_attribute]

        sigma22 = torch.empty(N_minus_q,N_minus_q)
        i = 0 #row counter
        for e_i in evidence_attributes:
          j = 0 #column counter
          for e_j in evidence_attributes:
            sigma22[i:i+q,j:j+q] = self.covar_dict[e_i + e_j] #how to backprop?
            j += q
          i += q
       
        sigma12 = torch.empty(q, N_minus_q)
        i=0
        for e in evidence_attributes:
          sigma12[0:q, i:i+q] = self.covar_dict[query_attribute + e]
          i += q
        
        sigma21 = torch.empty(N_minus_q, q)
        i = 0
        for e in evidence_attributes:
          sigma21[i:i+q,0:q] = self.covar_dict[e + query_attribute]
          i += q
        '''
        print("sigma11 shape")
        print(sigma11.shape)
        print(sigma11)
        print("sigma22 shape")
        print(sigma22.shape)
        print(sigma22)
        print("sigma12 shape")
        print(sigma12.shape)
        print(sigma12)
        print("sigma21 shape")
        print(sigma21.shape)
        print(sigma21)
        '''
        mu_cond = mu1 + torch.matmul(torch.matmul(sigma12,torch.inverse(sigma22)), (z-mu2))
        var_cond = sigma11 - torch.matmul(torch.matmul(sigma12,torch.inverse(sigma22)),sigma21)

        mu_cond_T = torch.transpose(mu_cond,0,1).detach().cpu().numpy()
        z_cond = mvn(mean=mu_cond_T[0],cov=var_cond).rvs(query_repetitions) #10k,latent_dims
        '''
        print("mu_cond")
        print(mu_cond_T[0])
        print("var_cond")
        print(var_cond)
        '''
        return torch.tensor(z_cond).float(), mu_cond_T[0], var_cond

    #return mu, logvar
    def encode(self, x, attribute):
      return self.marginalVAEs[attribute].encode(x)

    
    #return reconstruction
    def decode(self, z, attribute):
      return self.marginalVAEs[attribute].decode(z)

    
    def latent(self,x,attribute, add_variance=True):
      z = self.marginalVAEs[attribute].latent(x, add_variance)
      #z = z.unsqueeze(1) #[latent_dims,1] or [num_samples,1,latent_dims]
      #print("z shape in latent")
      #print(z.shape)
      return z

    #Given x, returns: reconstruction x_hat, mu, log_var
    def forward_single_attribute(self, x, attribute):
      q = self.marginalVAEs[attribute].forward(x)
      if attribute in self.real_vars:
        q = self.mms_dict[attribute].inverse_transform(q[0].reshape(1,-1))
      else:
        q = np.round(q[0],decimals=2)
      return q


    def query_single_attribute(self, x_evidence_dict, query_attribute, evidence_attributes, query_repetitions=10000):
      z_evidence_dict = {a: self.latent(torch.tensor(x_evidence_dict[a]).float(),a, add_variance=False) for a in evidence_attributes}#Could be add_variance=False
      z_query, mu_cond, var_cond = self.conditional(z_evidence_dict,evidence_attributes, query_attribute,query_repetitions)
      query_recon =  self.decode(z_query, query_attribute) #10k, input_dims of query attribute
      _, recon_max_idxs = query_recon.max(dim=1)
      if self.latent_dims ==2 and self.graph_samples == "True":
        checks.graphSamples(mu_cond,var_cond,z_query,recon_max_idxs.cpu().detach().numpy(),evidence_attributes,query_attribute, query_repetitions,self.cat_vars)
      
      print('Evidence input')
      print(x_evidence_dict)

      #print('{} query output, first 5 rows:'.format(str(query_attribute)))
      #print(np.round(query_recon[0:5].cpu().detach().numpy(),decimals=2))

      #print('Mean of each column:')
      #print(torch.mean(query_recon,0).detach().numpy())

      #Taking max of each row and counting times each element is max
      print("P(" + str(query_attribute) +"|" + str(evidence_attributes).lstrip('[').rstrip(']') + ")")
      if query_attribute in self.cat_vars:
        _,indices_max =query_recon.max(dim=1) 
        unique, counts = np.unique(indices_max.numpy(), return_counts=True)
        print(dict(zip(unique, counts)))   
      else:
        #query_recon = torch.mean(query_recon)
        #query_recon = query_recon.cpu().detach().numpy()
        query_recon = query_recon.cpu().detach().numpy()
        query_recon = self.mms_dict[query_attribute].inverse_transform(query_recon.reshape(-1, 1))
        query_recon = np.mean(query_recon)
        print(float(query_recon))
      return query_recon
      
def trainVAE_MRF(VAE_MRF,attributes,df,args):
  VAE_MRF.train() #set model mode to train
  #dict where each dict key is an attribute, each dict value is a np.array without axes labels
  x_dict = {a: Variable(torch.from_numpy(df.filter(like=a,axis=1).values)) for a in attributes}
  if VAE_MRF.emperical == "True":
    VAE_MRF.train_marginals(args)
    VAE_MRF.emp_covariance(x_dict)
  else:
    VAE_MRF.learned_covariance(x_dict)
  print("\nTraining MRF finished!")

