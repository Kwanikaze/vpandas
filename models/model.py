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
    def __init__(self,df,attributes, input_dims, num_samples, args):
        super().__init__()
        self.df = df
        self.latent_dims = args.latent_dims #later dict so latent_dim different for each attribute 
        self.input_dims = input_dims # dict # possible outcomes for each categorical attribute
        self.num_samples = num_samples
        self.attributes = attributes
        self.marginalVAEs = {a: marginalVAE.marginalVAE(self.input_dims[a], self.latent_dims, args) for a in attributes} #Dictionary of Marginal VAEs for each attribute
        self.mu_emp = {} # vector of mu's for all attributes
        self.covar_emp = {} # covariance matrix for all attributes
        self.mu_dict = {} # dict of tensors of mu's for each attribute
        self.covar_dict = {} # dict of tensors of variance for each pair of attributes
        #Need to make covarianceAB a parameter, requires_grad=True

    #Stage 1 - Train Marginal VAEs and then freeze parameters
    def train_marginals(self):
        for attribute_key in self.marginalVAEs:
          trainVAE(self.marginalVAEs[attribute_key], self.df, attribute_key)
          for param in self.marginalVAEs[attribute_key].parameters():
            param.requires_grad = False
        print('Parameters for Marginal VAEs fixed')

    def emp_covariance(self,x_dict): #x_dict['A'].shape is num_samples, input_dims
      z_dict = {a: self.latent(x_dict[a].float(), attribute=a, add_variance=True)  for a in self.attributes} #num_samples,latent_dims
      np_z_dict = {a: z_dict[a].cpu().detach().numpy().reshape(self.num_samples,self.latent_dims) for a in self.attributes}  #num_samples,latent_dims
      z_obs = np.concatenate(tuple(np_z_dict.values()),axis=1) #(num_samples,num_attrs*latent_dims)

      self.mu_emp = np.mean(z_obs,axis=0)
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


    # Conditional of Multivariate Gaussian
    def conditional(self, z_evidence_dict, evidence_attributes, query_attribute,query_repetitions):      
        relevant_attributes = self.attributes.copy() #keeps order
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
        return torch.tensor(z_cond).float(),mu_cond_T[0],var_cond

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
      return self.marginalVAEs[attribute].forward(x)


    def query_single_attribute(self, x_evidence_dict, query_attribute, evidence_attributes, query_repetitions=10000):
      z_evidence_dict = {a: self.latent(torch.tensor(x_evidence_dict[a]).float(),a, add_variance=False) for a in evidence_attributes}
      z_query, mu_cond, var_cond = self.conditional(z_evidence_dict,evidence_attributes, query_attribute,query_repetitions)
      query_recon =  self.decode(z_query, query_attribute) #10k,latent_dims
      if self.latent_dims ==2:
        checks.graphSamples(mu_cond,var_cond,z_query,evidence_attributes,query_attribute, query_repetitions)
      return query_recon 
      

def trainVAE_MRF(VAE_MRF,attributes,df):
    VAE_MRF.train() #set model mode to train
    #dict where each dict key is an attribute, each dict value is a np.array without axes labels
    x_dict = {a: Variable(torch.from_numpy(df.filter(like=a,axis=1).values)) for a in attributes}
    VAE_MRF.emp_covariance(x_dict)
    print("\nTraining MRF finished!")

def vae_loss(VAE,batch_recon, batch_targets, mu, logvar):
  criterion = nn.CrossEntropyLoss()
  CE = criterion(batch_recon, batch_targets)
  #print(CE)
  KLd = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # https://stats.stackexchange.com/questions/318748/deriving-the-kl-divergence-loss-for-vaes
  #print(KLd)
  return CE,VAE.variational_beta*KLd, CE + VAE.variational_beta*KLd

#Train marginal VAE
def trainVAE(VAE, sample1_OHE, attribute: str):
  print("\nTraining marginal VAE for " + attribute+ " started!")
  VAE.train() #set model mode to train
  optimizer = torch.optim.Adam(params = VAE.parameters(), lr = VAE.learning_rate)
  x = sample1_OHE.filter(like=attribute, axis=1).values
  
  inds = list(range(x.shape[0]))
  N = VAE.num_samples
  freq = VAE.num_epochs // 10 # floor division

  loss_hist = []
  x = Variable(torch.from_numpy(x))
  
  for epoch in range(VAE.num_epochs):
      VAE.train()
      #print('epoch' + str(epoch))
      inds = np.random.permutation(inds)
      x = x[inds]
      use_gpu=False
      device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
      x = x.to(device)
      
      loss = 0
      CE = 0
      KLd = 0
      #num_batches = N / batch_size
      for b in range(0, N, VAE.batch_size):
          #get the mini-batch
          x_batch = x[b: b+VAE.batch_size]
          #feed forward
          batch_recon,latent_mu,latent_logvar = VAE.forward(x_batch.float())
          #Convert x_batch from OHE vectors to single scalar
          # max returns index location of max value in each sample of batch 
          _, x_batch_targets = x_batch.max(dim=1)
          train_CE, train_KLd, train_loss = vae_loss(VAE,batch_recon, x_batch_targets, latent_mu, latent_logvar)
          loss += train_loss.item() / N # update epoch loss
          CE += train_CE.item() / N
          KLd += train_KLd.item() / N

          #Backprop the error, compute the gradient
          optimizer.zero_grad()
          train_loss.backward()

          #update parameters based on gradient
          optimizer.step()
          
      #Record loss per epoch        
      loss_hist.append(loss)
      
      if epoch % freq == 0:
          print('')
          print("Epoch %d/%d\t CE: %.5f, KLd: %.5f, Train loss=%.5f" % (epoch + 1, VAE.num_epochs,CE,KLd, loss), end='\t', flush=True)

          #Test with all training data
          VAE.eval()
          train_recon, train_mu, train_logvar = VAE.forward(x.float())
          _, x_targets = x.max(dim=1)
          CE_,KLd,test_loss = vae_loss(VAE,train_recon, x_targets, train_mu, train_logvar)
          print("\t CE: {:.5f}, KLd: {:.5f}, Test loss: {:.5f}".format(CE,KLd,test_loss.item()), end='')

  print("\nTraining marginal VAE for " + attribute+ " finished!")
  #print(loss_hist)