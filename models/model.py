from models import marginalVAE
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal

class VariationalAutoencoder_MRF(nn.Module):
    def __init__(self,df,attributes, input_dims, num_samples, args):
        super().__init__()
        self.df = df
        self.latent_dims = args.latent_dims #later dict so latent_dim different for each attribute 
        self.input_dims = input_dims # dict # possible outcomes for each categorical attribute
        self.num_samples = num_samples
        self.attributes = attributes
        
        #Dictionary of Marginal VAEs for each attribute
        self.marginalVAEs = {a: marginalVAE.marginalVAE(self.input_dims[a], self.latent_dims, args) for a in attributes}
        #need to enforce order of attributes A1,B2,C3 to be able to marginalize
        #ordered dict
        #https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
        self.mu_emp = {} # vector of mu's for all attributes
        self.covar_emp = {} # covariance matrix for all attributes
        self.mu_dict = {} # dict of tensors of mu's for each attribute
        self.covar_dict = {} # dict of tensors of variance for each pair of attributes
        #self.VAE_A = marginalVAE.marginalVAE(self.input_dims["A"],num_samples,args)
        #self.VAE_B = marginalVAE.marginalVAE(self.input_dims["B"],num_samples,args)
        #Emperical mu and logvar
        #self.muA_emp = 0
        #self.muB_emp = 0
        #self.logvarA_emp = 0
        #self.logvarB_emp = 0
        #self.covarianceAB =torch.randn(size=(self.latent_dims,self.latent_dims))
        #Need to make covarianceAB a parameter, requires_grad=True


    #Stage 1 - Train Marginal VAEs and then freeze parameters
    def train_marginals(self):
        for attribute_key in self.marginalVAEs:
          trainVAE(self.marginalVAEs[attribute_key], self.df, attribute_key)
          for param in self.marginalVAEs[attribute_key].parameters():
            param.requires_grad = False
        print('Parameters for Marginal VAEs fixed')
        #trainVAE(self.VAE_A,self.df, 'A')
        #trainVAE(self.VAE_B,self.df, 'B')
        #for param in self.VAE_A.parameters():
        #  param.requires_grad = False
        #for param in self.VAE_B.parameters():
        #  param.requires_grad = False



    def emp_covariance(self,x_dict):
      z_dict = {a: self.latent(x_dict[a].float(), attribute=a, add_variance=True)  for a in self.attributes}
      np_z_dict = {a: z_dict[a].cpu().detach().numpy().reshape(self.num_samples,self.latent_dims) for a in self.attributes}  
      
      #zA = self.latent(xA.float(), attribute='A', add_variance=True)
      #np_zA = zA.cpu().detach().numpy().reshape(self.num_samples,self.latent_dims)
      #zB = self.latent(xB.float(), attribute='B', add_variance=True)
      #np_zB = zB.cpu().detach().numpy().reshape(self.num_samples,self.latent_dims)
      
      z_obs = np.concatenate(tuple(np_z_dict.values()),axis=1)
      # z_obs = np.concatenate((np_zA, np_zB),axis=1) #(num_samples,num_attrs*latent_dims) #(500,4)
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
      #for attribute in self.attributes:
      #  self.mu_emp[attribute]
      #z_obs_mean = np.mean(z_obs,axis=0) #Sample mean, [num_attrs*latent_dims,] (4,)
      #z_obs_cov = np.cov(z_obs,rowvar=False) #Sample covariance, rowvar=false means each column a variable
      #if self.latent_dims == 1:
      #  self.muA_emp= torch.unsqueeze(torch.unsqueeze(torch.tensor(z_obs_mean[0]).float(),0),0)
      #  self.muB_emp = torch.unsqueeze(torch.unsqueeze(torch.tensor(z_obs_mean[1]).float(),0),0)
      #  self.logvarA_emp = torch.unsqueeze(torch.unsqueeze(torch.tensor(z_obs_cov[0][0]).float(),0),0)
      #  self.logvarB_emp = torch.unsqueeze(torch.unsqueeze(torch.tensor(z_obs_cov[1][1]).float(),0),0)
      #  self.covarianceAB = torch.unsqueeze(torch.unsqueeze(torch.tensor(z_obs_cov[1][0]).float(),0),0)
      #elif self.latent_dims > 1:
      #  i = self.latent_dims
    #self.muA_emp= torch.unsqueeze(torch.tensor(z_obs_mean[0:i]).float(),1)
      #  self.muB_emp= torch.unsqueeze(torch.tensor(z_obs_mean[i:i+i]).float(),1)
      #  self.logvarA_emp = torch.tensor(z_obs_cov[0:i,0:i]).float()
      #  self.logvarB_emp = torch.tensor(z_obs_cov[i:i+i,i:i+i]).float()
      #  self.covarianceAB = torch.tensor(z_obs_cov[0:i,i:i+i]).float()


      print("Means of Z")
      print(self.mu_emp)
      for a in self.attributes:
        print(self.mu_dict[a])
      #print(self.muA_emp)
      #print(self.muB_emp)
      print("Covariance Matrix of Z")
      print(self.covar_emp)
      for a in self.attributes:
        for b in self.attributes:
          print(a+b)
          print(self.covar_dict[a+b])
      #print(self.logvarA_emp)
      #print(self.logvarB_emp)
      #print(self.covarianceAB)


    # Conditional of Multivariate Gaussian
    # Attribute is the one to be estimated
    # z is evidence encoded, 10000 times
    #self.mu_emp,self.logvar_emp,z_evidence,evidence_attributes, query_attribute
    def conditional(self, z_evidence_dict, evidence_attributes, query_attribute,query_repetitions):      
        relevant_attributes = self.attributes.copy() #keeps order
        for a in self.attributes:
          if (a not in evidence_attributes) and (a != query_attribute):
            relevant_attributes.remove(a)
        
        evidence_attributes = relevant_attributes.copy()
        evidence_attributes.remove(query_attribute)

        non_relevant_attributes = self.attributes.copy()
        for a in self.attributes:
          if a in relevant_attributes:
            non_relevant_attributes.remove(a)
         

        #marginalize out non relevant attributes, keeps dict order the same
        mu_relevant = self.mu_dict.copy()
        for k,v in self.mu_dict.items():
          for nr in non_relevant_attributes:
            if (nr == k) and (k in mu_relevant.keys()):
              del mu_relevant[k]
        #mu_relevant = {r: self.mu_dict[r] for r in relevant_attributes} #efficient but order changed

        covar_relevant = self.covar_dict.copy()
        for k,v in self.covar_dict.items():
          for nr in non_relevant_attributes:
            if (nr in k) and (k in covar_relevant.keys()):
              del covar_relevant[k]
        print("mu_relevant")
        print(mu_relevant)
        print("covar_relevant")
        print(covar_relevant)
        
        #Unpack z_evidence_dict into single tensor with dimension (,1), keep order
        evidence_tensors = []
        for a in self.attributes:
          if a in z_evidence_dict.keys():
            evidence_tensors.append(z_evidence_dict[a]) #latent_dim,1
        z = torch.cat(evidence_tensors,dim=0)
        z = z.repeat_interleave(query_repetitions,dim=1) #[latent_dims*evidence_vars,10000]
        #z = z.transpose(0,1) 
        #print("z evidence shape 10k")
        #print(z.shape)

        #Following notation here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
        mu1 = mu_relevant[query_attribute]
        #mu2_dict = mu_relevant.copy()
        #del mu2_dict[query_attribute]
        mu2_tensors = []
        #for v in mu2_dict.values():
        #  mu2_tensors.append(v)
        for e in evidence_attributes:
          mu2_tensors.append(mu_relevant[e])
        mu2 = torch.cat(mu2_tensors,dim=0)
        print("mu2 shape")
        print(mu2.shape)

        sigma11 = covar_relevant[query_attribute+query_attribute]
        #sigma22_dict = covar_relevant.copy()
        #for k,v in covar_relevant.items():
        #  if (query_attribute in k) and (k in sigma22_dict.keys()):
        #    del sigma22_dict[k]
        # pre-allocate size of sigma22
        q = self.latent_dims
        N_minus_q = q*len(evidence_attributes)
        sigma22 = torch.empty(N_minus_q,N_minus_q)

        
        i = 0 #row counter
        for e_i in evidence_attributes:
          j = 0 #column counter
          for e_j in evidence_attributes:
            sigma22[i:i+q,j:j+q] = covar_relevant[e_i + e_j] #how to backprop?
            j += q
          i += q
        #sigma22_tensors.append(sigma22_dict[e_i + e_j])
        #sigma22 = torch.cat(sigma22_tensors, dim=1) # latent_dims, latent_dims* # evidence_attributes^2
        #sigma22 = sigma22.reshape(latent_dims*len(evidence_attributes), latent_dims*len(evidence_attributes)) #latent_dims*evidence attributes, latent_dims*evidence attributes
        sigma12 = torch.empty(q, N_minus_q)
        i=0
        for e in evidence_attributes:
          sigma12[0:q, i:i+q] = covar_relevant[query_attribute + e]
          i += q
        
        sigma21 = torch.empty(N_minus_q, q)
        i = 0
        for e in evidence_attributes:
          sigma12[i:i+q,0:q] = covar_relevant[e + query_attribute]
          i += q

        print("sigma11 shape")
        print(sigma11.shape)
        print("sigma22 shape")
        print(sigma22.shape)
        print("sigma12 shape")
        print(sigma12.shape)
        print("sigma21 shape")
        print(sigma21.shape)


        covarianceA = varA
        covarianceB = varB
        if query_attribute == 'A':
          mu_cond = muA + torch.matmul(torch.matmul(self.covarianceAB, 
                                                    torch.inverse(covarianceB)),
                                      (z - muB)) # z is zB
          var_cond = covarianceA - torch.matmul(torch.matmul(self.covarianceAB, 
                                                                torch.inverse(covarianceB)),
                                                  torch.transpose(self.covarianceAB,0,1))
          #var_cond = var_cond + 20*torch.eye(latent_dims) # regularization
        elif query_attribute == 'B':
          mu_cond = muB + torch.matmul(torch.matmul(torch.transpose(self.covarianceAB,0,1),
                                                    torch.inverse(covarianceA)), 
                                       (z - muA)) # z is zA
          var_cond = covarianceB - torch.matmul(torch.matmul(torch.transpose(self.covarianceAB,0,1), 
                                                              torch.inverse(covarianceA)),
                                                 self.covarianceAB)

        # METHOD1: re-parameterization trick to sample z_cond
        eps = torch.randn_like(mu_cond) #64x3x1, 64x3x3 if use var_cond
        z_cond = mu_cond + torch.matmul(torch.sqrt(torch.diag(var_cond)),eps) #64x3x1 #2,10000 
        z_cond = torch.transpose(z_cond,0,1) 
        #print("z_cond shape")
        #print(z_cond.shape) #10000, latent_dims
        #z_cond = mu_cond + torch.matmul(var_cond,eps)
        #z_cond = z_cond.squeeze(2) #64x3
        #return z_cond
        #____
        from scipy.stats import multivariate_normal as mvn
        n_samps_to_draw = 10000
        mu_cond_T = torch.transpose(mu_cond,0,1)
        print(mu_cond_T[0].shape)
        z_cond = mvn(mean=mu_cond_T[0],cov=var_cond).rvs(n_samps_to_draw)
        print(z_cond.shape)
        #print("z_cond shape 2")
        #print(z_cond.shape)
        print(z_cond[0:10])
        #z_cond = mu_cond + torch.matmul(var_cond,eps)
        #z_cond = z_cond.squeeze(2) #64x3
        return torch.tensor(z_cond).float()

    #return mu, logvar
    def encode(self, x, attribute):
      return self.marginalVAEs[attribute].encode(x)
      #if attribute == 'A':
      #  return self.VAE_A.encode(x)
      #elif attribute =='B':
      #  return self.VAE_B.encode(x)
    
    #return reconstruction
    def decode(self, z, attribute):
      return self.marginalVAEs[attribute].decode(z)
      #if attribute == 'A':
      #  return self.VAE_A.decode(z)
      #elif attribute =='B':
      #  return self.VAE_B.decode(z)
    
    def latent(self,x,attribute, add_variance=True):
      z = self.marginalVAEs[attribute].latent(x, add_variance)
      #if attribute == 'A':
      #    z = self.VAE_A.latent(x, add_variance) #[latent_dims]
      #elif attribute == 'B':
      #    z = self.VAE_B.latent(x, add_variance)
      z = z.unsqueeze(1) #[latent_dims,1]
      #print("z shape in latent")
      #print(z.shape)
      return z

    #Given x, returns: reconstruction x_hat, mu, log_var
    def forward_single_attribute(self, x, attribute):
      return self.marginalVAEs[attribute].forward(x)
      #if attribute == 'A':
      #  return self.VAE_A.forward(x)
      #elif attribute == 'B':
      #  return self.VAE_B.forward(x)

    def query_single_attribute(self, x_evidence_dict, query_attribute, evidence_attributes, query_repetitions=10000):
      #x_evidence = torch.tensor(x_evidence).float()
      z_evidence_dict = {a: self.latent(torch.tensor(x_evidence_dict[a]).float(),a, add_variance=False) for a in evidence_attributes}
      z_query = self.conditional(z_evidence_dict,evidence_attributes, query_attribute,query_repetitions)
      return self.decode(z_query, query_attribute)
      
      #if evidence_attribute =='A':
      #  zA = self.latent(x_evidence,evidence_attribute, add_variance,query_repetitions)
      #  #Use emperical mus and logvars
      #  zB = self.conditional(self.muA_emp, self.logvarA_emp, self.muB_emp, self.logvarB_emp, zA, attribute='B')
      #  return self.decode(zB,attribute='B')

      #elif evidence_attribute =='B':
      #  zB = self.latent(x_evidence,evidence_attribute,add_variance,query_repetitions)
      #  zA = self.conditional(self.muA_emp, self.logvarA_emp, self.muB_emp, self.logvarB_emp, zB, attribute='A')
      #  return self.decode(zA,attribute='A')

def trainVAE_MRF(VAE_MRF,attributes,df):
    VAE_MRF.train() #set model mode to train
    #dict where each dict key is an attribute, each dict value is a np.array without axes labels
    #x_dict = {}
    #inds_dict={}

    x_dict = {a: Variable(torch.from_numpy(df.filter(like=a,axis=1).values)) for a in attributes}
    #xA = df.filter(like='A', axis=1).values
    #xB = df.filter(like='B', axis=1).values
    #print(xA.shape)

    #sample2_OHE when do BC plate
    
    #indsA = list(range(xA.shape[0]))
    #indsB = list(range(xB.shape[0]))

    #loss_hist = []
    #xA = Variable(torch.from_numpy(xA))
    #xB = Variable(torch.from_numpy(xB))
    
    #VAE_MRF.emp_covariance(attributes,xA.float(),xB.float())
    VAE_MRF.emp_covariance(x_dict)

    print("\nTraining MRF finished!")
    #print(loss_hist)

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
  #sample2_OHE when do BC plate
  
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