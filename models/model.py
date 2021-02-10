from models import marginalVAE
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal

class VariationalAutoencoder_MRF(nn.Module):
    def __init__(self,df,attributes, input_dims, num_samples,args):
        super().__init__()
        self.df = df
        self.latent_dims = args.latent_dims
        self.input_dims = input_dims # dict with # possible outcomes for each categorical attribute
        self.num_samples = num_samples
        self.attributes = attributes
        #Dictionary of Marginal VAEs for each attribute
        #self.marginalVAEs = {a: marginalVAE.marginalVAE(self.input_dims[a], self.latent_dims) for a in attributes}
        #self.mu_emp = {} #dict of list of mu's for each attribute
        #self.logvar_emp = {} #dict of list of log variance for each pair of attributes
        self.VAE_A = marginalVAE.marginalVAE(self.input_dims["A"],num_samples,args)
        self.VAE_B = marginalVAE.marginalVAE(self.input_dims["B"],num_samples,args)
        #Emperical mu and logvar
        self.muA_emp = 0
        self.muB_emp = 0
        self.logvarA_emp = 0
        self.logvarB_emp = 0
        self.covarianceAB =torch.randn(size=(self.latent_dims,self.latent_dims))
        #self.covarianceAB = torch.nn.Parameter(self.covarianceAB,requires_grad=False)


    #Stage 1 - Train Marginal VAEs and then freeze parameters
    def train_marginals(self):
        trainVAE(self.VAE_A,self.df, 'A')
        trainVAE(self.VAE_B,self.df, 'B')
        for param in self.VAE_A.parameters():
          param.requires_grad = False
        for param in self.VAE_B.parameters():
          param.requires_grad = False
        print('Parameters for Marginal VAEs fixed')


    def emp_covariance(self,attributes,xA,xB):
      zA = self.latent(xA.float(), attribute='A', add_variance=True)
      np_zA = zA.cpu().detach().numpy().reshape(self.num_samples,self.latent_dims)
      zB = self.latent(xB.float(), attribute='B', add_variance=True)
      np_zB = zB.cpu().detach().numpy().reshape(self.num_samples,self.latent_dims)
      
      z_obs = np.concatenate((np_zA, np_zB),axis=1) #(num_samples,num_attrs*latent_dims) #(500,4)
      z_obs_mean = np.mean(z_obs,axis=0) #Sample mean, [num_attrs*latent_dims,] (4,)
      z_obs_cov = np.cov(z_obs,rowvar=False) #Sample covariance, rowvar=false means each column a variable
      mean_counter = 0
      #dictionary of emperical means of each attribute
      #for attribute in self.attributes:
      #  self.mu_emp[attribute]
      if self.latent_dims == 1:
        self.muA_emp= torch.unsqueeze(torch.unsqueeze(torch.tensor(z_obs_mean[0]).float(),0),0)
        self.muB_emp = torch.unsqueeze(torch.unsqueeze(torch.tensor(z_obs_mean[1]).float(),0),0)
        self.logvarA_emp = torch.unsqueeze(torch.unsqueeze(torch.tensor(z_obs_cov[0][0]).float(),0),0)
        self.logvarB_emp = torch.unsqueeze(torch.unsqueeze(torch.tensor(z_obs_cov[1][1]).float(),0),0)
        self.covarianceAB = torch.unsqueeze(torch.unsqueeze(torch.tensor(z_obs_cov[1][0]).float(),0),0)
      elif self.latent_dims == 2:
        self.muA_emp= torch.unsqueeze(torch.tensor(z_obs_mean[0:2]).float(),1)
        self.muB_emp= torch.unsqueeze(torch.tensor(z_obs_mean[2:4]).float(),1)
        self.logvarA_emp = torch.tensor(z_obs_cov[0:2,0:2]).float()
        self.logvarB_emp = torch.tensor(z_obs_cov[2:4,2:4]).float()
        self.covarianceAB = torch.tensor(z_obs_cov[0:2,2:4]).float()


      print("Means of zA,zB")
      print(z_obs_mean)
      #print(self.muA_emp)
      #print(self.muB_emp)
      print("Covariance Matrix zAzB")
      print(z_obs_cov)
      #print(self.logvarA_emp)
      #print(self.logvarB_emp)
      #print(self.covarianceAB)


    # Conditional of Multivariate Gaussian: matrix cookbook 353 and 354
    # Attribute is the one to be estimated
    # z is evidence encoded
    def conditional(self, muA, varA, muB, varB, z, attribute):
        #covarianceA = torch.diag_embed(varA) #Convert var vector to diagonal matrix
        #covarianceB = torch.diag_embed(varB) #batch_size,3,3
        covarianceA = varA
        covarianceB = varB
        #___
        #print("covarianceAB")
        #print(self.covarianceAB)
        #self.covarianceAB = torch.log(self.covarianceAB)
        #print(self.covarianceAB)
        #covarianceA = torch.log(covarianceA)
        #covarianceB = torch.log(covarianceB)
        #___
        #muA = muA.unsqueeze(2) #batch_size,latent dims
        #muB = muB.unsqueeze(2)
        #z = z.unsqueeze(2)
        if attribute == 'A':
          mu_cond = muA + torch.matmul(torch.matmul(self.covarianceAB, 
                                                    torch.inverse(covarianceB)),
                                      (z - muB)) # z is zB
          var_cond = covarianceA - torch.matmul(torch.matmul(self.covarianceAB, 
                                                                torch.inverse(covarianceB)),
                                                  torch.transpose(self.covarianceAB,0,1))
          #var_cond = var_cond + 20*torch.eye(latent_dims) # regularization
        elif attribute == 'B':
          print(self.covarianceAB.size()) #[2,2]
          print(torch.transpose(self.covarianceAB,0,1).size()) #[2,2]
          print(torch.inverse(covarianceA).size()) #[2,2]
          print(z.size()) #[2,10000] #zA
          print(muA.size()) #[2,1]
          test = torch.matmul(torch.transpose(self.covarianceAB,0,1),torch.inverse(covarianceA))
          test2 = z - muA #[2,10000]
          test3 = torch.matmul(test,test2) #[2,10000]
          print("test3.shape")
          print(test3.shape)
          print("muB.shape")
          print(muB.shape)
          #mu_cond = muB + torch.matmul(torch.matmul(torch.transpose(self.covarianceAB,0,1),#,0,1) before
          #                                          torch.inverse(covarianceA)), 
          #                             (z - muA)) # z is zA
          #var_cond = covarianceB - torch.matmul(torch.matmul(torch.transpose(self.covarianceAB,0,1), 
          #                                                    torch.inverse(covarianceA)),
          #                                       self.covarianceAB)
          #__
          from scipy.special import logsumexp
          #def log_space_product(A,B):
          #    Astack = np.stack([A]*A.shape[0]).transpose(2,1,0)
          #    Bstack = np.stack([B]*B.shape[1]).transpose(1,0,2)
          #    return logsumexp(Astack+Bstack, axis=0)
          def log_space_product(A,B):
            return np.log(np.dot(np.exp(A), np.exp(B)))
          print("transpose covarianceAB")
          print(torch.transpose(self.covarianceAB,0,1).numpy())
          print("inverse covariance A")
          print(torch.inverse(covarianceA).numpy())
          test4 = torch.tensor(log_space_product(torch.transpose(self.covarianceAB,0,1).numpy(),
                                          torch.inverse(covarianceA).numpy())) #2x2

          #x = torch.transpose(x,0,1) #[10000, 2])
          print(test4.shape) #2x2
          x=z-muA #[2, 10000]
          #print(x.shape)
          print(test4.numpy().shape)
          print(x.numpy().shape)
          print(test4)
          print(x)
          #test55 = torch.matmul()
          test5 = log_space_product(test4.numpy(), x.numpy()) #2,10000
          print(test5)
          print("test5.shape")
          print(test5.shape)
          mu_cond = muB + torch.exp(torch.tensor(log_space_product(torch.tensor(log_space_product(torch.transpose(self.covarianceAB,0,1),
                                          torch.inverse(covarianceA))), 
                              (z - muA))))
          logvar_cond = covarianceB - torch.tensor(log_space_product(torch.tensor(log_space_product(
                      torch.transpose(self.covarianceAB,0,1), torch.inverse(covarianceA))),self.covarianceAB))
          #print(var_cond)
          #logvar_cond = torch.log(var_cond)
          #__
              # var_cond is not a diagonal covariance matrix
          #var_cond = var_cond + 20*torch.eye(latent_dims)

        # METHOD1: re-parameterization trick to sample z_cond
        eps = torch.randn_like(mu_cond) #64x3x1, 64x3x3 if use var_cond
        print("mu_cond shape")
        print(mu_cond.shape)
        #print(mu_cond_new.shape)
        #print(var_cond.shape)
        print(logvar_cond.shape)
        z_cond = mu_cond + torch.matmul(torch.sqrt(torch.diag(var_cond)),eps) #64x3x1 #2,10000 
        print(logvar_cond)
        z_cond = mu_cond + torch.matmul(torch.exp(0.5*logvar_cond),eps)
        z_cond = torch.transpose(z_cond,0,1) #10000,2
        print("z_cond shape")
        print(z_cond.shape)
        #z_cond = mu_cond + torch.matmul(var_cond,eps)
        #z_cond = z_cond.squeeze(2) #64x3
        return z_cond

    #return mu, logvar
    def encode(self, x, attribute):
      if attribute == 'A':
        return self.VAE_A.encode(x)
      elif attribute =='B':
        return self.VAE_B.encode(x)
      raise Exception('Invalid attribute {} provided.'.format(x))
    
    #return reconstruction
    def decode(self, z, attribute):
      if attribute == 'A':
        return self.VAE_A.decode(z)
      elif attribute =='B':
        return self.VAE_B.decode(z)
      raise Exception('Invalid attribute {} provided.'.format(x))
    
    #Given xA, xB and attribute to reconstruct, return reconstruction
    def forward(self, xA, xB, attribute):
      muA, logvarA = self.encode(xA, attribute='A') #logvar is size [64,3]
      muB, logvarB = self.encode(xB, attribute='B')
      #When given both xA and xB, need to recalculate mu's and logvar's??
      #self.emp_covariance(xA,xB)


      # Take batch emperical average of mus and logvars
      #size_placeholder = muA.size() #[batch_size,latent_dims]
      #muA_emp = torch.mean(muA,0,keepdim=True).repeat(size_placeholder,1) #(batchsize,latent_dims) all repeated values of avg
      #logvarA_emp = torch.mean(logvarA,0,keepdim=True).repeat(size_placeholder,1)
      #muB_emp = torch.mean(muB,0,keepdim=True).repeat(size_placeholder,1)
      #logvarB_emp = torch.mean(logvarB,0,keepdim=True).repeat(size_placeholder,1)
      if attribute == 'A':
        zB = self.VAE_B.reparameterize(muB, logvarB)
        zA = self.conditional(self.muA_emp, self.logvarA_emp, self.muB_emp, self.logvarB_emp, zB, attribute)
        return self.decode(zA,attribute), self.muA_emp, self.logvarA_emp #should error use emperical avg or not?
      elif attribute == 'B':
        zA = self.VAE_A.reparameterize(muA, logvarA)
        zB = self.conditional(self.muA_emp, self.logvarA_emp, self.muB_emp, self.logvarB_emp, zA, attribute)
        return self.decode(zB,attribute), self.muB_emp, self.logvarB_emp
      raise Exception('Invalid attribute {} provided.'.format(attribute))

    def latent(self,x,attribute, add_variance=True, query_repetitions=1):
      if attribute == 'A':
          z = self.VAE_A.latent(x, add_variance) #[latent_dims]
      elif attribute == 'B':
          z = self.VAE_B.latent(x, add_variance)
      z = z.unsqueeze(1) #[latent_dims,1]
      z = z.repeat_interleave(query_repetitions,dim=1) #[latent_dims,10000]
      print("zA shape")
      print(z.shape)
      return z
      raise Exception('Invalid attribute {} provided.'.format(attribute))

    #Given x, returns: reconstruction x_hat, mu, log_var
    def forward_single_attribute(self, x, attribute):
      if attribute == 'A':
        return self.VAE_A.forward(x)
      elif attribute == 'B':
        return self.VAE_B.forward(x)
      raise Exception('Invalid attribute {} provided.'.format(x))

    def query_single_attribute(self, x_evidence, evidence_attribute, query_repetitions=10000):
      add_variance=False
      x_evidence = torch.tensor(x_evidence).float()
      if evidence_attribute =='A':
        zA = self.latent(x_evidence,evidence_attribute, add_variance,query_repetitions)
        #Use emperical mus and logvars
        zB = self.conditional(self.muA_emp, self.logvarA_emp, self.muB_emp, self.logvarB_emp, zA, attribute='B')
        return self.decode(zB,attribute='B')

      elif evidence_attribute =='B':
        zB = self.latent(x_evidence,evidence_attribute,add_variance,query_repetitions)
        zA = self.conditional(self.muA_emp, self.logvarA_emp, self.muB_emp, self.logvarB_emp, zB, attribute='A')
        return self.decode(zA,attribute='A')

def trainVAE_MRF(VAE_MRF,attributes,df):
    VAE_MRF.train() #set model mode to train
    xA = df.filter(like='A', axis=1).values
    xB = df.filter(like='B', axis=1).values
    #print(xA.shape)

    #sample2_OHE when do BC plate
    
    indsA = list(range(xA.shape[0]))
    indsB = list(range(xB.shape[0]))

    loss_hist = []
    xA = Variable(torch.from_numpy(xA))
    xB = Variable(torch.from_numpy(xB))
    
    VAE_MRF.emp_covariance(attributes,xA.float(),xB.float())

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
          # Error
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

          #print('Visualize ' + attribute + 'predictions')
          #print(train_recon[0:5])
          #print(x_targets[0:5])

  print("\nTraining marginal VAE for " + attribute+ " finished!")
  #print(loss_hist)