import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F

#Each attribute has a marginal VAE
class marginalVAE(nn.Module):
    def __init__(self,input_dims,num_samples, args, cat_var):
        super().__init__()
        self.cat_var = cat_var
        self.latent_dims = args.latent_dims
        self.learning_rate = args.learning_rate
        self.num_samples = num_samples
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.anneal_factor = args.anneal_factor
        self.variational_beta = args.variational_beta
        self.fc1 = nn.Linear(input_dims, self.latent_dims)
        self.fc_mu = nn.Linear(self.latent_dims, self.latent_dims)
        self.fc_logvar = nn.Linear(self.latent_dims, self.latent_dims)
        self.fc_out = nn.Linear(self.latent_dims,input_dims)
    
    #accepts OHE input of an attribute, returns mu and log variance
    def encode(self, x):
        h1 = torch.sigmoid(self.fc1(x))
        #h1 = = torch.relu(self.fc1(x))
        #h1 = F.leaky_relu(self.fc1(x)) #TODO leakyRELU or RELU
        return self.fc_mu(h1), self.fc_logvar(h1)

    #Given mu and logvar generates latent z
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar) 
        eps = torch.randn_like(std)
        return mu + eps*std

    #Decodes latent z into reconstruction with dimension equal to num
    def decode(self, z): #z is size [batch_size,latent_dims]
        if z.size()[0] == self.latent_dims: #resize from [3] to [1,3]
            if len(z.size()) == 1:
                z = z.view(1, self.latent_dims)
        softmax = nn.Softmax(dim=1)  #normalizes reconstruction to range [0,1] and sum to 1
        recon = softmax(self.fc_out(z))
        return recon
    
    #Given x, returns: reconstruction x_hat, mu, log_var
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    #Given x, returns latent z
    def latent(self, x, add_variance=True):
        if add_variance == False:
          mu, _ = self.encode(x)
          return mu
        else:
          mu, logvar = self.encode(x)
          z = self.reparameterize(mu, logvar)
          return z

    def vae_loss(self, epoch,batch_recon, batch_targets, mu, logvar):
        #schedule starts beta at 0 increases it to 1
        #print(anneal_factor)
        variational_beta = self.variational_beta*min(1, (epoch)/(self.num_epochs*self.anneal_factor)) #annealing schedule
        #print(batch_recon)
        #print(batch_targets)
        #if epoch % 25 == 0:
            #print(variational_beta)
        if self.cat_var:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCEWithLogitsLoss()
        CE = criterion(batch_recon, batch_targets)
        #print(CE)
        KLd = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # https://stats.stackexchange.com/questions/318748/deriving-the-kl-divergence-loss-for-vaes
        #print(KLd)
        return CE, variational_beta*KLd, CE + variational_beta*KLd