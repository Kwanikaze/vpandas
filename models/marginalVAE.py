import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F

#import pytorchtools.EarlyStopping as EarlyStopping
from .pytorchtools import EarlyStopping

#Each attribute has a marginal VAE
class marginalVAE(nn.Module):
    def __init__(self,input_dims,num_samples, args, cat_var):
        super().__init__()
        self.cat_var = cat_var #True or False
        self.latent_dims = args.latent_dims
        self.learning_rate = args.learning_rate
        self.num_samples = num_samples
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.anneal_factor = args.anneal_factor
        self.variational_beta = args.variational_beta
        self.activation = args.activation
        self.early_stopping = args.early_stopping
        self.fc1 = nn.Linear(input_dims, self.latent_dims)
        self.fc_mu = nn.Linear(self.latent_dims, self.latent_dims)
        self.fc_logvar = nn.Linear(self.latent_dims, self.latent_dims)
        #self.fc1 = nn.Linear(input_dims, 50)
        #self.fc_mu = nn.Linear(50, self.latent_dims)
        #self.fc_logvar = nn.Linear(50, self.latent_dims)
        self.fc_out = nn.Linear(self.latent_dims,input_dims)
    
    def update_args(self, args):
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.activation = args.activation
        self.anneal_factor = args.anneal_factor
        self.num_epochs = args.num_epochs

    #accepts OHE input of an attribute, returns mu and log variance
    def encode(self, x):
        if self.activation == "sigmoid":
            h1 = torch.sigmoid(self.fc1(x))
        elif self.activation == "relu":
            h1 = torch.relu(self.fc1(x))
        elif self.activation == "leaky_relu":
            h1 = F.leaky_relu(self.fc1(x))
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
        if self.cat_var:
            softmax = nn.Softmax(dim=1)  #normalizes reconstruction to range [0,1] and sum to 1
            return softmax(self.fc_out(z)) #recon
        else:
            return torch.sigmoid(self.fc_out(z)) #BCE
            #return self.fc_out(z) #MSE
    
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

    def vae_loss(self, epoch, batch_recon, batch_targets, mu, logvar):
        #schedule starts beta at 0 increases to 1
        #print(anneal_factor)
        variational_beta = self.variational_beta*min(1, epoch/(self.num_epochs*self.anneal_factor)) #annealing schedule
        #variational_beta = self.variational_beta
        if self.cat_var:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCELoss()
            #criterion = nn.MSELoss()
            batch_recon = batch_recon.double()
        #print(batch_recon)
        #print(batch_targets)
        CE = criterion(batch_recon, batch_targets)
        #print(CE)
        KLd = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # https://stats.stackexchange.com/questions/318748/deriving-the-kl-divergence-loss-for-vaes
        #print(KLd)
        return CE, variational_beta*KLd, CE + variational_beta*KLd

#Train marginal VAE
def trainVAE(VAE, train_df_OHE,val_df_OHE, attribute,args):
    VAE.update_args(args)
    #print("\nTraining marginal VAE for " + attribute + " started!")
    use_gpu=False
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    VAE.train() #set model mode to train
    optimizer = torch.optim.Adam(params = VAE.parameters(), lr = VAE.learning_rate)
    
    x = train_df_OHE.filter(like=attribute, axis=1).values
    inds = list(range(x.shape[0]))
    N = VAE.num_samples
    freq = VAE.num_epochs // 10 # floor division

    x = Variable(torch.from_numpy(x))
        
    val = val_df_OHE.filter(like=attribute, axis=1).values
    val = Variable(torch.from_numpy(val))
    val = val.to(device)

    train_loss_hist = []
    val_loss_hist = []
    val_loss = 0
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=args.patience, verbose=False)
    for epoch in range(VAE.num_epochs):
        VAE.train()
        #print('epoch' + str(epoch))
        inds = np.random.permutation(inds)
        x = x[inds]
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
            if VAE.cat_var:
                _, x_batch_targets = x_batch.max(dim=1) # indices for categorical
            else:
                x_batch_targets = x_batch  #values for real valued
            train_CE, train_KLd, train_loss = VAE.vae_loss(epoch,batch_recon, x_batch_targets, latent_mu, latent_logvar)
            loss += train_loss.item() / N # update epoch loss
            CE += train_CE.item() / N
            KLd += train_KLd.item() / N

            #Backprop the error, compute the gradient
            optimizer.zero_grad()
            train_loss.backward()

            #update parameters based on gradient
            optimizer.step()
            
        #Record loss per epoch        
        train_loss_hist.append(loss)
        
        #if epoch % freq == 0:
        #print('')
        #print("Epoch %d/%d\t CE: %.5f, KLd: %.5f, Train loss=%.5f" % (epoch + 1, VAE.num_epochs,CE,KLd, loss), end='\t', flush=True)

        #Test with all validation data
        VAE.eval()
        val_recon, val_mu, val_logvar = VAE.forward(val.float())          
        if VAE.cat_var:
            _, val_targets = val.max(dim=1) # indices for categorical
        else:
            val_targets = val  #values for real valued
        #print(val_recon)
        #print(val_targets)
        CE, KLd, val_loss = VAE.vae_loss(epoch,val_recon, val_targets, val_mu, val_logvar)
        val_loss_hist.append(val_loss)
        #print("\t CE: {:.5f}, KLd: {:.5f}, Validation loss: {:.5f}".format(CE, KLd, val_loss), end='')
        
        # early_stopping needs the validation loss to check if it has decreased, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(val_loss, VAE)
        if VAE.early_stopping =="True" and early_stopping.early_stop:
            #print("Early stopping")
            break
    #print("\nTraining marginal VAE for " + attribute+ " finished!")
    if VAE.early_stopping == "True":
        VAE.load_state_dict(torch.load('checkpoint.pt'))  # load the last checkpoint with the best model
        return VAE, float(early_stopping.val_loss_min.item())
    else:
        return VAE, float(val_loss.item())

