import utils.checks as checks
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


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
        self.mms_dict = mms_dict #MinMaxScalar for real vars
        self.num_runs = args.num_runs # Sensitive to initial parameters, take best performing model out of runs
        self.graph_samples = args.graph_samples
        self.covar_dict = nn.ParameterDict({'A'+'B': nn.Parameter(torch.rand(size=(self.latent_dims,self.latent_dims))/100,requires_grad=True) })
    
    #Query attribute's mu is 0, logvar is 1
    def conditional(self, z_evidence_dict, mu_dict, logvar_dict, query_attribute,sample_flag=True):
        evidence_attributes = self.attributes.copy() #ToDO, accept evidence attributes as input to function,dropout
        evidence_attributes.remove(query_attribute)
        evidence_tensors = []  #Unpack z_evidence_dict into single tensor 
        for a in evidence_attributes:         #z_evidence has all attributes during training, need to filter out query_attribute     
            evidence_tensors.append(z_evidence_dict[a])
            #print(z_evidence_dict[a].shape) #batch_size,latent_dims

        z = torch.cat(evidence_tensors,dim=1) #batch_size,evidence_vars*latent_dim
        #z = torch.transpose(z,0,1) #evidence_vars*latent_dim,batch_size
        q = self.latent_dims
        N_minus_q = q*len(evidence_attributes)

        mu1 = torch.zeros(z.size(0), q ,1) #standard normal prior, batch_size,latent_dims,1
        
        mu2_vectors = []
        for e in evidence_attributes:
            mu2_vectors.append(mu_dict[e].detach())
        mu2 = torch.cat(mu2_vectors,axis=0)


        sigma11 = torch.diag(torch.ones(q)) #standard normal prior
        
        sigma22_vectors = []
        for e in evidence_attributes:
            sigma22_vectors.append(logvar_dict[e].detach()) #don't need torch.exp
        sigma22_diag = torch.cat(sigma22_vectors, axis=1)
        sigma22 = torch.diag_embed(sigma22_diag)

        """
        print(var_dict[e])
        print(torch.diag(var_dict[e]))
        print(sigma22)
        """

        sigma12_vectors = []
        for e in evidence_attributes:
            if str(query_attribute + e) in self.covar_dict.keys():
                sigma12_vectors.append(self.covar_dict[query_attribute + e])
            elif str(e + query_attribute) in self.covar_dict.keys():
                sigma12_vectors.append(torch.transpose(self.covar_dict[e + query_attribute],0,1))
        sigma12s = torch.cat(sigma12_vectors, axis=1)
 
        sigma12 = sigma12s.repeat(z.size(0),1,1)
        
        sigma21 = torch.transpose(sigma12,1,2)
        
        #Check if covariance of joint is PSD
        """
        print(mu1.shape)
        print(sigma12.shape)
        print(sigma22.shape)
        print(z.shape)
        print(mu2.shape)
        print('xx')
        xx = torch.matmul(sigma12,torch.inverse(sigma22)) #batch_size, latent_dim, latent_dim
        print(xx.shape) # 256,2,2
        print('yy')
        yy = torch.unsqueeze(z - mu2,2)  # 256, 2, 1
        print(yy.shape)
        zz = torch.matmul(xx, yy)
        """
        mu_cond = mu1 + torch.matmul(torch.matmul(sigma12,torch.inverse(sigma22)), torch.unsqueeze((z-mu2),2))
        
        """
        print(sigma11)
        print(sigma22)
        print(sigma12)
        print(sigma21)
        """

        #var_cond must be PSD
        var_cond = sigma11 - torch.matmul(torch.matmul(sigma12,torch.inverse(sigma22)),sigma21) #latent_dims*evidence_vars,latent_dims*evidence_vars
        #var_cond_PSD = torch.bmm(var_cond,torch.transpose(var_cond,1,2)) + 0.00001*torch.eye(self.latent_dims)
        
        #LTmask = torch.tril(torch.ones(self.latent_dims,self.latent_dims)) #Lower triangular mask matrix
        #LTmask = Variable(LTmask)
        #LTmask = LTmask.repeat(z.size(0),1,1) 
        #L = torch.mul(LTmask, var_cond)
        #var_cond_PSD = torch.bmm(L,torch.transpose(L,1,2)) + 0.001*torch.eye(self.latent_dims) #256,2,2
        
        if sample_flag == False:
            return mu_cond, var_cond
        else:
            mu_cond_T = torch.transpose(mu_cond,1,2) #256,2,1 to 256,1,2
            z_cond = self.multivariate_reparameterize(mu_cond_T, var_cond) 
            x_query_batch_cond_recon = self.decode(z_cond, query_attribute)
            return x_query_batch_cond_recon

    def multivariate_reparameterize(self,mu,covariance):
        L = torch.cholesky(covariance) #256,2,2
        eps = torch.randn_like(mu) #batch_size, latent_dims
        return mu + torch.bmm(eps, L) #256,1,2
    
    def update_args(self, args):
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.activation = args.activation
        self.anneal_factor = args.anneal_factor
        self.num_epochs = args.num_epochs
        self.variational_beta = args.variational_beta

    #accepts OHE input of an attribute, returns mu and log variance
    def encode(self, x, attribute):
        if self.activation == "sigmoid":
            h1 = torch.sigmoid(self.layers['fc1' + str(attribute)](x))
        elif self.activation == "relu":
            h1 = torch.relu(self.layers['fc1' + str(attribute)](x))
        elif self.activation == "leaky_relu":
            h1 = F.leaky_relu(self.layers['fc1' + str(attribute)](x))
        return self.layers['fc_mu' + str(attribute)](h1), self.layers['fc_logvar' + str(attribute)](h1)

    #Given mu and logvar generates latent z
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std) #256,2
        return mu + eps*std
    
    #Decodes latent z into reconstruction with dimension equal to num
    def decode(self, z,attribute): #z is size [batch_size,latent_dims]
        if z.size()[0] == self.latent_dims: #resize from [latent_dims] to [1,latent_dims]
            if len(z.size()) == 1:
                z = z.view(1, self.latent_dims)
        if attribute in self.cat_vars:
            softmax = nn.Softmax(dim=1)  #normalizes reconstruction to range [0,1] and sum to 1
            return softmax(self.layers['fc_out' + str(attribute)](z)) #recon
        else:
            return torch.sigmoid(self.layers['fc_out' + str(attribute)](z)) #BCE
            #return self.fc_out(z) #MSE
    
    def forward(self,x,attribute):
        mu, logvar = self.encode(x,attribute)
        z = self.reparameterize(mu,logvar)
        return self.decode(z,attribute), mu, logvar

    def recon_loss(self, batch_recon, batch_targets, attribute):
        if attribute in self.cat_vars:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCELoss()
            #criterion = nn.MSELoss()
            batch_recon = batch_recon.double()
        return criterion(batch_recon, batch_targets)

    def vae_loss(self, epoch, batch_recon, batch_targets, mu, logvar, attribute):
        variational_beta = self.variational_beta*min(1, epoch/(self.num_epochs*self.anneal_factor)) #annealing schedule,starts beta at 0 increases to 1
        #variational_beta = self.variational_beta
        variational_beta = variational_beta*(len(self.attributes))
        CE = self.recon_loss(batch_recon, batch_targets, attribute)
        #print(CE)
        KLd = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # https://stats.stackexchange.com/questions/318748/deriving-the-kl-divergence-loss-for-vaes
        #print(KLd)
        return CE, variational_beta*KLd, CE + variational_beta*KLd

    def cond_vae_loss(self,latent_mu_dict,latent_logvar_dict,mu_cond_dict,var_cond_dict,query_attribute):
        evidence_attributes = self.attributes.copy() #ToDO, accept evidence attributes as input to function,dropout
        evidence_attributes.remove(query_attribute)
        #query_attr is A
        # LHS of cond_KL, product of two Gaussians
        #mu_prod = torch.bmm()
        print(var_cond_dict['A'].shape) #256,2,2
        return

    def latent(self,x, attribute, add_variance=True):
        mu, logvar = self.encode(x, attribute)
        if add_variance == False:
            return mu
        else:
            z = self.reparameterize(mu, logvar)
            #z = z.unsqueeze(1) #[latent_dims,1] or [num_samples,1,latent_dims]
            #print("z shape in latent")
            #print(z.shape)
            return z