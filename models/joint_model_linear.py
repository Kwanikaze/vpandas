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
            self.layers['fc_mu' + str(a)] = nn.Linear(self.input_dims[a], self.latent_dims)
            self.layers['fc_out' + str(a)] = nn.Linear(self.latent_dims,input_dims[a])
        self.real_vars = real_vars
        self.cat_vars = cat_vars
        self.learning_rate = args.learning_rate
        self.mms_dict = mms_dict #MinMaxScalar for real vars
        #self.num_runs = args.num_runs # Sensitive to initial parameters, take best performing model out of runs
        self.graph_samples = args.graph_samples
        self.covar_dict = nn.ParameterDict({'A'+'B': nn.Parameter(torch.rand(size=(self.latent_dims,self.latent_dims))/100,requires_grad=True) })
    
    #Query attribute's mu is 0, logvar is 1
    """
    sample_flag = True then samples using cholesky
    #sample_flag = False, then trains using additional KL loss term
    z_evidence_dict[a] shape is    batch_size,latent_dims
    """
    def conditional(self, z_evidence_dict, mu_dict, logvar_dict, evidence_attributes, query_attribute,sample_flag=True):
        evidence_tensors = []  #Unpack z_evidence_dict into single tensor 
        for a in evidence_attributes:         #z_evidence has all attributes during training, need to filter out query_attribute     
            evidence_tensors.append(z_evidence_dict[a])

        z = torch.cat(evidence_tensors,dim=1) #batch_size,evidence_vars*latent_dim
        z = z.unsqueeze(2) #batch_size,evidence_vars*latent_dim,1
        q = self.latent_dims
        N_minus_q = q*len(evidence_attributes)

        mu1 = torch.zeros(z.size(0), q ,1) #standard normal prior, shape is batch_size,q,1
        
        mu2_vectors = []
        for e in evidence_attributes:
            mu2_vectors.append(mu_dict[e].detach())
        mu2 = torch.cat(mu2_vectors,dim=0) 
        mu2 = mu2.unsqueeze(2) #mu2 shape is batch_size,N_minus_q,1

        sigma11s = torch.diag(torch.ones(q)) #standard normal prior has identity covariance, sigma11 is of shape q,q
        sigma11 = sigma11s.repeat(z.size(0),1,1)

        sigma22_vectors = []
        for e in evidence_attributes:
            sigma22_vectors.append(torch.exp(logvar_dict[e].detach()))
        sigma22_diag = torch.cat(sigma22_vectors, dim=0)
        sigma22 = torch.diag_embed(sigma22_diag) #sigma22 shape is batch_size, N_minus_q, N_minus_q
        

        sigma12_vectors = []
        """
        if str(query_attribute + e) in self.covar_dict.keys():
            L_matrix = self.covar_dict[query_attribute + e]
        elif str(e + query_attribute) in self.covar_dict.keys():
            L_matrix = torch.transpose(self.covar_dict[e + query_attribute],0,1)
        else:
            print("Invalid Evidence") 
        LTmask = torch.tril(torch.ones(self.latent_dims,self.latent_dims))
        LTmask = Variable(LTmask)
        L = torch.mul(LTmask, L_matrix)
        LL = torch.matmul(L,torch.transpose(L,0,1))
        sigma12_vectors.append(LL)

        """
        for e in evidence_attributes:
            if str(query_attribute + e) in self.covar_dict.keys():
                sigma12_vectors.append(self.covar_dict[query_attribute + e])
            elif str(e + query_attribute) in self.covar_dict.keys():
                sigma12_vectors.append(torch.transpose(self.covar_dict[e + query_attribute],0,1))
        #"""
        sigma12s = torch.cat(sigma12_vectors, axis=1)
        sigma12 = sigma12s.repeat(z.size(0),1,1)
        sigma21 = torch.transpose(sigma12,1,2)
        

        mu_cond = mu1 + torch.matmul(torch.matmul(sigma12,torch.inverse(sigma22)), (z-mu2))
        
        var_cond = sigma11 - torch.matmul(torch.matmul(sigma12,torch.inverse(sigma22)),sigma21) #latent_dims*evidence_vars,latent_dims*evidence_vars
        
        LTmask = torch.tril(torch.ones(self.latent_dims,self.latent_dims)) #Lower triangular mask matrix
        LTmask = Variable(LTmask)
        LTmask = LTmask.repeat(z.size(0),1,1) 
        L = torch.mul(LTmask, var_cond)
        var_cond = torch.bmm(L,torch.transpose(L,1,2)) + 0.1*torch.eye(self.latent_dims) #256,2,2
        
        #var_cond = var_cond + 10*torch.eye(self.latent_dims)
        #var_cond = torch.bmm(var_cond,torch.transpose(var_cond,1,2)) + 0.1*torch.eye(self.latent_dims)

        """
        print("z")
        print(z.shape)
        print("mu1")
        print(mu1.shape)
        print("mu2")
        print(mu2.shape)
        print("sigma11")
        print(sigma11.shape)
        print("sigma22")
        print(sigma22.shape)
        print("sigma21")
        print(sigma21.shape)
        print("sigma12")
        print(sigma12.shape)
        print("mu_cond")
        print(mu_cond.shape) #256,2,1
        print("var_cond")
        print(var_cond.shape)
        """

        if sample_flag == False:
            return -1, -1, mu_cond, var_cond
        else:
            z_cond = self.multivariate_reparameterize(mu_cond, var_cond)  #256,2,1 
            z_cond = z_cond.squeeze(2) #256,2
            x_query_batch_cond_recon = self.decode(z_cond, query_attribute) # batch_size,input_dims[a]
            """
            print("z_cond")
            print(z_cond.shape)
            print("x_query_batch_cond_recon")
            print(x_query_batch_cond_recon.shape)
            """
            return x_query_batch_cond_recon, z_cond,mu_cond,var_cond

    def multivariate_reparameterize(self,mu,covariance):
        L = torch.cholesky(covariance) #256,2,2
        eps = torch.randn_like(mu) #batch_size, latent_dims,1
        return mu + torch.bmm(L,eps) #256,2,1
    
    def update_args(self, args):
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.activation = args.activation
        self.anneal_factor = args.anneal_factor
        self.num_epochs = args.num_epochs
        self.variational_beta = args.variational_beta
        self.train_on_query = args.train_on_query

    #accepts OHE input of an attribute, returns mu and log variance
    def encode(self, x, attribute):
        return self.layers['fc_mu' + str(attribute)](x)

    
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
            #return self.layers['fc_out' + str(attribute)](z) #MSE
    
    def forward(self,x,attribute):
        z = self.encode(x,attribute)
        return self.decode(z,attribute),z

    def latent(self,x, attribute, add_variance=True):
        z = self.encode(x, attribute)
        if add_variance == False:
            return z

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

    #Given x, returns: reconstruction x_hat, mu, log_var
    def forward_single_attribute(self, x, attribute):
      q = self.forward(x,attribute)
      if attribute in self.real_vars:
        q = self.mms_dict[attribute].inverse_transform(q[0].detach().numpy().reshape(1,-1))
      else:
        q = np.round(q[0].detach().numpy(),decimals=2)
      return q

    def query_single_attribute(self, x_evidence_dict, query_attribute, evidence_attributes, query_repetitions=10000):
        x_evidence_dict_rep = {}
        for e in evidence_attributes:
            x_evidence_dict_rep[e] = x_evidence_dict[e].repeat(query_repetitions,1) #1, input_dims of evidence --> 10k, input_dims of evidence
        
        z_evidence_dict, mu_dict, logvar_dict = {}, {}, {}
        for e in evidence_attributes:
              z_evidence_dict[e], mu_dict[e], logvar_dict[e] = self.latent(torch.tensor(x_evidence_dict_rep[e]).float(),e, add_variance=False)#Could be add_variance=False
        query_recon, z_query, mu_cond, var_cond = self.conditional(z_evidence_dict, mu_dict, logvar_dict, evidence_attributes, query_attribute,sample_flag=True)
        # query_recon is 10k,input_dims[query_attr]
        # mu_cond is 10k,2,1
        # var_cond is 10k,2,2
        _, recon_max_idxs = query_recon.max(dim=1)
        if self.latent_dims ==2 and self.graph_samples == "True": #Cannot graph since have 10k different mu_cond, since have 10k different z_evidence
            checks.graphSamples(mu_cond[0].squeeze(1).detach().numpy(),var_cond[0].detach().numpy(),z_query, recon_max_idxs.cpu().detach().numpy(),evidence_attributes,query_attribute, query_repetitions,self.cat_vars)
        
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