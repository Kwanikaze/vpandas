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
            sigma22_vectors.append(torch.exp(logvar_dict[e].detach())) #don't need torch.exp
        sigma22_diag = torch.cat(sigma22_vectors, axis=1)
        sigma22 = torch.diag_embed(sigma22_diag)

        """
        print(var_dict[e])
        print(torch.diag(var_dict[e]))
        print(sigma22)
        """

        sigma12_vectors = []
        
        if str(query_attribute + e) in self.covar_dict.keys():
            L_matrix = self.covar_dict[query_attribute + e]
        elif str(e + query_attribute) in self.covar_dict.keys():
            L_matrix = torch.transpose(self.covar_dict[e + query_attribute],0,1)
        else:
            print("Invalid Evidence") 
        LTmask = torch.tril(torch.ones(self.latent_dims,self.latent_dims))
        LTmask = Variable(LTmask)
        L = torch.mul(LTmask, L_matrix)
        #print("L")
        #print(L)
        LL = torch.matmul(L,torch.transpose(L,0,1))
        sigma12_vectors.append(LL)
        #test=torch.cholesky(LL)
        """
        for e in evidence_attributes:
            if str(query_attribute + e) in self.covar_dict.keys():
                sigma12_vectors.append(self.covar_dict[query_attribute + e])
            elif str(e + query_attribute) in self.covar_dict.keys():
                sigma12_vectors.append(torch.transpose(self.covar_dict[e + query_attribute],0,1))
        """
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
            #test = torch.cholesky(var_cond)
            #print(test)
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

    def cond_vae_loss(self,mu_dict,logvar_dict,mu_cond_dict,var_cond_dict,query_attribute):
        evidence_attributes = self.attributes.copy() #ToDO, accept evidence attributes as input to function,dropout
        evidence_attributes.remove(query_attribute)
        #query_attr is A
        
        # LHS of cond_KL, product of two Gaussians
        mu_evid_vectors = []
        for e in evidence_attributes:
            mu_evid_vectors.append(mu_dict[e].detach())
        mu_evid = torch.cat(mu_evid_vectors,axis=0).unsqueeze(2)

        mu_query = mu_dict[query_attribute].detach().unsqueeze(2)
        
        var_evid_vectors = []
        for e in evidence_attributes:
            var_evid_vectors.append(torch.exp(logvar_dict[e].detach()))
        var_evid_diag = torch.cat(var_evid_vectors, axis=1)
        var_evid = torch.diag_embed(var_evid_diag)

        var_query = torch.diag_embed(torch.exp(logvar_dict[query_attribute].detach()))
        
        mu_cond = mu_cond_dict[query_attribute]
        var_cond = var_cond_dict[query_attribute]
        """
        print("inside cond_vae")
        print(mu_evid.shape)  #256,2,1
        print(mu_query.shape)  #256,2,1
        print(var_evid.shape) #256,2,2
        print(var_query.shape) #256,2,2
        print(mu_cond.shape) #256,2,1
        print(var_cond.shape)#256,2,2
        """
        #mu_prod_term1 = torch.bmm(var_evid,torch.bmm(torch.inverse(var_query + var_evid),mu_query))
        #mu_prod_term2 = torch.bmm(var_query,torch.bmm(torch.inverse(var_query + var_evid),mu_evid))
        #mu_q = mu_prod_term1 + mu_prod_term2
        #print(mu_LHS.shape)

        #var_q = torch.bmm(var_query,torch.bmm(torch.inverse(var_query + var_evid),var_evid))
        #print(var_LHS.shape)
        mu_q,var_q = product_two_gaussians(mu_query,mu_evid,var_query,var_evid)

        #RHS of cond_KL, product of two Gaussians
        #mu_prod_term1 = torch.bmm(var_evid,torch.bmm(torch.inverse(var_cond + var_evid),mu_cond))
        #mu_prod_term2 = torch.bmm(var_cond,torch.bmm(torch.inverse(var_cond + var_evid),mu_evid))
        #mu_p = mu_prod_term1 + mu_prod_term2
        #print(mu_RHS.shape)

        #var_p = torch.bmm(var_cond,torch.bmm(torch.inverse(var_cond + var_evid),var_evid))
        #print(var_RHS.shape)

        mu_p,var_p = product_two_gaussians(mu_cond,mu_evid,var_cond,var_evid)
        
        KL1 = torch.log(torch.det(var_p)/ torch.det(var_q)) - self.latent_dims
        KL_term2 = torch.bmm(torch.inverse(var_p),var_q)
        KL2 = KL_term2.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1) #trace of a batch of matrices, size 256
        KL3 = torch.matmul(torch.transpose(mu_p-mu_q,1,2), torch.matmul(torch.inverse(var_p), (mu_p - mu_q) ))
        KL = 0.5 * torch.sum(KL1 + KL2 + KL3) #negative missing
        return KL
    

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

    #Given x, returns: reconstruction x_hat, mu, log_var
    def forward_single_attribute(self, x, attribute):
      q = self.forward(x,attribute)
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

def product_two_gaussians(mu1,mu2,var1,var2):
    mu_prod_term1 = torch.bmm(var2,torch.bmm(torch.inverse(var1 + var2),mu1))
    mu_prod_term2 = torch.bmm(var1,torch.bmm(torch.inverse(var1 + var2),mu2))
    mu_prod = mu_prod_term1 + mu_prod_term2
    var_prod = torch.bmm(var1,torch.bmm(torch.inverse(var1 + var2),var2))
    return mu_prod, var_prod