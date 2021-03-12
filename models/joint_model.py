from models import marginalVAE
import utils.checks as checks
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#from scipy.stats import multivariate_normal as mvn
from .pytorchtools import EarlyStopping

def trainVAE_MRF(VAE_MRF,attributes,train_df_OHE, args):
    use_gpu=False
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    VAE_MRF.update_args(args)       
    optimizer = torch.optim.Adam(params = VAE_MRF.parameters(), lr = VAE_MRF.learning_rate) #single optimizer or multiple https://discuss.pytorch.org/t/two-optimizers-for-one-model/11085/12
    print(list(VAE_MRF.parameters()))
    x_train_dict = {a: Variable(torch.from_numpy(VAE_MRF.train_df_OHE.filter(like=a,axis=1).values)).to(device) for a in attributes}
    
    #take population mean and variance for each attribute
    mu_dict,logvar_dict,var_dict = {}, {}, {}
    for a in attributes:
        mu_dict[a],logvar_dict[a] = VAE_MRF.encode(x_train_dict[a].float(),attribute=a) #train samples,latent_dims
        mu_dict[a] = torch.mean(mu_dict[a],0) #latent_dims
        logvar_dict[a] = torch.mean(logvar_dict[a],0) #take mean of logvar correct?
        mu_dict[a] = mu_dict[a].unsqueeze(1) #latent_dims, 1
        logvar_dict[a] = logvar_dict[a] #latent_dims
        var_dict[a] = torch.exp(logvar_dict[a])
    """
    print(mu_dict['A'].shape)
    print("muA avg")
    print(mu_dict['A'])
    print("muB avg")
    print(mu_dict['B'])
    print("logvarA avg")
    print(logvar_dict['A'])
    print("varA avg")
    print(torch.exp(logvar_dict['A']))
    print("std_devA avg")
    print(torch.exp(0.5*logvar_dict['A']))

    print("logvarB avg")
    print(logvar_dict['B'])
    print("varB avg")
    print(torch.exp(logvar_dict['B']))
    print("std_devB avg")
    print(torch.exp(0.5*logvar_dict['B']))
    """

    """
    z_dict = {a: VAE_MRF.latent(x_train_dict[a].float(), attribute=a, add_variance=True)  for a in VAE_MRF.attributes} #num_samples,latent_dims
    np_z_dict = {a: z_dict[a].cpu().detach().numpy().reshape(VAE_MRF.num_samples,VAE_MRF.latent_dims) for a in VAE_MRF.attributes}  #num_samples,latent_dims
    z_obs = np.concatenate(tuple(np_z_dict.values()),axis=1) #(num_samples,num_attrs*latent_dims)
    mu_emp = np.mean(z_obs,axis=0) #mean of each column
    covar_emp = np.cov(z_obs,rowvar=False)
    print("muAB_vector")
    print(mu_emp)
    print("covaranceAB_matrix")
    print(covar_emp)
    """


    x_val_dict = {a: Variable(torch.from_numpy(VAE_MRF.val_df_OHE.filter(like=a, axis=1).values)).to(device) for a in attributes}
    
    val_loss = 0

    early_stopping = EarlyStopping(patience=args.patience, verbose=False)
    N = VAE_MRF.num_samples
    for epoch in range(VAE_MRF.num_epochs):
        VAE_MRF.train() #set model mode to train
        loss,CE,KLd = {},{},{}
        for a in VAE_MRF.attributes:
            loss[a]=0
            CE[a]=0
            KLd[a]=0
        train_loss = 0
        for b in range(0, N, VAE_MRF.batch_size):
            x_batch_dict, x_batch_recon_dict, latent_mu_dict, latent_logvar_dict, x_batch_targets_dict,train_CE_dict,train_KLd_dict,train_loss_dict,z_evidence_dict = {},{},{},{},{},{},{},{},{}
            for a in VAE_MRF.attributes:
                x_batch_dict[a] = x_train_dict[a][b: b+VAE_MRF.batch_size]
                if a in VAE_MRF.cat_vars:
                    _, x_batch_targets_dict[a] = x_batch_dict[a].max(dim=1) # indices for categorical
                else:
                    x_batch_targets_dict[a] = x_batch_dict[a]  #values for real valued
            
                x_batch_recon_dict[a],latent_mu_dict[a],latent_logvar_dict[a] = VAE_MRF.forward(x_batch_dict[a].float(),attribute=a)
                #Marginal VAE loss for each attribute: recon_loss+KLd
                train_CE_dict[a], train_KLd_dict[a], train_loss_dict[a] = VAE_MRF.vae_loss(epoch, x_batch_recon_dict[a], x_batch_targets_dict[a], latent_mu_dict[a], latent_logvar_dict[a], attribute=a)
                
                loss[a] += train_loss_dict[a].item() / VAE_MRF.batch_size # update epoch loss
                CE[a] += train_CE_dict[a].item() / VAE_MRF.batch_size
                KLd[a] += train_KLd_dict[a].item() / VAE_MRF.batch_size
            #z_evidence_dict[a] = VAE_MRF.latent(x_batch_dict[a].float(), a, add_variance=False) #same as mu!
            z_evidence_dict = latent_mu_dict #batch_size, latent_dims

            #Reconstruct x_N = a, given only x_1 to x_N-1
            for query_attribute in VAE_MRF.attributes:
                #x_batch_cond_recon = VAE_MRF.conditional(z_evidence_dict,latent_mu_dict,latent_logvar_dict,query_attribute)
                x_batch_cond_recon = VAE_MRF.conditional(z_evidence_dict,mu_dict,var_dict,query_attribute)
                #print(VAE_MRF.recon_loss(x_batch_cond_recon, x_batch_targets_dict[query_attribute], query_attribute))
                train_loss_dict[str(query_attribute) + "cond"] = VAE_MRF.recon_loss(x_batch_cond_recon, x_batch_targets_dict[query_attribute], query_attribute) #/ VAE_MRF.batch_size
            """
            optimizer.zero_grad()
            for k in attributes:
                train_loss_dict[k].backward()
            optimizer.step()
            optimizer.zero_grad()
            for k in ["Acond","Bcond"]:
                print(k)
                train_loss_dict[k].backward()
            optimizer.step()
            #aa = list(VAE_MRF.parameters())[-1].clone()
            """
            with torch.autograd.set_detect_anomaly(True):
                for k in train_loss_dict.keys():
                    print(k)
                    train_loss_dict[k].backward(retain_graph=True)
                    train_loss += train_loss_dict[k]
                    #print(train_loss_dict[k])
                #train_loss.backward(retain_graph=True)  #Backprop the error, compute the gradient
                optimizer.step()        #update parameters based on gradient

            #bb = list(VAE_MRF.parameters())[-1].clone()
            #print(bb)
            #print(torch.equal(aa.data, bb.data))
        
        #for a in VAE_MRF.attributes:
        #    print("Attribute: %s, Epoch %d/%d\t CE: %.5f, KLd: %.5f, Train loss=%.5f" % (a, epoch + 1, VAE_MRF.num_epochs, CE[a], KLd[a], loss[a]), end='\n', flush=True)
        #    print("%s_cond_recon loss: %.5f" % (a,train_loss_dict[str(a) + "cond"]), end='\n', flush=True)
        print("Total Train Loss: %.5f" % (train_loss), end='\n', flush=True)
        #print(VAE_MRF.covar_dict.items())

    print(list(VAE_MRF.parameters()))
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
        self.mms_dict = mms_dict #MinMaxScalar for real vars
        self.num_runs = args.num_runs # Sensitive to initial parameters, take best performing model out of runs
        self.graph_samples = args.graph_samples
        #self.covar_dict = {}
        #self.covar_dict['A'+'B'] = torch.nn.Parameter(torch.zeros(size=(self.latent_dims,self.latent_dims))+0.1,requires_grad=True)
        self.covar_dict = nn.ParameterDict({'A'+'B': nn.Parameter(torch.rand(size=(self.latent_dims,self.latent_dims))/100,requires_grad=True) }) 
        #self.covar_dict['A'+'B'] = torch.nn.Parameter(torch.zeros(size=(10,10))+0.1,requires_grad=True)
    
    #Query attribute's mu is 0, logvar is 1
    def conditional(self, z_evidence_dict, mu_dict, var_dict, query_attribute):
        evidence_attributes = self.attributes.copy()
        evidence_attributes.remove(query_attribute)
        evidence_tensors = []  #Unpack z_evidence_dict into single tensor 
        for a in evidence_attributes:         #z_evidence has all attributes during training, need to filter out query_attribute
        #for a in z_evidence_dict.keys():      
            evidence_tensors.append(z_evidence_dict[a])
            #print(z_evidence_dict[a].shape) #batch_size,latent_dims

        z = torch.cat(evidence_tensors,dim=1) #batch_size,evidence_vars*latent_dim
        z = torch.transpose(z,0,1) #evidence_vars*latent_dim,batch_size
        q = self.latent_dims
        N_minus_q = q*len(evidence_attributes)

        mu1 = torch.zeros(q,1) #standard normal prior
        
        mu2 = torch.empty(N_minus_q,1)
        i=0
        for e in evidence_attributes:
            mu2[i:i+q, 0:1] = mu_dict[e]
            i += q
        """
            print(mu_dict[e])
        print(mu2)
        """
        sigma11 = torch.diag(torch.ones(q)) #standard normal prior
        sigma22 = torch.empty(N_minus_q,N_minus_q)

        i=0
        for e in evidence_attributes:
            sigma22[i:i+q, i:i+q] = torch.diag(var_dict[e])
            i += q
        """
            print(var_dict[e])
            print(torch.diag(var_dict[e]))
        print(sigma22)
        """

        sigma12 = torch.empty(q, N_minus_q)
        i=0
        for e in evidence_attributes:
            if str(query_attribute + e) in self.covar_dict.keys():
                sigma12[0:q, i:i+q] = self.covar_dict[query_attribute + e] #'A'+'B'
            else: #'B'+'A'
                sigma12[0:q, i:i+q] = torch.transpose(self.covar_dict[e + query_attribute],0,1) # 'A'+'B'
            i += q
        
        sigma21 = torch.transpose(sigma12,0,1)
        mu_cond = mu1 + torch.matmul(torch.matmul(sigma12,torch.inverse(sigma22)), (z-mu2))
        """
        print(sigma11)
        print(sigma22)
        print(sigma12)
        print(sigma21)
        """
        var_cond = sigma11 - torch.matmul(torch.matmul(sigma12,torch.inverse(sigma22)),sigma21) #latent_dims*evidence_vars,latent_dims*evidence_vars
        #inplace operation 2x2
        mu_cond_T = torch.transpose(mu_cond,0,1)

        """
        print(mu_cond_T.shape)
        print(mu_cond_T[0])
        print(mu_cond_T[1])
        print(var_cond)
        """

        z_cond = self.multivariate_reparameterize(mu_cond_T, var_cond) 
        #print(z_cond.shape)
        #z_cond = torch.zeros((mu_cond_T.shape[0], self.latent_dims))
        #for b in range(mu_cond_T.shape[0]): #VECTORIZE
        #    """
        #    print(mu_cond_T[b])
        #    print(var_cond)
        #    print(self.multivariate_reparameterize(mu_cond_T[b], var_cond))
        #    """
        #    z_cond[b,0:self.latent_dims] = self.multivariate_reparameterize(mu_cond_T[b], var_cond) 

        x_query_batch_cond_recon = self.decode(z_cond, query_attribute)
        return x_query_batch_cond_recon

    def multivariate_reparameterize(self,mu,covariance):
        #cholesky factor L, covariance = L L^T
        L = torch.cholesky(covariance)
        #L = L.repeat(1,mu.shape[0]/2)
        eps = torch.randn_like(mu) #batch_size, latent_dims
        return mu + torch.matmul(eps, L) #256x2 by 2x2
    
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
        eps = torch.randn_like(std)
        return mu + eps*std
    
    #Decodes latent z into reconstruction with dimension equal to num
    def decode(self, z,attribute): #z is size [batch_size,latent_dims]
        if z.size()[0] == self.latent_dims: #resize from [3] to [1,3]
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

    #def joint_training(self, args):
        # Feed in A to reconstruct A and predict B
        # Backprop on A loss and B loss and KL divergence with Beta scaled by number of reconstructions