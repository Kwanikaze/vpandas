import utils.checks as checks
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .pytorchtools import EarlyStopping
import random

def trainVAE_MRF(VAE_MRF, args: dict):
    use_gpu=False
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    VAE_MRF.update_args(args)       
    optimizer = torch.optim.Adam(params = VAE_MRF.parameters(), lr = VAE_MRF.learning_rate) #single optimizer or multiple https://discuss.pytorch.org/t/two-optimizers-for-one-model/11085/12
    
    for name, param in VAE_MRF.named_parameters():
        if param.requires_grad:
            print (name, param.data)
    #print(list(VAE_MRF.parameters()))
    x_train_dict = {a: Variable(torch.from_numpy(VAE_MRF.train_df_OHE.filter(like=a,axis=1).values)).to(device) for a in VAE_MRF.attributes}
    
    x_val_dict = {a: Variable(torch.from_numpy(VAE_MRF.val_df_OHE.filter(like=a, axis=1).values)).to(device) for a in VAE_MRF.attributes}

    val_loss = 0

    early_stopping = EarlyStopping(patience=args.patience, verbose=False)
    N = VAE_MRF.num_samples
    #print("CovarianceAB")
    #print(VAE_MRF.covar_dict.items())
    for epoch in range(VAE_MRF.num_epochs):
        VAE_MRF.train() #set model mode to train
        loss,CE,KLd,KLd_cond,CE_evidence = {},{},{},{},{}
        for a in VAE_MRF.attributes:
            loss[a]=0
            CE[a]=0
            KLd[a]=0
            KLd_cond[a]=0
            CE_evidence[a]=0
        train_loss = 0
        rand_attrs = random.sample(VAE_MRF.attributes, len(VAE_MRF.attributes))
        #rand_attrs = random.sample(VAE_MRF.attributes, 1)
        #print(rand_attrs)
        for b in range(0, N, VAE_MRF.batch_size):  
            train_loss_batch = 0
            x_batch_dict, x_batch_targets_dict = {},{}
            x_batch_recon_dict, z_evidence_dict, latent_mu_dict, latent_logvar_dict = {},{},{},{}
            train_CE_dict, train_KLd_dict, train_loss_dict = {},{},{}
            x_batch_cond_recon, mu_cond_dict,var_cond_dict = {},{},{}
            for a in VAE_MRF.attributes:
                x_batch_dict[a] = x_train_dict[a][b: b+VAE_MRF.batch_size] #shape is batchsize, input_dims[a]
                if a in VAE_MRF.cat_vars:
                    _, x_batch_targets_dict[a] = x_batch_dict[a].clone().max(dim=1) # indices for categorical
                else:
                    x_batch_targets_dict[a] = x_batch_dict[a].clone()  #values for real valued
            
                x_batch_recon_dict[a],z_evidence_dict[a], latent_mu_dict[a], latent_logvar_dict[a] = VAE_MRF.forward(x_batch_dict[a].float(),attribute=a) 
                #x_batch_recon_dict[a] is   batch_size, input_dims[a]
                #z_evidence_dict[a], latent_mu_dict[a], latent_logvar_dict[a] is    batch_size,latent_dims
            VAE_MRF.emp_covariance(z_evidence_dict)
            for a in rand_attrs:#VAE_MRF.attributes:
                
                #Marginal VAE loss for each attribute: recon_loss+KLd
                train_CE_dict[a], train_KLd_dict[a], train_loss_dict[a] = VAE_MRF.vae_loss(epoch, 
                            x_batch_recon_dict[a], x_batch_targets_dict[a], latent_mu_dict[a], latent_logvar_dict[a], attribute=a) 
                loss[a] += train_loss_dict[a].item() / VAE_MRF.batch_size # update epoch loss
                CE[a] += train_CE_dict[a].item() / VAE_MRF.batch_size
                KLd[a] += train_KLd_dict[a].item() / VAE_MRF.batch_size
                
                #Cross-directional VAE loss
                evidence_attributes = VAE_MRF.attributes.copy() #ToDO, accept evidence attributes as input to function,dropout
                evidence_attributes.remove(a)
                if VAE_MRF.train_on_query == "True": 
                    sample_flag = False 
                else:
                    sample_flag = True #Reconstruct x_N = a, given only x_1 to x_N-1, Sigma_cond not PD matrix, cholesky fails
                x_batch_cond_recon[a], _, mu_cond_dict[a], var_cond_dict[a] = \
                    VAE_MRF.conditional(z_evidence_dict,latent_mu_dict,latent_logvar_dict, evidence_attributes, a, sample_flag) 

            for a in rand_attrs:#VAE_MRF.attributes:
                if VAE_MRF.train_on_query == "True": #Calculate new KL loss term
                    train_loss_dict[str(a) + "cond_KL"] = \
                        VAE_MRF.cond_vae_loss(latent_mu_dict,latent_logvar_dict,mu_cond_dict,var_cond_dict,a)
                    KLd_cond[a] +=  train_loss_dict[str(a) + "cond_KL"].item() / VAE_MRF.batch_size
                else:      
                    train_loss_dict[str(a) + "cond_recon"] = \
                        VAE_MRF.recon_loss(x_batch_cond_recon[a], x_batch_targets_dict[a], a)
                    CE_evidence[a] += train_loss_dict[str(a) + "cond_recon"].item() / VAE_MRF.batch_size
                #"""
            #aa = list(VAE_MRF.parameters())[-1].clone()
            with torch.autograd.set_detect_anomaly(True):
                """
                for k in train_loss_dict.keys():
                    #print(k)
                    optimizer.zero_grad()
                    train_loss_dict[k].backward(retain_graph=True) #https://stackoverflow.com/questions/53994625/how-can-i-process-multi-loss-in-pytorch/53995165
                    train_loss += train_loss_dict[k] / VAE_MRF.batch_size
                optimizer.step()        #update parameters based on gradient
                """
                optimizer.zero_grad()
                #print(train_loss_dict.keys())
                for k in train_loss_dict.keys():
                    train_loss_batch += train_loss_dict[k]
                train_loss_batch.backward(retain_graph=True)
                train_loss += train_loss_batch / VAE_MRF.batch_size
                optimizer.step() 
                #"""
            #bb = list(VAE_MRF.parameters())[-1].clone()
            #print(torch.equal(aa.data, bb.data))
            #print(list(VAE_MRF.parameters())[-1].grad)
            #print("CovarianceAB")
            #print(VAE_MRF.covar_dict.items())
        
        if epoch % 10 == 0:
            print("")
            print("Epoch %d/%d" % (epoch,VAE_MRF.num_epochs))
            for a in VAE_MRF.attributes:
                print("Marginal VAE %s: \t CE: %.5f, KLd: %.5f, Train loss=%.5f" % (a,CE[a], KLd[a], loss[a]), end='\n', flush=True)
                #print("Attribute: %s, Epoch %d/%d\t CE: %.5f, KLd: %.5f, Train loss=%.5f" 
                #    % (a, epoch + 1, VAE_MRF.num_epochs, CE[a], KLd[a], loss[a]), end='\n', flush=True)
                #print("%s_cond_recon loss: %.5f" % (a,train_loss_dict[str(a) + "cond_KL"]), end='\n', flush=True)
                if VAE_MRF.train_on_query == "True": 
                    print("%s_cond_KL: %.5f" % (a, KLd_cond[str(a)]), end='\n', flush=True)
                else:
                    print("%s_cond_recon: %.5f" % (a, CE_evidence[str(a)]), end='\n', flush=True)
            print("Total Train Loss: %.5f" % (train_loss), end='\n', flush=True)

        #Test with all validation data
        VAE_MRF.eval()
    #print(list(VAE_MRF.parameters()))
    print("\nTraining MRF finished!")
    #print("CovarianceAB")
    #print(VAE_MRF.covar_dict.items())
