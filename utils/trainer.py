import utils.checks as checks
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .pytorchtools import EarlyStopping


def trainVAE_MRF(VAE_MRF,attributes, args):
    use_gpu=False
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    VAE_MRF.update_args(args)       
    optimizer = torch.optim.Adam(params = VAE_MRF.parameters(), lr = VAE_MRF.learning_rate) #single optimizer or multiple https://discuss.pytorch.org/t/two-optimizers-for-one-model/11085/12
    print(list(VAE_MRF.parameters()))
    x_train_dict = {a: Variable(torch.from_numpy(VAE_MRF.train_df_OHE.filter(like=a,axis=1).values)).to(device) for a in attributes}
    x_val_dict = {a: Variable(torch.from_numpy(VAE_MRF.val_df_OHE.filter(like=a, axis=1).values)).to(device) for a in attributes}

    #mu_dict,mu_dict_batch,logvar_dict,var_dict,logvar_dict_mean = {}, {}, {}, {}, {}
    #for a in attributes:
        #mu_dict_batch[a],logvar_dict[a] = VAE_MRF.encode(x_train_dict[a].float(),attribute=a) #train samples,latent_dims
        #mu_dict[a] = torch.mean(mu_dict_batch[a],0) #latent_dims
        #mu_dict[a] = mu_dict[a].unsqueeze(1) #latent_dims, 1
        #logvar_dict_mean[a] = torch.mean(logvar_dict[a],0) #take mean of logvar correct?
        #var_dict[a] = torch.exp(logvar_dict_mean[a])

    val_loss = 0

    early_stopping = EarlyStopping(patience=args.patience, verbose=False)
    N = VAE_MRF.num_samples
    for epoch in range(VAE_MRF.num_epochs):
        VAE_MRF.train() #set model mode to train
        loss,CE,KLd,KLd_cond = {},{},{},{}
        for a in VAE_MRF.attributes:
            loss[a]=0
            CE[a]=0
            KLd[a]=0
            KLd_cond[a]=0
        train_loss = 0
        for b in range(0, N, VAE_MRF.batch_size):  
            train_loss_batch = 0
            x_batch_dict, x_batch_targets_dict = {},{}
            x_batch_recon_dict, latent_mu_dict, latent_logvar_dict = {},{},{}
            train_CE_dict, train_KLd_dict, train_loss_dict, z_evidence_dict = {},{},{},{}
            mu_cond_dict,var_cond_dict = {},{}
            for a in VAE_MRF.attributes:
                x_batch_dict[a] = x_train_dict[a][b: b+VAE_MRF.batch_size]
                if a in VAE_MRF.cat_vars:
                    _, x_batch_targets_dict[a] = x_batch_dict[a].clone().max(dim=1) # indices for categorical
                else:
                    x_batch_targets_dict[a] = x_batch_dict[a].clone()  #values for real valued
            
                x_batch_recon_dict[a], latent_mu_dict[a], latent_logvar_dict[a] = VAE_MRF.forward(x_batch_dict[a].float(),attribute=a) 
                #x_batch_recon_dict[a] is                       batch_size, input_dims[a]
                #latent_mu_dict[a], latent_logvar_dict[a] is    batch_size,latent_dims

                #Marginal VAE loss for each attribute: recon_loss+KLd
                train_CE_dict[a], train_KLd_dict[a], train_loss_dict[a] = VAE_MRF.vae_loss(epoch, 
                            x_batch_recon_dict[a], x_batch_targets_dict[a], latent_mu_dict[a], latent_logvar_dict[a], attribute=a) 
                loss[a] += train_loss_dict[a].item() / VAE_MRF.batch_size # update epoch loss
                CE[a] += train_CE_dict[a].item() / VAE_MRF.batch_size
                KLd[a] += train_KLd_dict[a].item() / VAE_MRF.batch_size
            
                z_evidence_dict[a] = VAE_MRF.latent(x_batch_dict[a].float(), a, add_variance=True)  #batch_size, latent_dims
            
            for a in VAE_MRF.attributes:
                mu_cond_dict[a],var_cond_dict[a] = \
                    VAE_MRF.conditional(z_evidence_dict,latent_mu_dict,latent_logvar_dict,query_attribute=a, sample_flag=False)

            for query_attribute in VAE_MRF.attributes:
                train_loss_dict[str(query_attribute) + "cond_KL"] = \
                    VAE_MRF.cond_vae_loss(latent_mu_dict,latent_logvar_dict,mu_cond_dict,var_cond_dict,query_attribute)
                KLd_cond[query_attribute] +=  train_loss_dict[str(query_attribute) + "cond_KL"].item() / VAE_MRF.batch_size
            
            #Reconstruct x_N = a, given only x_1 to x_N-1,Sigma_cond not PD matrix, cholesky fails
            """
            for query_attribute in VAE_MRF.attributes:
                x_batch_cond_recon = VAE_MRF.conditional(z_evidence_dict,latent_mu_dict,latent_logvar_dict,query_attribute)
                x_batch_cond_recon = x_batch_cond_recon.squeeze(1) #256,1,3 to 256,3
                train_loss_dict[str(query_attribute) + "cond"] = \
                    VAE_MRF.recon_loss(x_batch_cond_recon, x_batch_targets_dict[query_attribute], query_attribute) / VAE_MRF.batch_size
            """
            
            #aa = list(VAE_MRF.parameters())[-1].clone()
            with torch.autograd.set_detect_anomaly(True):
                #optimizer.zero_grad()
                for k in train_loss_dict.keys():
                    #print(k)
                    #train_loss += train_loss_dict[k]
                    optimizer.zero_grad()
                    train_loss_dict[k].backward(retain_graph=True) #https://stackoverflow.com/questions/53994625/how-can-i-process-multi-loss-in-pytorch/53995165
                    train_loss += train_loss_dict[k] / VAE_MRF.batch_size
                optimizer.step()        #update parameters based on gradient
                #train_loss.backward(retain_graph=True)
                #optimizer.step() 
                    #print(train_loss_dict[k])

            #bb = list(VAE_MRF.parameters())[-1].clone()
            #print(torch.equal(aa.data, bb.data))
            #print("CovarianceAB")
            #print(VAE_MRF.covar_dict.items())
        print("")
        print("Epoch %d/%d" % (epoch+1,VAE_MRF.num_epochs))
        for a in VAE_MRF.attributes:
            print("Marginal VAE %s: \t CE: %.5f, KLd: %.5f, Train loss=%.5f" % (a,CE[a], KLd[a], loss[a]), end='\n', flush=True)
            #print("Attribute: %s, Epoch %d/%d\t CE: %.5f, KLd: %.5f, Train loss=%.5f" 
            #    % (a, epoch + 1, VAE_MRF.num_epochs, CE[a], KLd[a], loss[a]), end='\n', flush=True)
            #print("%s_cond_recon loss: %.5f" % (a,train_loss_dict[str(a) + "cond_KL"]), end='\n', flush=True)
            print("%s_cond_KL: %.5f" % (a, KLd_cond[str(a)]), end='\n', flush=True)
        print("Total Train Loss: %.5f" % (train_loss), end='\n', flush=True)

        #Test with all validation data
        #VAE_MRF.eval()
    #print(list(VAE_MRF.parameters()))
    print("\nTraining MRF finished!")
