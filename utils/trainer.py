import utils.checks as checks
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .pytorchtools import EarlyStopping


def trainVAE_MRF(VAE_MRF,attributes,train_df_OHE, args):
    use_gpu=False
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    VAE_MRF.update_args(args)       
    optimizer = torch.optim.Adam(params = VAE_MRF.parameters(), lr = VAE_MRF.learning_rate) #single optimizer or multiple https://discuss.pytorch.org/t/two-optimizers-for-one-model/11085/12
    #print(list(VAE_MRF.parameters()))
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
        loss,CE,KLd = {},{},{}
        for a in VAE_MRF.attributes:
            loss[a]=0
            CE[a]=0
            KLd[a]=0
        train_loss = 0
        for b in range(0, N, VAE_MRF.batch_size):
            x_batch_dict, x_batch_targets_dict = {},{}
            x_batch_recon_dict, latent_mu_dict, latent_logvar_dict = {},{},{}
            train_CE_dict, train_KLd_dict, train_loss_dict, z_evidence_dict = {},{},{},{}
            for a in VAE_MRF.attributes:
                x_batch_dict[a] = x_train_dict[a][b: b+VAE_MRF.batch_size]
                if a in VAE_MRF.cat_vars:
                    _, x_batch_targets_dict[a] = x_batch_dict[a].max(dim=1) # indices for categorical
                else:
                    x_batch_targets_dict[a] = x_batch_dict[a]  #values for real valued
            
                x_batch_recon_dict[a], latent_mu_dict[a], latent_logvar_dict[a] = VAE_MRF.forward(x_batch_dict[a].float(),attribute=a) 
                #x_batch_recon_dict[a] is                       batch_size, input_dims[a]
                #latent_mu_dict[a], latent_logvar_dict[a] is    batch_size,latent_dims

                #Marginal VAE loss for each attribute: recon_loss+KLd
                train_CE_dict[a], train_KLd_dict[a], train_loss_dict[a] = VAE_MRF.vae_loss(epoch, 
                            x_batch_recon_dict[a], x_batch_targets_dict[a], latent_mu_dict[a], latent_logvar_dict[a], attribute=a) 
                loss[a] += train_loss_dict[a].item() / VAE_MRF.batch_size # update epoch loss
                CE[a] += train_CE_dict[a].item() / VAE_MRF.batch_size
                KLd[a] += train_KLd_dict[a].item() / VAE_MRF.batch_size
                
            #z_evidence_dict[a] = VAE_MRF.latent(x_batch_dict[a].float(), a, add_variance=False) #same as mu!
            z_evidence_dict = latent_mu_dict #batch_size, latent_dims
            
            #Reconstruct x_N = a, given only x_1 to x_N-1
            for query_attribute in VAE_MRF.attributes:
                #x_batch_cond_recon = VAE_MRF.conditional(z_evidence_dict,latent_mu_dict,latent_logvar_dict,query_attribute)
                x_batch_cond_recon = VAE_MRF.conditional(z_evidence_dict, mu_dict, var_dict,query_attribute)
                #print(VAE_MRF.recon_loss(x_batch_cond_recon, x_batch_targets_dict[query_attribute], query_attribute))
                train_loss_dict[str(query_attribute) + "cond"] 
                    = VAE_MRF.recon_loss(x_batch_cond_recon, x_batch_targets_dict[query_attribute], query_attribute) / VAE_MRF.batch_size
            #aa = list(VAE_MRF.parameters())[-1].clone()
            with torch.autograd.set_detect_anomaly(True):
                optimizer.zero_grad()
                for k in train_loss_dict.keys():
                    #print(k)
                    #train_loss_dict[k].backward(retain_graph=True)
                    train_loss += train_loss_dict[k]
                train_loss.backward()
                    #print(train_loss_dict[k])
                optimizer.step()        #update parameters based on gradient
            #print("CovarianceAB")
            #print(VAE_MRF.covar_dict.items())
            #bb = list(VAE_MRF.parameters())[-1].clone()
            #print(torch.equal(aa.data, bb.data))
        
        for a in VAE_MRF.attributes:
            print("Attribute: %s, Epoch %d/%d\t CE: %.5f, KLd: %.5f, Train loss=%.5f" 
                % (a, epoch + 1, VAE_MRF.num_epochs, CE[a], KLd[a], loss[a]), end='\n', flush=True)
            print("%s_cond_recon loss: %.5f" % (a,train_loss_dict[str(a) + "cond"]), end='\n', flush=True)
        print("Total Train Loss: %.5f" % (train_loss), end='\n', flush=True)
        #print(VAE_MRF.covar_dict.items())

    #print(list(VAE_MRF.parameters()))
    #VAE_MRF.joint_training(args)
    print("\nTraining MRF finished!")
