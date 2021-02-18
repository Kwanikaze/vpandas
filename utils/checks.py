import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
color_dict = {'0':'blue','1':'brown','2':'red','3':'orange','4':'purple','5':'black'}

def graphLatentSpace(VAE_MRF,df,df_OHE,attributes,num_samples,args):
    if args.latent_dims > 3:
        return
    else:
        x_dict = {a: df.filter(like=a,axis=1).values for a in attributes}
        x_dict_OHE = {a: Variable(torch.from_numpy(df_OHE.filter(like=a,axis=1).values)) for a in attributes}
        z_dict = {a: VAE_MRF.latent(x_dict_OHE[a].float(), attribute = a, add_variance=True) for a in attributes} 
        np_z_dict = {a: z_dict[a].cpu().detach().numpy().reshape(num_samples,args.latent_dims) for a in attributes}  #num_samples,latent_dims

        for a in attributes:
            for s in range(0,num_samples):
                val = str(x_dict[a][s]).lstrip('[').rstrip(']')
                if args.latent_dims==1:
                    plt.plot(np_z_dict[a], 'o', color='black',label="z"+str(a));
                elif args.latent_dims ==2:
                    plt.plot(np_z_dict[a][s,0],np_z_dict[a][s,1], 'o', color=color_dict[val] ,label=val);
            if args.latent_dims ==3:
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure()
                #ax = fig.gca(projection='3d')
                ax = Axes3D(fig)
                #t = np.arange(1000)
                #ax.scatter(np_z_dict[a][s,0], np_z_dict[a][s,1],np_z_dict[a][s,2], 'o', color=color_dict[val] ,label=val);
                ax.scatter(np_z_dict[a][:,0], np_z_dict[a][:,1],np_z_dict[a][:,2], 'o', color='black',label='z'+str(a));

            plt.title("Latent Encodings of Observed Data from {} Marginal Encoder ".format(a))
            if args.latent_dims ==2:
                handles, labels = plt.gca().get_legend_handles_labels()
                labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys(),loc=1)
            else:
                plt.legend(loc=1)
            plt.show()


def graphSamples(mu_cond,var_cond,z_cond,recon_max_idxs,evidence_attributes,query_attribute,query_repetitions):
    #print(z_cond)
    x, y = np.mgrid[-3:3:.01, -3:3:.01]
    pos = np.dstack((x, y))
    rv = mvn(mu_cond,var_cond)
    fig2 = plt.figure()
    plt.title("P(z{} | z{}) Multivariate Normal".format(query_attribute,str(evidence_attributes).lstrip('[').rstrip(']')))
    ax2 = fig2.add_subplot(111)
    ax2.contourf(x, y, rv.pdf(pos))
    plt.show()

    for s in range(0,query_repetitions): 
        val = str(recon_max_idxs[s])
        plt.plot(z_cond[s,0],z_cond[s,1], 'o', color=color_dict[val] ,label=val);
        #plt.plot(z_cond[s,0],z_cond[s,1], 'o', color='black' ,label=str(query_attribute));
    plt.title("P(z{} | z{}) Multivariate Normal Samples".format(query_attribute,str(evidence_attributes).lstrip('[').rstrip(']')))
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),loc=1)
    plt.show()