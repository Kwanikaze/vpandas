import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils

#Parameters
latent_dims = 2
num_epochs = 10
batch_size = 128
capacity = 64 
learning_rate = 1e-3
variational_beta = 1
use_gpu = True
device = "cuda:0"
#device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
# 10 epochs on GPU: 20 seconds
# 10 epochs on CPU: 244 seconds

# Load MNIST # digits 0-9 in 28 x 28 grayscale images (1 channel, not 3 channel RGB)
import torchvision.transforms as transforms
from torch.utils.data import DataLoader # combines dataset and sampler
from torchvision.datasets import MNIST

img_transform = transforms.Compose([transforms.ToTensor()])
train_dataset = MNIST(root='/data/MNIST', download=True, train=True, transform=img_transform)
train_dataloader = DataLoader(train_dataset,batch_size=batch_size, shuffle = True)

test_dataset = MNIST(root='/data/MNIST', download=True, train=False, transform=img_transform)
test_dataloader = DataLoader(test_dataset,batch_size=batch_size, shuffle = True)

#VAE Model, convolutional layers instead of fully connected layers
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__() #call __init__() method of the parent class of Encoder which is nn.Module.
        c = capacity
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = c, kernel_size=4,stride=2,padding=1) #out: c x 14 x 14
        self.conv2 = nn.Conv2d(in_channels = c, out_channels = c*2, kernel_size=4,stride=2,padding=1)  #out: c x 2 x 7 x 7
        self.fc_mu = nn.Linear(in_features = c*2*7*7, out_features = latent_dims) 
        self.fc_logvar = nn.Linear(in_features = c*2*7*7, out_features = latent_dims)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # use view() to get [batch_size, num_features]
        # # -1 calculates the missing value given the other dim.
        # flatten batch of multi-channel feature maps to a batch of feature vectors
        x = x.view(x.size(0), -1) #128,6272 = 64*2*7*7 
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        c = capacity
        self.fc = nn.Linear(in_features=latent_dims,out_features = c*2*7*7)
        self.conv2 = nn.ConvTranspose2d(in_channels = c*2, out_channels = c, kernel_size=4, stride = 2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels = c, out_channels =1, kernel_size=4,stride=2,padding=1)
    
    def forward(self,x):
        x = self.fc(x)
        x = x.view(x.size(0), capacity*2, 7, 7) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv1(x))  # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        return x

class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder,self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def latent_sample(self,mu,logvar):
        if self.training:
            # Reparameterization Trick for training
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self,x):
        latent_mu, latent_logvar = self.encoder(x) # added forward
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent) # added forward
        return x_recon, latent_mu, latent_logvar

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(recon_x.view(-1,784), x.view(-1,784), reduction = 'sum')
    KLdivergence = -0.5 * torch.sum(1+logvar - mu.pow(2) - logvar.exp())
    return recon_loss + variational_beta*KLdivergence


vae = VariationalAutoencoder()
vae = vae.to(device) # gpu
num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
print("# params : %d" % num_params)

#Train VAE
optimizer = torch.optim.Adam(params = vae.parameters(),lr=learning_rate,weight_decay = 1e-5)

vae.train() #Set to training mode
train_loss_avg = []
print("Training...")
start = time.time()
for epoch in range(num_epochs):
    train_loss_avg.append(0)
    num_batches = 0

    for image_batch, _ in train_dataloader:
        image_batch = image_batch.to(device)
        #VAE recon
        image_batch_recon, latent_mu, latent_logvar = vae.forward(image_batch)
        #recon error
        loss = vae_loss(image_batch_recon,image_batch,latent_mu,latent_logvar)
        #backprop
        optimizer.zero_grad()
        loss.backward()
        #one step of the optimizer
        optimizer.step()
        train_loss_avg [-1] += loss.item() #Update current epoch training loss
        num_batches += 1
    
    train_loss_avg[-1] /= num_batches
    print("Epoch [%d / %d] average reconstruction error: %f" % (epoch+1,num_epochs, train_loss_avg[-1]))
total = time.time() - start
print("%d seconds required for training." % total)

# this is how the VAE parameters can be saved:
# torch.save(vae.state_dict(), './pretrained/my_vae.pth')

#Plot Training curve
fig = plt.figure()
plt.plot(train_loss_avg)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

#Evalulate on the MNIST Test Set
vae.eval() # set model to evaluation mode

test_loss_avg, num_batches = 0,0
for image_batch, _ in test_dataloader:
    with torch.no_grad():
        image_batch = image_batch.to(device)
        #VAE reconstruction
        image_batch_recon, latent_mu, latent_logvar = vae(image_batch)
        #reconstruction error
        loss = vae_loss(image_batch_recon, image_batch, latent_mu, latent_logvar)
        test_loss_avg+= loss.item()
        num_batches += 1

test_loss_avg /= num_batches
print("Average test reconstruction error %f" % test_loss_avg)

#Visualize Reconstructions
def show_image(img):
    img = img.clamp(0,1)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))

def visualise_output(images,model):
    with torch.no_grad():
        images = images.to(device)
        images, _, _ = model(images)
        images = images.cpu()
        images = images.clamp(0,1)
        np_imagegrid = torchvision.utils.make_grid(images[1:50],10,5).numpy()
        plt.imshow(np.transpose(np_imagegrid, (1,2,0)))
        plt.show()

images, labels = iter(test_dataloader).next()

#Show original images
print('Original images')
show_image(torchvision.utils.make_grid(images[1:50],10,5))
plt.show()

#Reconstruction images
print("VAE Reconstruction")
visualise_output(images, vae)