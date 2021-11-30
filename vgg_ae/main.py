import os
from tqdm import tqdm
import time
import sys
import copy
import torch
import torchvision
from utils.utils import train
from utils.classes import VGGAutoEncoder, RMSELoss



base_channel_size = 10
latent_dim = 10
num_input_channels=3
#hyperparams
lr=1e-3
num_epochs=2
#saving path
saving_path = os.path.join('.', 'weights')



if __name__ == '__main__':

    #Get directory of the data
    data_dir = sys.argv[-1]
    print(data_dir)

    #Prepare the data
    data_transforms = {
    'train': torchvision.transforms.Compose([
        torchvision.transforms.Resize((64, 64)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ]),
    'val': torchvision.transforms.Compose([
        torchvision.transforms.Resize((64, 64)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ]),}

    image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x+'_images'),
                        data_transforms[x]) for x in ['train', 'val']}

    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], 
                        batch_size=64, shuffle=True, num_workers=1) for x in ['train','val']}

    #Define the model
    vgg_ae = VGGAutoEncoder(base_channel_size=base_channel_size, 
                latent_dim=latent_dim, 
                input_channel_size=num_input_channels)
    optimizer = torch.optim.Adam(vgg_ae.parameters(), lr=lr)
    criterion = torch.nn.MSELoss(reduction="none")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #Start the training
    vgg_ae, loss_values_train, loss_values_val = train(vgg_ae,
                                                dataloaders_dict,
                                                optimizer, 
                                                criterion, 
                                                device, 
                                                num_epochs)
    
    #map the model to the CPU again
    if torch.cuda.is_available():
        vgg_ae.to(torch.device("cpu"))

    #Saving the weights of the encoder and the entire model
    if not os.path.exists(saving_path):
        os.makedirs(saving_path, exist_ok=True)
    
    saving_path_encoder = os.path.join(saving_path, "encoder.pth")
    torch.save(vgg_ae.encoder.state_dict(), saving_path_encoder)
    saving_path_model = os.path.join(saving_path, "all.pth")
    torch.save(vgg_ae.state_dict(), saving_path_model)

    print("Terminated !")