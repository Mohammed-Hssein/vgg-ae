import torch
import torchvision



class VGGEncoder(torch.nn.Module):
    """
    """
    def __init__(self, 
                input_channel_size : int,
                base_channel_size : int,
                latent_dim : int,
                activation_function : object=torch.nn.GELU):
        super().__init__()
        c_hid = base_channel_size
        self.net = torch.nn.Sequential(
            #block 1
            torch.nn.Conv2d(input_channel_size, c_hid, kernel_size=3, padding=1, stride=1),
            activation_function(), #64*64
            torch.nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1, stride=1),
            activation_function(), #64*64
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2), #32*32
            
            #block 2
            torch.nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=1),
            activation_function(), #32*32
            torch.nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=1),
            activation_function(), #32*32
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2), #16*16
            
            #block 3
            torch.nn.Conv2d(2*c_hid, 3*c_hid, kernel_size=3, padding=1, stride=1),
            activation_function(), #16*16
            torch.nn.Conv2d(3*c_hid, 3*c_hid, kernel_size=3, padding=1, stride=1),
            activation_function(), #16*16
            torch.nn.Conv2d(3*c_hid, 3*c_hid, kernel_size=3, padding=1, stride=1),
            activation_function(), #16*16
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2), #8*8
            
            #block 4
            torch.nn.Conv2d(3*c_hid, 4*c_hid, kernel_size=3, padding=1, stride=1),
            activation_function(), #8*8
            torch.nn.Conv2d(4*c_hid, 4*c_hid, kernel_size=3, padding=1, stride=1),
            activation_function(), #8*8
            torch.nn.Conv2d(4*c_hid, 4*c_hid, kernel_size=3, padding=1, stride=1),
            activation_function(), #8*8
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2), #4*4
            
            #block 4
            torch.nn.Conv2d(4*c_hid, 5*c_hid, kernel_size=3, padding=1, stride=1),
            activation_function(), #4*4
            torch.nn.Conv2d(5*c_hid, 5*c_hid, kernel_size=3, padding=1, stride=1),
            activation_function(), #4*4
            torch.nn.Conv2d(5*c_hid, 5*c_hid, kernel_size=3, padding=1, stride=1),
            activation_function(), #4*4
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2), #2*2
            
            
            #linear block
            torch.nn.Flatten(),
            torch.nn.Linear(2*2*5*c_hid, latent_dim)
        )
        return
    
    def forward(self, x):
        """
        """
        return self.net(x)



class VGGDecoder(torch.nn.Module):
    """
    """
    def __init__(self,
                 input_channel_size: int,
                 base_channel_size : int,
                 latent_dim : int, 
                 activation_function : object = torch.nn.GELU):
        super().__init__()
        c_hid = base_channel_size
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 2*2*5*c_hid),
            activation_function(),
        )
        self.net = torch.nn.Sequential(
            #block1
            torch.nn.ConvTranspose2d(5*c_hid, 4*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            activation_function(),
            
            #block 2
            torch.nn.ConvTranspose2d(4*c_hid, 3*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            activation_function(),
            
            #block 3
            torch.nn.ConvTranspose2d(3*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            activation_function(),
            
            #block 2
            torch.nn.ConvTranspose2d(2*c_hid, 1*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            activation_function(),
            
            #block 1
            torch.nn.ConvTranspose2d(c_hid, input_channel_size, kernel_size=3, output_padding=1, padding=1, stride=2),
            torch.nn.Tanh(),
        )
        return
    
    def forward(self, x):
        """
        """
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 2, 2)
        x = self.net(x)
        return x


class VGGAutoEncoder(torch.nn.Module):
    """
    """
    def __init__(self, 
                 base_channel_size : int,
                 latent_dim : int,
                 encoder_class : object = VGGEncoder,
                 decoder_class : object = VGGDecoder,
                 input_channel_size : int=3):
        
        super().__init__()
        self.encoder = encoder_class(input_channel_size, base_channel_size, latent_dim)
        self.decoder = decoder_class(input_channel_size, base_channel_size, latent_dim)
        return
    
    def forward(self, x):
        """
        """
        return self.decoder(self.encoder(x))


class RMSELoss(torch.nn.Module):
    """
    """
    def __init__(self, eps=1e-10):
        super().__init__()
        self.mse = torch.nn.MSELoss(reduction="none")
        self.eps = eps
        return
    
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y) + self.eps)