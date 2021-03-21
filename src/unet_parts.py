import torch
import torch.nn as nn
import torch.nn.functional as F

class Unet_block(nn.Module):
    """ Standard block as used in the UNET paper"""
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv1 = nn.conv2d(ch_in, ch_out, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.conv2d(ch_out, ch_out, 3)

    def forward(self, x):
        """

        Parameters
        ----------
        x :
            

        Returns
        -------

        """
        return self.relu(self.conv2(self.relu(self.conv1(x))))

class UNET_encoder(nn.Module):
    def __init__(self, channels = (3, 64, 128, 256. 512. 1024)):
        super().__init__()
        self.encode_UNET = nn.MolduleList([Unet_block(channels[i], channels[i+1]) for i in range(len(channels)-1)])
        self.mpool = nn.MaxPool2d(2)

    def forward(self, x):
        features = []
        for u_block in self.encode_UNET:
            x = u_block(x)
            features.append(x)
            x = self.pool(x)
        return features
