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


