from .unet_parts import *
import torch.nn as nn
import torch.nn.functional as F


class UNET(nn.Module):
    """ Implementation of UNET model as per the paper of Ronneberg when using default values.  """
    def __init__(self, encode_chs=(3, 64, 128, 256, 512, 1024), decode_chs=(1024, 512, 256, 128, 64), num_class=1,
                 keep_dim=False, batchNorm: bool = False, dropout: float = 0., padding=0, out_size=(572, 572)):
        super().__init__()
        self.keep_dim = keep_dim
        self.output_size = out_size
        self.encode = UNETEncoder(encode_chs, batchNorm, dropout, padding)
        self.decode = UNETDecoder(decode_chs, batchNorm, dropout, padding)
        self.output = nn.Conv2d(decode_chs[-1], num_class, (1, 1))

    def forward(self, x):
        """ Forward pass function for the UNET model as required when using Pytorch.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of dimensions N x C x H x W where N is the amount of image stacks in the minibatch, C is the
            size amount of channels, H is the height of each image and W the width.

        Returns
        -------
        torch.Tensor
            Output of the UNET model with dimensions N x C x H x W where N is the amount of image stacks in the
            minibatch, C is the amount of classes to be predicted, H is the height and W is the width.
        """
        encoder_features = self.encode(x)
        out = self.decode(encoder_features[::-1][0], encoder_features[::-1][1:])
        out = self.output(out)
        if self.keep_dim:
            out = F.interpolate(out, self.out_size)
        return out
