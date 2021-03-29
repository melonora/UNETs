from mics.segment.models.unet_parts import *
import torch.nn as nn
import torch.nn.functional as F


class UNET(nn.Module):
    """ Implementation of UNET model as per the paper of Ronneberg when using default values. In case of skip being True
    the feature maps are passed through a convolutional layer before being concatenated with feature maps from the
    decoder. https://github.com/upashu1/Pytorch-UNet-2 claims this works better with noisy microarray data.

    Attributes
    ----------
    keep_dim: bool
        True or false indicating whether the output of the model should have the same dimensions as the input.
    output_size: Tuple[int, int]
        Tuple indicating what the output height and width should be.
    encode: UNETEncoder
        The encoder part of the UNET model.
    decode: UNETDecoder
        The decoder part of the UNET model.
    output: nn.Conv2d
        Final convolutional layer creating the output of the UNET model.
    skip: bool
        True or false indicating whether skip connections will be passed through a UNETBlock.
    skipConnects: SkipConnectsConvs
        Skip connections being passed through two convolutional layers.
     """

    def __init__(self, encode_chs: Tuple[int, ...] = (3, 64, 128, 256, 512, 1024),
                 decode_chs: Tuple[int, ...] = (1024, 512, 256, 128, 64), num_class: int = 1,
                 keep_dim: bool = False, skip: bool = False, batchNorm: bool = False, dropout: float = 0.,
                 padding: int = 0, out_size: Tuple[int, int] = (572, 572)):
        """
        Parameters
        ----------
        encode_chs: Tuple[int, ...]
            Tuple containing integers as elements indicating the amount of channels to be used for each UNETBlock layer
            in the encoder.
        decode_chs: Tuple[int, ...]
            Tuple containing integers as elements indicating the amount of channels to be used for each UNETBlock layer
            in the decoder.
        num_class: int
            Amount of classes to be segmented
        keep_dim: bool
            True or false indicating whether the output of the model should have the same dimensions as the input.
        skip: bool
            True or false indicating whether skip connections will be passed through a UNETBlock.
        batchNorm: bool
            True or false indicating whether batch normalization should be applied in all the convolutional layers.
        dropout: float
            Float value between 0. and 1. indicating probability p to completely zero out a given channel. A value of 0.
            is equal to no dropout being applied and a value of 1. would be equal to zeroing all channels.
        padding: int
            Amount of implicit padding on different sides of the input.
        out_size: Tuple[int, int]
            Tuple indicating the output height and width.
        """
        super().__init__()
        self.keep_dim = keep_dim
        self.output_size = out_size
        self.encode = UNETEncoder(encode_chs, batchNorm, dropout, padding)
        self.decode = UNETDecoder(decode_chs, batchNorm, dropout, padding)
        self.output = nn.Conv2d(decode_chs[-1], num_class, (1, 1))
        self.skip = skip
        if skip:
            self.skipConnects = SkipConnectsConvs(encode_chs[1:], batchNorm, dropout, padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        if self.skip:
            skip_features = self.skipConnects(encoder_features)
            out = self.decode(encoder_features[::-1][0], skip_features[::-1][1:])
        else:
            out = self.decode(encoder_features[::-1][0], encoder_features[::-1][1:])
        out = self.output(out)
        if self.keep_dim:
            out = F.interpolate(out, self.out_size)
        return out
