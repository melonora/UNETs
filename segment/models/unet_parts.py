import torch
import torch.nn as nn
# TODO: implement type hinting
from typing import Tuple, List, Any


class UNETBlock(nn.Module):
    """ Generalization of the standard UNET block as used in the encoder and decoder in UNET paper. In the UNET paper
    batch normalization, dropout and padding are not applied.

    Attributes
    ----------
    ublock: nn.sequential
        Pytorch sequential container containing two convolutional layers.
    """
    @staticmethod
    def append_block(block: List[Any, ...], ch_in: int, ch_out: int, batchNorm: bool, dropout: float, padding: int)\
            -> List[Any, ...]:
        """ Function to append a convolutional layer to a list.

        Parameters
        ----------
        block : List[Any]
            Empty list or list containing a given amount of UNETBlocks as elements
        ch_in : int
            Amount of input channels
        ch_out : int
            Amount of output channels
        batchNorm : bool
            Boolean True or False indicating whether batch normalization should be applied with default parameters in
            Pytorch. These default parameters are eps=1e-05 (value added to denominator for numerical stability to avoid
            division by zero), momentum=0.1, affine=True (learnable parameters for batch normalization),
            track_running_stats=True (whether to use running mean and variance statistics).
        dropout : float
            Float value between 0. and 1. indicating probability p to completely zero out a given channel. A value of 0.
            is equal to no dropout being applied and a value of 1. would be equal to zeroing all channels.
        padding : int
            Amount of implicit padding on different sides of the input.

        Returns
        -------
        block: List[Any, ...]
            List containing at least one full convolutional layer block.
        """
        block.append(nn.Conv2d(ch_in, ch_out, (3, 3), padding=(padding, padding)))
        block.append(nn.ReLU())
        if batchNorm:
            block.append(nn.BatchNorm2d(ch_out))
        if dropout != 0:
            block.append(nn.Dropout2d(p=dropout))
        return block

    def __init__(self, ch_in: int, ch_out: int, batchNorm: bool = False, dropout: float = 0., padding: int = 0):
        """
        Parameters
        ----------
        ch_in : int
            Amount of input channels
        ch_out : int
            Amount of output channels
        batchNorm : bool
            Boolean True or False indicating whether batch normalization should be applied with default parameters in
            Pytorch. These default parameters are eps=1e-05 (value added to denominator for numerical stability to avoid
            division by zero), momentum=0.1, affine=True (learnable parameters for batch normalization),
            track_running_stats=True (whether to use running mean and variance statistics).
        dropout : float
            Float value between 0. and 1. indicating probability p to completely zero out a given channel. A value of 0.
            is equal to no dropout being applied and a value of 1. would be equal to zeroing all channels.
        padding : int
            Amount of implicit padding on different sides of the input.
        """
        super().__init__()
        ublock = list()
        ublock = self.append_block(ublock, ch_in, ch_out, batchNorm, dropout, padding)
        ublock = self.append_block(ublock, ch_out, ch_out, batchNorm, dropout, padding)
        self.ublock = nn.Sequential(*ublock)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass function for UNET_block as required when using Pytorch.

        Parameters
        ----------
        x : torch.Tensor
            Input for the UNET block.

        Returns
        -------
        x: torch.Tensor
            Feature map of dimensions N x C x H x W where N is the amount of image stacks in the
            minibatch, C is the amount of channels, H is the height and W is the width.
        """
        return self.ublock(x)


class UNETEncoder(nn.Module):
    """ Generalization of the encoder part of the UNET model as described in the UNET paper. If default parameters are
    used the encoder is equal to the encoder from the UNET paper. Batch normalization, dropout and padding can
    additionally be applied.

    Attributes
    ----------
    encode_UNET: nn.ModuleList
        List containing UNETBlocks as submodules for the encoder.
    mpool: nn.MaxPool
        MaxPool to be applied after each pass through a UNETBlock.
    """
    def __init__(self, channels: Tuple[int, ...] = (3, 64, 128, 256, 512, 1024), batchNorm: bool = False,
                 dropout: float = 0., padding: int = 0):
        """
        Parameters
        ----------
        channels: Tuple[int, ...]
            Tuple containing integers as elements indicating the amount of channels to be used for each UNETBlock layer.
        batchNorm: bool
            True or false indicating whether batch normalization has to be applied
        dropout : float
            Float value between 0. and 1. indicating probability p to completely zero out a given channel. A value of 0.
            is equal to no dropout being applied and a value of 1. would be equal to zeroing all channels.
        padding : int
            Amount of implicit padding on different sides of the input.
        """
        super().__init__()
        self.encode_UNET = nn.ModuleList([UNETBlock(channels[i], channels[i+1], batchNorm, dropout, padding)
                                          for i in range(len(channels)-1)])
        self.mpool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> List[Any, ...]:
        """Forward pass function for the encoder part of UNET as required when using Pytorch. This function returns
        the features obtained from each UNET_block in the encoder part of the UNET model.

        Parameters
        ----------
        x : torch.Tensor
            Input for the UNET encoder where each input channel in the case of MICS represents an image containing
            spatial intensity for a given marker.

        Returns
        -------
        features: List[Any, ...]
            List containing feature maps of the different layers of the UNET encoder.
        """
        features = []
        for u_block in self.encode_UNET:
            x = u_block(x)
            features.append(x)
            x = self.mpool(x)
        return features


class UNETDecoder(nn.Module):
    """ Generalization of the decoder part of the UNET model as described in the UNET paper. If default parameters
    are used the decoder is equal to the decoder used in the UNET paper. Batch normalization, dropout and padding
    can additionally be applied.

    Attributes
    ----------
    ups: nn.ModuleList
        List containing the upsample steps of the decoder as submodules.
    decode_UNET: nn.ModuleList
        List containing the UNETBlocks of the decoder as submodules.
    channels: Tuple[int, ...]
        Tuple containing integers as element indicating the amount of channels for each layer of the decoder.
    """
    def __init__(self, channels: Tuple[int, ...] = (1024, 512, 256, 128, 64), batchNorm: bool = False,
                 dropout: float = 0., padding: int = 0, mode: str = "convTrans"):
        """"
         Parameters
        ----------
        channels: Tuple[int, ...]
            Tuple containing integers as elements indicating the amount of channels to be used for each UNETBlock layer.
        batchNorm: bool
            True or false indicating whether batch normalization has to be applied
        dropout : float
            Float value between 0. and 1. indicating probability p to completely zero out a given channel. A value of 0.
            is equal to no dropout being applied and a value of 1. would be equal to zeroing all channels.
        padding : int
            Amount of implicit padding on different sides of the input.
        mode: str
            Mode by which the upsampling in the decoder is performed. Must be equal to either "convTrans" or
            "bilinear".
        """
        super().__init__()
        if mode == "convTrans":
            self.ups = nn.ModuleList(
                [nn.ConvTranspose2d(channels[i], channels[i+1], (2, 2), (2, 2)) for i in range(len(channels)-1)])
        if mode == "up_bilinear":
            self.ups = nn.ModuleList([nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(channels[i], channels[i+1], (1, 1)),
            ) for i in range(len(channels)-1)])
        self.decode_UNET = nn.ModuleList([UNETBlock(channels[i], channels[i+1], batchNorm, dropout, padding)
                                          for i in range(len(channels)-1)])
        self.channels = channels

    @staticmethod
    def crop(encoder_feature: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Function to perform a center crop for the skip connection as the feature map size on the left side of the
        UNET model has a larger size than the feature map on the right side. Cropping allows for concatenating feature
        maps of the contracting and expanding path of the UNET model.

        Parameters
        ----------
        encoder_feature : torch.Tensor
            Given feature map of the encoder part of the UNET model.
        x : torch.Tensor
            The output from the upsampled output of the previous decoder UNET block.

        Returns
        -------
        encoder_feature : torch.Tensor
            Center cropped feature map of dimensions N x C x H x W where N is the amount of image stacks in the
            minibatch, C is the amount of channels, H is the height and W is the Width.
        """
        _, _, H, W = x.shape
        _, _, TH, TW = encoder_feature.shape
        diff_h = int(round((TH-H) / 2.))
        diff_w = int(round((TW - W) / 2.))
        return encoder_feature[:, :, diff_h: (diff_h + H), diff_w: (diff_w + W)]

    def forward(self, x: torch.Tensor, encoder_features: List[torch.Tensor, ...]) -> torch.Tensor:
        """Forward pass function for the decoder part of UNET as required when using Pytorch.

        Parameters
        ----------
        x : torch.Tensor
            Output from the encoder part of the UNET model
        encoder_features : List[torch.Tensor, ...]
            The feature maps from the encoding part of the UNET model.

        Returns
        -------
        x: torch.Tensor
            Feature map of dimensions N x C x H x W where N is the amount of image stacks in the
            minibatch, C is the amount of channels, H is the height and W is the Width
        """
        for i in range(len(self.channels)-1):
            x = self.ups[i](x)
            encoder_feature = self.crop(encoder_features[i], x)
            x = torch.cat([x, encoder_feature], dim=1)
            x = self.decode_UNET[i](x)
        return x


class SkipConnectsConvs(nn.Module):
    """ Class allowing for skip connections of a UNET model to be passed through a convolutional layer.
     https://github.com/upashu1/Pytorch-UNet-2 claims this works better when working with noisy microarray data."""
    def __init__(self, channels=(64, 128, 256, 512, 1024), batchNorm: bool = False, dropout: float = 0., padding=0):
        super().__init__()
        self.channels = channels
        self.skips = nn.ModuleList([UNETBlock(channels[i], channels[i], batchNorm, dropout, padding)
                                    for i in range(len(channels))])

    def forward(self, encoder_features: List[torch.Tensor, ...]) -> List[torch.Tensor, ...]:
        """ Forward pass function for the skip connection part of adjusted UNET as required when using Pytorch.

        Parameters
        ----------
        encoder_features : List[torch.Tensor, ...]
            The feature maps from the encoding part of the UNET model.

        Returns
        -------
        features: List[torch.Tensor, ...]
            The feature maps from the encoding part of the UNET model passed through a convolutional layer.
        """
        features = []
        for i in range(len(self.channels)):
            x = self.skips[i](encoder_features[i])
            features.append(x)
        return features
