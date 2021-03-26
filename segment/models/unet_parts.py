import torch
import torch.nn as nn
# TODO: implement type hinting
# from typing import Tuple


class UNETBlock(nn.Module):
    """Generalization of the standard UNET block as used in the encoder and decoder in UNET paper"""
    @staticmethod
    def append_block(block, ch_in, ch_out, batchNorm, dropout, padding):
        """

        Parameters
        ----------
        block :
            
        ch_in :
            
        ch_out :
            
        batchNorm :
            
        dropout :
            
        padding :
            

        Returns
        -------

        """
        block.append(nn.Conv2d(ch_in, ch_out, (3, 3), padding=padding))
        block.append(nn.ReLU())
        if batchNorm:
            block.append(nn.BatchNorm2d(ch_out))
        if dropout != 0:
            block.append(nn.Dropout2d(p=dropout))
        return block

    def __init__(self, ch_in: int, ch_out: int, batchNorm: bool = False, dropout: float = 0., padding: int = 0):
        super().__init__()
        ublock = list()
        ublock = self.append_block(ublock, ch_in, ch_out, batchNorm, dropout, padding)
        ublock = self.append_block(ublock, ch_out, ch_out, batchNorm, dropout, padding)
        self.ublock = nn.Sequential(*ublock)

    def forward(self, x):
        """Forward pass function for UNET_block as required when using Pytorch.

        Parameters
        ----------
        x : torch.Tensor
            Input for the UNET block.

        Returns
        -------

        
        """
        return self.ublock(x)


class UNETEncoder(nn.Module):
    """ """
    # TODO: Adjust to work with extended UNETBlock
    """Generalization of the encoder part of the UNET model as described in the UNET paper."""
    def __init__(self, channels=(3, 64, 128, 256, 512, 1024), batchNorm: bool = False, dropout: float = 0., padding=0):
        super().__init__()
        self.encode_UNET = nn.ModuleList([UNETBlock(channels[i], channels[i+1], batchNorm, dropout, padding)
                                          for i in range(len(channels)-1)])
        self.mpool = nn.MaxPool2d(2)

    def forward(self, x):
        """Forward pass function for the encoder part of UNET as required when using Pytorch. This function returns
        the features obtained from each UNET_block in the encoder part of the UNET model.

        Parameters
        ----------
        x : torch.Tensor
            Input for the UNET encoder where each input channel in the case of MICS represents an image containing
            spatial intensity for a given marker.

        Returns
        -------

        
        """
        features = []
        for u_block in self.encode_UNET:
            x = u_block(x)
            features.append(x)
            x = self.mpool(x)
        return features


class UNETDecoder(nn.Module):
    """Generalization of the decoder part of the UNET model as described in the UNET paper"""
    def __init__(self, channels=(1024, 512, 256, 128, 64), batchNorm: bool = False, dropout: float = 0., padding=0,
                 mode="convTrans"):
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
    def crop(encoder_feature, x):
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

        
        """
        _, _, H, W = x.shape
        _, _, TH, TW = encoder_feature.shape
        diff_h = int(round((TH-H) / 2.))
        diff_w = int(round((TW - W) / 2.))
        return encoder_feature[:, :, diff_h: (diff_h + H), diff_w: (diff_w + W)]

    def forward(self, x, encoder_features):
        """Forward pass function for the decoder part of UNET as required when using Pytorch.

        Parameters
        ----------
        x : torch.Tensor
            Output from the encoder part of the UNET model
        encoder_features : List[torch.Tensor, ...]
            The feature maps from the encoding part of the UNET model.

        Returns
        -------

        
        """
        for i in range(len(self.channels)-1):
            x = self.ups[i](x)
            encoder_feature = self.crop(encoder_features[i], x)
            x = torch.cat([x, encoder_feature], dim=1)
            x = self.decode_UNET[i](x)
        return x


class SkipConnectsConvs(nn.Module):
    """ """
    def __init__(self, channels=(64, 128, 256, 512, 1024), batchNorm: bool = False, dropout: float = 0., padding=0):
        super().__init__()
        self.channels = channels
        self.skips = nn.ModuleList([UNETBlock(channels[i], channels[i], batchNorm, dropout, padding)
                                    for i in range(len(channels)-1)])

    def forward(self, encoder_features):
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
        for i in range(len(self.channels)-1):
            x = self.skips[i](encoder_features[i])
            features.append(x)
        return features
