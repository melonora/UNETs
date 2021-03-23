import torch
import torch.nn as nn
import torchvision


class UNETBlock(nn.Module):
    """Generalization of the standard UNET block as used in the encoder and decoder in UNET paper"""
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, (3, 3))
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(ch_out, ch_out, (3, 3))

    def forward(self, x):
        """ Forward pass function for UNET_block as required when using Pytorch.

        Parameters
        ----------
        x : torch.Tensor
            Input for the UNET block.
            

        Returns
        -------
        torch.Tensor
            Output of the UNET block as per used in the UNET paper.
        """
        return self.relu(self.conv2(self.relu(self.conv1(x))))


class UNETEncoder(nn.Module):
    """Generalization of the encoder part of the UNET model as described in the UNET paper."""
    def __init__(self, channels=(3, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.encode_UNET = nn.ModuleList([UNETBlock(channels[i], channels[i+1]) for i in range(len(channels)-1)])
        self.mpool = nn.MaxPool2d(2)

    def forward(self, x):
        """ Forward pass function for the encoder part of UNET as required when using Pytorch. This function returns
        the features obtained from each UNET_block in the encoder part of the UNET model.

        Parameters
        ----------
        x : torch.Tensor
            Input for the UNET encoder where each input channel in the case of MICS represents an image containing
            spatial intensity for a given marker.

        Returns
        -------
        List[torch.Tensor, ...]
            List of torch.Tensors where each torch.Tensor element of the list corresponds to the features of a given
            UNET_block.
        """
        features = []
        for u_block in self.encode_UNET:
            x = u_block(x)
            features.append(x)
            x = self.mpool(x)
        return features


class UNETDecoder(nn.Module):
    """Generalization of the decoder part of the UNET model as described in the UNET paper """
    def __init__(self, channels=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(channels[i], channels[i+1], 2, 2) for i in range(len(channels)-1)])
        self.decode_UNET = nn.ModuleList([UNETBlock(channels[i], channels[i+1]) for i in range(len(channels)-1)])
        self.channels = channels

    def crop(self, encoder_feature, x):
        """ Function to perform a centercrop for the skip connection as the feature map size on the left side of the
        UNET model has a larger size than the feature map on the right side. Cropping allows for concatenating feature
        maps of the contracting and expanding path of the UNET model.

        Parameters
        ----------
        encoder_feature : torch.Tensor
            Given feature map of the encoder part of the UNET model.
            
        x : torch.Tensor
            The
            

        Returns
        -------

        """
        _, _, H, W = x.shape
        encoder_feature = torchvision.transforms.CenterCrop([H, W])(encoder_feature)
        return encoder_feature

    def forward(self, x, encoder_features):
        """ Forward pass function for the decoder part of UNET as required when using Pytorch.

        Parameters
        ----------
        x : torch.Tensor
            Output from the encoder part of the UNET model
            
        encoder_features : List[torch.Tensor, ...]
            The feature maps from the encoding part of the UNET model.
            

        Returns
        -------
        torch.Tensor
            Output from the UNET decoder.
        """
        for i in range(len(self.channels)-1):
            x = self.upconvs[i](x)
            encoder_feature = self.crop(encoder_features[i], x)
            x = torch.cat([x, encoder_feature])
            x = self.decode_UNET[i](x)
        return x
