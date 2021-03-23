from .unet_parts import *
import torch.nn as nn
import torch.nn.functional as F


class UNET(nn.Module):
    def __init__(self, encode_chs=(3, 64, 128, 256, 512, 1024), decode_chs=(1024, 512, 256, 128, 64), num_class=1,
                 keep_dim=False, out_size=(572, 572)):
        super().__init__()
        self.keep_dim = keep_dim
        self.output_size = out_size
        self.encode = UNETEncoder(encode_chs)
        self.decoder = UNETDecoder(decode_chs)
        self.output = nn.Conv2d(decode_chs[-1], num_class, (1, 1))

    def forward(self, x):
        encoder_features = self.encoder(x)
        out = self.decoder(encoder_features[::-1][0], encoder_features[::-1][1:])
        out = self.output(out)
        if self.keep_dim:
            out = F.interpolate(out, self.out_size)
        return out
