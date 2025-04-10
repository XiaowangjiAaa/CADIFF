from .unet_utils import SSformer
from .unet import UNetModel, timestep_embedding, SiLU, conv_nd, normalization, zero_module
import torch
import torch.nn as nn



class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


class Upsample_res(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample_res, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)



class ResUnet(nn.Module):
    def __init__(self, channel, filters=[64, 128, 256, 512]):
        super(ResUnet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 1, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[1], 2, 1)
        self.residual_conv_3 = ResidualConv(filters[1], filters[2], 2, 1)
        self.residual_conv_4 = ResidualConv(filters[2], filters[2], 2, 1)
        self.residual_conv_5 = ResidualConv(filters[2], filters[3], 2, 1)

        self.bridge = ResidualConv(filters[3], filters[3], 2, 1)

        self.upsample_1 = Upsample_res(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[3], filters[3], 1, 1)

        self.upsample_2 = Upsample_res(filters[3], filters[3], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_3 = Upsample_res(filters[2], filters[2], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[2], filters[2], 1, 1)

        self.upsample_4 = Upsample_res(filters[2], filters[2], 2, 2)
        self.up_residual_conv4 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        
        self.upsample_5 = Upsample_res(filters[1], filters[1], 2, 2)
        self.up_residual_conv5 = ResidualConv(filters[1] + filters[1], filters[0], 1, 1)
        self.up_residual_conv6 = ResidualConv(filters[0] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], 1, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        x4 = self.residual_conv_3(x3)
        x5 = self.residual_conv_4(x4)
        x6 = self.residual_conv_5(x5)
        # Bridge
        x7 = self.bridge(x6)
        # Decode
        x7 = self.upsample_1(x7)
        x8 = torch.cat([x7, x6], dim=1)
        x9 = self.up_residual_conv1(x8)

        x9 = self.upsample_2(x9)
        x10 = torch.cat([x9, x5], dim=1)
        x11 = self.up_residual_conv2(x10)

        x11 = self.upsample_3(x11)
        x12 = torch.cat([x11, x4], dim=1)
        x13 = self.up_residual_conv3(x12)

        x13 = self.upsample_4(x13)
        x14 = torch.cat([x13, x3], dim=1)
        x15 = self.up_residual_conv4(x14)

        x15 = self.upsample_5(x15)
        x16 = torch.cat([x15, x2], dim=1)
        x17 = self.up_residual_conv5(x16)

        x18 = torch.cat([x17, x1], dim=1)
        x19 = self.up_residual_conv6(x18)

        output = self.output_layer(x19)

        return output
    
class ResUnet_encoder(nn.Module):
    def __init__(self, channel, filters=[64, 128, 256, 512]):
        super(ResUnet_encoder, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 1, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[1], 2, 1)
        self.residual_conv_3 = ResidualConv(filters[1], filters[2], 2, 1)
        self.residual_conv_4 = ResidualConv(filters[2], filters[2], 2, 1)
        self.residual_conv_5 = ResidualConv(filters[2], filters[3], 2, 1)

        self.bridge = ResidualConv(filters[3], filters[3], 2, 1)
    
    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        x4 = self.residual_conv_3(x3)
        x5 = self.residual_conv_4(x4)
        x6 = self.residual_conv_5(x5)
        # Bridge
        x7 = self.bridge(x6)

        return x7

class UNetModel_WithSSF(UNetModel):
    def __init__(self, in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions, dropout=0, channel_mult=(1, 2, 4, 8), conv_resample=True, dims=2, num_classes=None, use_checkpoint=False, num_heads=1, num_heads_upsample=-1, use_scale_shift_norm=False):
        super().__init__(in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions, dropout, channel_mult, conv_resample, dims, num_classes, use_checkpoint, num_heads, num_heads_upsample, use_scale_shift_norm)

        self.condition_encoder = None

        time_embed_dim = model_channels * 4
        dim = self.model_channels * max(self.channel_mult)
        heads = 8
        dim_head = int(dim / heads)
        self.ssformer = SSformer(dim=dim, image_size=8, time_emd_dim=time_embed_dim, dim_head=dim_head, heads=heads)

        self.out = nn.Sequential(
            normalization(model_channels),
            SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 1, padding=0)),
        )

    def load_resunet(self, if_pre=False, in_channels=3, model_path=None):
        self.condition_encoder = ResUnet_encoder(in_channels)
        if if_pre:
            state_dict = torch.load(model_path)
            encoder_state_dict = {k: v for k, v in state_dict.items() if k.startswith('in') or k.startswith('bri') or k.startswith('res')}
            self.condition_encoder.load_state_dict(encoder_state_dict)
            self.condition_encoder.to("cuda")
            self.condition_encoder.requires_grad_(False)

    def forward(self, noisy_mask, timesteps, img):
        """
        Apply the model to an input batch.

        :param img: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        img = img.type(self.inner_dtype)
        h = noisy_mask.type(self.inner_dtype)
        inner_code = self.condition_encoder(img)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.ssformer(h, inner_code, emb)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
        h = h.type(noisy_mask.dtype)
        return self.out(h)