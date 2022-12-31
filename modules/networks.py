import antialiased_cnns
from torchvision import models
import numpy as np
import timm
import torch
from torch import nn
from torchvision.ops import FeaturePyramidNetwork

from modules.layers import BasicBlock
from utils.generic_utils import upsample


def double_basic_block(num_ch_in, num_ch_out, num_repeats=2):
    layers = nn.Sequential(BasicBlock(num_ch_in, num_ch_out))
    for i in range(num_repeats - 1):
        layers.add_module(f"conv_{i}", BasicBlock(num_ch_out, num_ch_out))
    return layers


class DepthDecoderPP(nn.Module):
    def __init__(
                self, 
                num_ch_enc, 
                scales=range(4), 
                num_output_channels=1,  
                use_skips=True
            ):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([64, 64, 128, 256])

        # decoder
        self.convs = nn.ModuleDict()
        # i is encoder depth (top to bottom)
        # j is decoder depth (left to right)
        for j in range(1, 5):
            max_i = 4 - j
            for i in range(max_i, -1, -1):

                num_ch_out = self.num_ch_dec[i]
                total_num_ch_in = 0

                num_ch_in = self.num_ch_enc[i + 1] if j == 1 else self.num_ch_dec[i + 1]
                self.convs[f"diag_conv_{i + 1}{j - 1}"] = BasicBlock(num_ch_in, 
                                                                    num_ch_out)
                total_num_ch_in += num_ch_out

                num_ch_in = self.num_ch_enc[i] if j == 1 else self.num_ch_dec[i]
                self.convs[f"right_conv_{i}{j - 1}"] = BasicBlock(num_ch_in, 
                                                                    num_ch_out)
                total_num_ch_in += num_ch_out

                if i + j != 4:
                    num_ch_in = self.num_ch_dec[i + 1]
                    self.convs[f"up_conv_{i + 1}{j}"] = BasicBlock(num_ch_in, 
                                                                    num_ch_out)
                    total_num_ch_in += num_ch_out

                self.convs[f"in_conv_{i}{j}"] = double_basic_block(
                                                                total_num_ch_in, 
                                                                num_ch_out,
                                                            )

                self.convs[f"output_{i}"] = nn.Sequential(
                BasicBlock(num_ch_out, num_ch_out) if i != 0 else nn.Identity(),
                nn.Conv2d(num_ch_out, self.num_output_channels, 1),
                )

    def forward(self, input_features):
        prev_outputs = input_features
        outputs = []
        depth_outputs = {}
        for j in range(1, 5):
            max_i = 4 - j
            for i in range(max_i, -1, -1):

                inputs = [self.convs[f"right_conv_{i}{j - 1}"](prev_outputs[i])]
                inputs += [upsample(self.convs[f"diag_conv_{i + 1}{j - 1}"](prev_outputs[i + 1]))]

                if i + j != 4:
                    inputs += [upsample(self.convs[f"up_conv_{i + 1}{j}"](outputs[-1]))]

                output = self.convs[f"in_conv_{i}{j}"](torch.cat(inputs, dim=1))
                outputs += [output]

                depth_outputs[f"log_depth_pred_s{i}_b1hw"] = self.convs[f"output_{i}"](output)

            prev_outputs = outputs[::-1]

        return depth_outputs


class CVEncoder(nn.Module):
    def __init__(self, num_ch_cv, num_ch_enc, num_ch_outs):
        super().__init__()

        self.convs = nn.ModuleDict()
        self.num_ch_enc = []

        self.num_blocks = len(num_ch_outs)

        for i in range(self.num_blocks):
            num_ch_in = num_ch_cv if i == 0 else num_ch_outs[i - 1]
            num_ch_out = num_ch_outs[i]
            self.convs[f"ds_conv_{i}"] = BasicBlock(num_ch_in, num_ch_out, 
                                                    stride=1 if i == 0 else 2)

            self.convs[f"conv_{i}"] = nn.Sequential(
                BasicBlock(num_ch_enc[i] + num_ch_out, num_ch_out, stride=1),
                BasicBlock(num_ch_out, num_ch_out, stride=1),
            )
            self.num_ch_enc.append(num_ch_out)

    def forward(self, x, img_feats):
        outputs = []
        for i in range(self.num_blocks):
            x = self.convs[f"ds_conv_{i}"](x)
            x = torch.cat([x, img_feats[i]], dim=1)
            x = self.convs[f"conv_{i}"](x)
            outputs.append(x)
        return outputs

class MLP(nn.Module):
    def __init__(self, channel_list, disable_final_activation = False):
        super(MLP, self).__init__()

        layer_list = []
        for layer_index in list(range(len(channel_list)))[:-1]:
            layer_list.append(
                            nn.Linear(channel_list[layer_index], 
                                channel_list[layer_index+1])
                            )
            layer_list.append(nn.LeakyReLU(inplace=True))

        if disable_final_activation:
            layer_list = layer_list[:-1]

        self.net = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.net(x)

class ResnetMatchingEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(
                self, 
                num_layers, 
                num_ch_out, 
                pretrained=True,
                antialiased=True,
            ):
        super().__init__()

        self.num_ch_enc = np.array([64, 64])

        model_source = antialiased_cnns if antialiased else models
        resnets = {18: model_source.resnet18,
                   34: model_source.resnet34,
                   50: model_source.resnet50,
                   101: model_source.resnet101,
                   152: model_source.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers"
                                                            .format(num_layers))

        encoder = resnets[num_layers](pretrained)

        resnet_backbone = [
            encoder.conv1,
            encoder.bn1,
            encoder.relu,
            encoder.maxpool,
            encoder.layer1,
        ]

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        self.num_ch_out = num_ch_out

        self.net = nn.Sequential(
            *resnet_backbone,
            nn.Conv2d(self.num_ch_enc[-1], 128, (1, 1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(
                    128, 
                    self.num_ch_out, 
                    (3, 3), 
                    padding=1, 
                    padding_mode="replicate"
                ),
            nn.InstanceNorm2d(self.num_ch_out)
        )

    def forward(self, input_image):
        return self.net(input_image)

class UNetMatchingEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
                                        "mnasnet_100", 
                                        pretrained=True, 
                                        features_only=True,
                                    )

        self.decoder = FeaturePyramidNetwork(
                                        self.encoder.feature_info.channels(), 
                                        out_channels=32,
                                    )
        self.outconv = nn.Sequential(
                                    nn.LeakyReLU(0.2, True),
                                    nn.Conv2d(32, 16, 1),
                                    nn.InstanceNorm2d(16),
                                )

    def forward(self, x):
        encoder_feats = {f"feat_{i}": f for i, f in enumerate(self.encoder(x))}
        return self.outconv(self.decoder(encoder_feats)["feat_1"])
