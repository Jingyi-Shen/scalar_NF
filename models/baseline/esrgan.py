# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# modified by JY, 03/08/2023
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from torchvision import transforms
from torchvision.models.vgg import vgg19
# from torchvision.models.feature_extraction import create_feature_extractor

__all__ = [
    "Discriminator", "RRDBNet", "ContentLoss",
    "discriminator", "rrdbnet_x1", "rrdbnet_x2", "rrdbnet_x4", "rrdbnet_x8", "content_loss",
]


class _ResidualDenseBlock(nn.Module):
    """Achieves densely connected convolutional layers.
    `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993v5.pdf>` paper.

    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(_ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv3d(channels + growth_channels * 0, growth_channels, 3, 1, 1)
        self.conv2 = nn.Conv3d(channels + growth_channels * 1, growth_channels, 3, 1, 1)
        self.conv3 = nn.Conv3d(channels + growth_channels * 2, growth_channels, 3, 1, 1)
        self.conv4 = nn.Conv3d(channels + growth_channels * 3, growth_channels, 3, 1, 1)
        self.conv5 = nn.Conv3d(channels + growth_channels * 4, channels, 3, 1, 1)

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.identity = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out1 = self.leaky_relu(self.conv1(x))
        out2 = self.leaky_relu(self.conv2(torch.cat([x, out1], 1)))
        out3 = self.leaky_relu(self.conv3(torch.cat([x, out1, out2], 1)))
        out4 = self.leaky_relu(self.conv4(torch.cat([x, out1, out2, out3], 1)))
        out5 = self.identity(self.conv5(torch.cat([x, out1, out2, out3, out4], 1)))
        out = torch.mul(out5, 0.2)
        out = torch.add(out, identity)

        return out


class _ResidualResidualDenseBlock(nn.Module):
    """Multi-layer residual dense convolution block.

    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(_ResidualResidualDenseBlock, self).__init__()
        self.rdb1 = _ResidualDenseBlock(channels, growth_channels)
        self.rdb2 = _ResidualDenseBlock(channels, growth_channels)
        self.rdb3 = _ResidualDenseBlock(channels, growth_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        out = torch.mul(out, 0.2)
        out = torch.add(out, identity)

        return out


# class Discriminator(nn.Module):
#     def __init__(self) -> None:
#         super(Discriminator, self).__init__()
#         self.features = nn.Sequential(
#             # input size. (1) x 24 x 24 x 24
#             nn.Conv3d(1, 8, 3, 1, 1, bias=True),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv3d(8, 8, 4, 2, 1, bias=False),
#             nn.BatchNorm3d(8),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv3d(8, 8, 3, 1, 1, bias=False),
#             nn.BatchNorm3d(8),
#             nn.LeakyReLU(0.2, True),
#             # state size. (8) x 12 x 12 x 12
#             nn.Conv3d(8, 16, 4, 2, 1, bias=False),
#             nn.BatchNorm3d(16),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv3d(16, 32, 3, 1, 1, bias=False),
#             nn.BatchNorm3d(32),
#             nn.LeakyReLU(0.2, True),
#             # state size. (32) x 6 x 6 x 6
#             nn.Conv3d(32, 64, 4, 2, 1, bias=False),
#             nn.BatchNorm3d(64),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv3d(64, 128, 3, 1, 1, bias=False),
#             nn.BatchNorm3d(128),
#             nn.LeakyReLU(0.2, True),
#             # state size. (128) x 3 x 3 x 3
#             # nn.Conv3d(512, 512, 4, 2, 1, bias=False),
#             # nn.BatchNorm3d(512),
#             # nn.LeakyReLU(0.2, True),
#             # nn.Conv3d(512, 512, 3, 1, 1, bias=False),
#             # nn.BatchNorm3d(512),
#             # nn.LeakyReLU(0.2, True),
#         )

#         self.classifier = nn.Sequential(
#             nn.Linear(128 * 3 * 3 * 3, 100),
#             nn.LeakyReLU(0.2, True),
#             nn.Linear(100, 1)
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         out = self.features(x)
#         out = torch.flatten(out, 1)
#         out = self.classifier(out)

#         return out

class Discriminator(nn.Module):
    def __init__(self, num_conv_block=3):
        super(Discriminator, self).__init__()

        block = []

        in_channels = 1
        out_channels = 32

        for _ in range(num_conv_block):
            block += [
                    #   nn.ReflectionPad3d(1),
                      nn.Conv3d(in_channels, out_channels, 3, 1, 1),
                      nn.LeakyReLU(),
                      nn.BatchNorm3d(out_channels)]
            in_channels = out_channels

            block += [
                    #   nn.ReflectionPad3d(1),
                      nn.Conv3d(in_channels, out_channels, 3, 2, 1),
                      nn.LeakyReLU()]
            out_channels *= 2

        out_channels //= 2
        in_channels = out_channels

        block += [nn.Conv3d(in_channels, out_channels, 3),
                  nn.LeakyReLU(0.2),
                  nn.Conv3d(out_channels, out_channels, 3)]

        self.feature_extraction = nn.Sequential(*block)

        # self.avgpool = nn.AdaptiveAvgPool3d((512, 512))

        self.classification = nn.Sequential(
            nn.Linear(512, 100),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.size(0), -1)
        x = self.classification(x)
        return x


def upsample_block2d(nf, scale_factor=2):
    block = []
    for _ in range(scale_factor//2):
        block += [
            nn.Conv3d(nf, nf * (2 ** 2), 1),
            nn.PixelShuffle(2),
            nn.ReLU()
        ]
    return nn.Sequential(*block)


class PixelShuffle3d(nn.Module):
    def __init__(self, scale):
        super(PixelShuffle3d, self).__init__()
        self.scale = scale
    
    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale ** 3
        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width)
        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        return output.view(batch_size, nOut, out_depth, out_height, out_width)
        

def upsample_block3d(nf, scale_factor=2):
    block = []
    # shuffle3d = PixelShuffle3d(scale=2)
    for _ in range(scale_factor//2):
        block += [
            nn.Conv3d(nf, nf * (2 ** 3), 1),
            PixelShuffle3d(scale=2),
            nn.ReLU()
        ]
    return nn.Sequential(*block)


class ESRGAN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, nf=32, gc=16, scale_factor=2, n_basic_block=23):
        super(ESRGAN, self).__init__()

        # self.conv1 = nn.Sequential(nn.ReflectionPad3d(1), nn.Conv3d(in_channels, nf, 3), nn.ReLU())
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, nf, 3, 1, 1), nn.ReLU())
        self.shuffle3d = PixelShuffle3d(scale=2)

        basic_block_layer = []

        for _ in range(n_basic_block):
            basic_block_layer += [_ResidualResidualDenseBlock(nf, gc)]

        self.basic_block = nn.Sequential(*basic_block_layer)
        self.conv2 = nn.Sequential(nn.Conv3d(nf, nf, 3, 1, 1), nn.ReLU())
        self.upsample = upsample_block3d(nf, scale_factor=scale_factor)
        self.conv3 = nn.Sequential(nn.Conv3d(nf, nf, 3, 1, 1), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv3d(nf, out_channels, 3, 1, 1), nn.ReLU())

        # self.conv2 = nn.Sequential(nn.ReflectionPad3d(1), nn.Conv3d(nf, nf, 3), nn.ReLU())
        # self.upsample = upsample_block3d(nf, scale_factor=scale_factor)
        # self.conv3 = nn.Sequential(nn.ReflectionPad3d(1), nn.Conv3d(nf, nf, 3), nn.ReLU())
        # self.conv4 = nn.Sequential(nn.ReflectionPad3d(1), nn.Conv3d(nf, out_channels, 3), nn.ReLU())

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.basic_block(x1)
        x = self.conv2(x)

        x = self.upsample(x + x1)

        x = self.conv3(x)
        x = self.conv4(x)
        return x


class RRDBNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            channels: int = 32,
            growth_channels: int = 16,
            num_blocks: int = 23,
            upscale_factor: int = 2,
    ) -> None:
        super(RRDBNet, self).__init__()
        self.upscale_factor = upscale_factor

        # The first layer of convolutional layer.
        self.conv1 = nn.Conv3d(in_channels, channels, 3, 1, 1)

        # Feature extraction backbone network.
        trunk = []
        for _ in range(num_blocks):
            trunk.append(_ResidualResidualDenseBlock(channels, growth_channels))
        self.trunk = nn.Sequential(*trunk)

        # After the feature extraction network, reconnect a layer of convolutional blocks.
        self.conv2 = nn.Conv3d(channels, channels, 3, 1, 1)

        # Upsampling convolutional layer.
        if upscale_factor == 2:
            self.upsampling1 = nn.Sequential(
                nn.Conv3d(channels, channels, 3, 1, 1),
                nn.LeakyReLU(0.2, True)
            )
        if upscale_factor == 4:
            self.upsampling1 = nn.Sequential(
                nn.Conv3d(channels, channels, 3, 1, 1),
                nn.LeakyReLU(0.2, True)
            )
            self.upsampling2 = nn.Sequential(
                nn.Conv3d(channels, channels, 3, 1, 1),
                nn.LeakyReLU(0.2, True)
            )
        if upscale_factor == 8:
            self.upsampling1 = nn.Sequential(
                nn.Conv3d(channels, channels, 3, 1, 1),
                nn.LeakyReLU(0.2, True)
            )
            self.upsampling2 = nn.Sequential(
                nn.Conv3d(channels, channels, 3, 1, 1),
                nn.LeakyReLU(0.2, True)
            )
            self.upsampling3 = nn.Sequential(
                nn.Conv3d(channels, channels, 3, 1, 1),
                nn.LeakyReLU(0.2, True)
            )

        # Reconnect a layer of convolution block after upsampling.
        self.conv3 = nn.Sequential(
            nn.Conv3d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.2, True)
        )

        # Output layer.
        self.conv4 = nn.Conv3d(channels, out_channels, 3, 1, 1)

        # Initialize all layer
        self._initialize_weights()

    # The model should be defined in the Torch.script method.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.conv1(x)
        out = self.trunk(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)

        if self.upscale_factor == 2:
            out = self.upsampling1(F.interpolate(out, scale_factor=2, mode="nearest"))
        if self.upscale_factor == 4:
            out = self.upsampling1(F.interpolate(out, scale_factor=2, mode="nearest"))
            out = self.upsampling2(F.interpolate(out, scale_factor=2, mode="nearest"))
        if self.upscale_factor == 8:
            out = self.upsampling1(F.interpolate(out, scale_factor=2, mode="nearest"))
            out = self.upsampling2(F.interpolate(out, scale_factor=2, mode="nearest"))
            out = self.upsampling3(F.interpolate(out, scale_factor=2, mode="nearest"))

        out = self.conv3(out)
        out = self.conv4(out)

        out = torch.clamp_(out, 0.0, 1.0)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight)
                module.weight.data *= 0.1
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


# only for 2D
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        vgg = vgg19(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:35]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.l1_loss = nn.L1Loss()

    def forward(self, high_resolution, fake_high_resolution):
        perception_loss = self.l1_loss(self.loss_network(high_resolution), self.loss_network(fake_high_resolution))
        return perception_loss


# class ContentLoss(nn.Module):
#     """Constructs a content loss function based on the VGG19 network.
#     Using high-level feature mapping layers from the latter layers will focus more on the texture content of the image.

#     Paper reference list:
#         -`Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network <https://arxiv.org/pdf/1609.04802.pdf>` paper.
#         -`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks                    <https://arxiv.org/pdf/1809.00219.pdf>` paper.
#         -`Perceptual Extreme Super Resolution Network with Receptive Field Block               <https://arxiv.org/pdf/2005.12597.pdf>` paper.

#      """

#     def __init__(
#             self,
#             feature_model_extractor_node: str,
#             feature_model_normalize_mean: list,
#             feature_model_normalize_std: list
#     ) -> None:
#         super(ContentLoss, self).__init__()
#         # Get the name of the specified feature extraction node
#         self.feature_model_extractor_node = feature_model_extractor_node
#         # Load the VGG19 model trained on the ImageNet dataset.
#         model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
#         # Extract the thirty-fifth layer output in the VGG19 model as the content loss.
#         self.feature_extractor = create_feature_extractor(model, [feature_model_extractor_node])
#         # set to validation mode
#         self.feature_extractor.eval()

#         # The preprocessing method of the input data.
#         # This is the VGG model preprocessing method of the ImageNet dataset
#         self.normalize = transforms.Normalize(feature_model_normalize_mean, feature_model_normalize_std)

#         # Freeze model parameters.
#         for model_parameters in self.feature_extractor.parameters():
#             model_parameters.requires_grad = False

#     def forward(self, sr_tensor: torch.Tensor, gt_tensor: torch.Tensor) -> torch.Tensor:
#         # Standardized operations
#         sr_tensor = self.normalize(sr_tensor)
#         gt_tensor = self.normalize(gt_tensor)

#         sr_feature = self.feature_extractor(sr_tensor)[self.feature_model_extractor_node]
#         gt_feature = self.feature_extractor(gt_tensor)[self.feature_model_extractor_node]

#         # Find the feature map difference between the two images
#         loss = F.l1_loss(sr_feature, gt_feature)

#         return loss


def discriminator() -> Discriminator:
    model = Discriminator()

    return model


def rrdbnet_x1(**kwargs: Any) -> RRDBNet:
    model = RRDBNet(upscale_factor=1, **kwargs)

    return model


def rrdbnet_x2(**kwargs: Any) -> RRDBNet:
    model = RRDBNet(upscale_factor=2, **kwargs)

    return model


def rrdbnet_x4(**kwargs: Any) -> RRDBNet:
    model = RRDBNet(upscale_factor=4, **kwargs)

    return model


def rrdbnet_x8(**kwargs: Any) -> RRDBNet:
    model = RRDBNet(upscale_factor=8, **kwargs)

    return model


# def content_loss(feature_model_extractor_node,
#                  feature_model_normalize_mean,
#                  feature_model_normalize_std) -> ContentLoss:
#     content_loss = ContentLoss(feature_model_extractor_node,
#                                feature_model_normalize_mean,
#                                feature_model_normalize_std)

#     return content_loss