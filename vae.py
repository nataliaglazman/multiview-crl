"""
Author: Duy-Phuong Dao
Email : phuongdd.1997@gmail.com or duyphuongcri@gmail.com
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math 
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, k_size, stride=1, p=1, num_groups=1):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            #nn.GroupNorm(num_groups=num_groups, num_channels=ch_in),
            # group_norm.GroupNorm3d(num_features=ch_in, num_groups=num_groups),
            # nn.ReLU(inplace=True), 
            nn.Conv3d(ch_in, ch_out, kernel_size=k_size, stride=stride, padding=p),  
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        out = self.conv(x)
        return out


class ResNet_block(nn.Module):
    "A ResNet-like block with the GroupNorm normalization providing optional bottle-neck functionality"
    def __init__(self, ch, k_size, stride=1, p=1, num_groups=16):
        super(ResNet_block, self).__init__()
        # Ensure num_groups divides ch evenly; fall back to valid divisor
        while num_groups > 1 and ch % num_groups != 0:
            num_groups //= 2
        self.conv = nn.Sequential(
            nn.Conv3d(ch, ch, kernel_size=k_size, stride=stride, padding=p), 
            nn.GroupNorm(num_groups=num_groups, num_channels=ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch, ch, kernel_size=k_size, stride=stride, padding=p),  
            nn.GroupNorm(num_groups=num_groups, num_channels=ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        out = self.conv(x) + x
        return out


class up_conv(nn.Module):
    "Reduce the number of features by 2 using Conv with kernel size 1x1x1 and double the spatial dimension using 3D trilinear upsampling"
    def __init__(self, ch_in, ch_out, k_size=1, scale=2, align_corners=False):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=k_size),
            nn.Upsample(scale_factor=scale, mode='trilinear', align_corners=align_corners),
        )
    def forward(self, x):
        return self.up(x)


class up_conv_transposed(nn.Module):
    """Upsampling using transposed convolution for sharper results."""
    def __init__(self, ch_in, ch_out, k_size=4, stride=2, padding=1):
        super(up_conv_transposed, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(ch_in, ch_out, kernel_size=k_size, stride=stride, padding=padding),
            nn.BatchNorm3d(ch_out),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x):
        return self.up(x)

class Encoder(nn.Module):
    """ Encoder module """
    def __init__(self):
        super(Encoder, self).__init__()


        self.conv1 = conv_block(ch_in=1, ch_out=32, k_size=3, num_groups=1)
        self.res_block1 = ResNet_block(ch=32, k_size=3, num_groups=16)
        self.MaxPool1 = nn.MaxPool3d(3, stride=2, padding=1)

        self.conv2 = conv_block(ch_in=32, ch_out=64, k_size=3, num_groups=16)
        self.res_block2 = ResNet_block(ch=64, k_size=3, num_groups=16)
        self.MaxPool2 = nn.MaxPool3d(3, stride=2, padding=1)

        self.conv3 = conv_block(ch_in=64, ch_out=128, k_size=3, num_groups=16)
        self.res_block3 = ResNet_block(ch=128, k_size=3, num_groups=16)
        self.MaxPool3 = nn.MaxPool3d(3, stride=2, padding=1)

        self.conv4 = conv_block(ch_in=128, ch_out=256, k_size=3, num_groups=16)
        self.res_block4 = ResNet_block(ch=256, k_size=3, num_groups=16)
        self.MaxPool4 = nn.MaxPool3d(3, stride=2, padding=1)

        self.conv5 = conv_block(ch_in=256, ch_out=512, k_size=3, num_groups=16)
        self.res_block5 = ResNet_block(ch=512, k_size=3, num_groups=16)
        self.MaxPool5 = nn.MaxPool3d(3, stride=2, padding=1)
        
        # Adaptive pooling to ensure output is (batch, 512, 1, 1, 1)
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.reset_parameters()
      
    def reset_parameters(self):
        for weight in self.parameters():
            stdv = 1.0 / math.sqrt(weight.size(0))
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.res_block1(x1)
        x1 = self.MaxPool1(x1) # torch.Size([1, 32, 26, 31, 26])
        
        x2 = self.conv2(x1)
        x2 = self.res_block2(x2)
        x2 = self.MaxPool2(x2) # torch.Size([1, 64, 8, 10, 8])

        x3 = self.conv3(x2)
        x3 = self.res_block3(x3)
        x3 = self.MaxPool3(x3) # torch.Size([1, 128, 2, 3, 2])
        
        x4 = self.conv4(x3)
        x4 = self.res_block4(x4) # torch.Size([1, 256, 2, 3, 2])
        x4 = self.MaxPool4(x4) # torch.Size([1, 256, 1, 1, 1])

        x5 = self.conv5(x4)
        x5 = self.res_block5(x5) # torch.Size([1, 512, ?, ?, ?])
        x5 = self.MaxPool5(x5)
        x5 = self.adaptive_pool(x5)  # torch.Size([1, 512, 1, 1, 1])

        return x5

class Decoder(nn.Module):
    """ Decoder Module with transposed convolutions for sharper output """
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.linear_up = nn.Linear(latent_dim, 512 * 2 * 2 * 2)  # Start from 2x2x2 spatial
        self.relu = nn.ReLU()
        
        # Use transposed convolutions for sharper upsampling
        self.upsize8 = up_conv_transposed(ch_in=512, ch_out=512, k_size=4, stride=2, padding=1)
        self.res_block8 = ResNet_block(ch=512, k_size=3, num_groups=16)
        self.upsize7 = up_conv_transposed(ch_in=512, ch_out=256, k_size=4, stride=2, padding=1)
        self.res_block7 = ResNet_block(ch=256, k_size=3, num_groups=16)
        self.upsize6 = up_conv_transposed(ch_in=256, ch_out=128, k_size=4, stride=2, padding=1)
        self.res_block6 = ResNet_block(ch=128, k_size=3, num_groups=16)
        self.upsize5 = up_conv_transposed(ch_in=128, ch_out=64, k_size=4, stride=2, padding=1)
        self.res_block5 = ResNet_block(ch=64, k_size=3, num_groups=16)
        self.upsize4 = up_conv_transposed(ch_in=64, ch_out=32, k_size=4, stride=2, padding=1)
        self.res_block4 = ResNet_block(ch=32, k_size=3, num_groups=16)
        self.upsize3 = up_conv_transposed(ch_in=32, ch_out=16, k_size=4, stride=2, padding=1)
        self.res_block3 = ResNet_block(ch=16, k_size=3, num_groups=16)        
        
        # Output refinement layers for sharper details
        self.refine = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(8, 1, kernel_size=3, padding=1),
        )

        self.final_upsample = nn.Upsample(size=(181, 217, 181), mode='trilinear', align_corners=False)

        self.reset_parameters()
      
    def reset_parameters(self):
        for weight in self.parameters():
            stdv = 1.0 / math.sqrt(weight.size(0))
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x):
        x = self.linear_up(x)
        x = self.relu(x)

        x = x.view(-1, 512, 2, 2, 2)  # Start from 2x2x2
        x = self.upsize8(x)  # -> 4x4x4
        x = self.res_block8(x)

        x = self.upsize7(x)  # -> 8x8x8
        x = self.res_block7(x)
        x = self.upsize6(x)  # -> 16x16x16
        x = self.res_block6(x)

        x = self.upsize5(x)  # -> 32x32x32
        x = self.res_block5(x)

        x = self.upsize4(x)  # -> 64x64x64
        x = self.res_block4(x)

        x = self.upsize3(x)  # -> 128x128x128
        x = self.res_block3(x)
        
        # Refine details before final upsampling
        x = self.refine(x)
        
        x = self.final_upsample(x)  # -> 181x217x181

        return x


class VAE(nn.Module):
    def __init__(self, latent_dim=256):
        super(VAE, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.latent_dim = latent_dim
        self.z_mean = nn.Linear(512, latent_dim)
        self.z_log_sigma = nn.Linear(512, latent_dim)
        self.epsilon = torch.normal(size=(1, latent_dim), mean=0, std=1.0, device=self.device)
        self.encoder = Encoder()
        self.decoder = Decoder(latent_dim)

        self.reset_parameters()
      
    def reset_parameters(self):
        for weight in self.parameters():
            stdv = 1.0 / math.sqrt(weight.size(0))
            torch.nn.init.uniform_(weight, -stdv, stdv)

    # In forward method, generate epsilon per batch:
    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        z_mean = self.z_mean(x)
        z_log_sigma = self.z_log_sigma(x)
        epsilon = torch.randn_like(z_mean)  # Generate fresh epsilon per batch
        z = z_mean + z_log_sigma.exp() * epsilon
        y = self.decoder(z)
        return y, z_mean, z_log_sigma
