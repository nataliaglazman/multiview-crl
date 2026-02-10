import torch
import torch.nn as nn
import torch.nn.functional as F

from math import log2
from typing import Tuple

from helper import HelperModule

class ReZero(HelperModule):
    """3D ReZero residual block with learnable scaling parameter."""
    def build(self, in_channels: int, res_channels: int):
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, res_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(res_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(res_channels, in_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x) * self.alpha + x

class ResidualStack(HelperModule):
    """Stack of 3D ReZero residual blocks."""
    def build(self, in_channels: int, res_channels: int, nb_layers: int):
        self.stack = nn.Sequential(*[ReZero(in_channels, res_channels) 
                        for _ in range(nb_layers)
                    ])

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.stack(x)

class Encoder(HelperModule):
    """3D Encoder with strided convolutions for downsampling."""
    def build(self, 
            in_channels: int, hidden_channels: int, 
            res_channels: int, nb_res_layers: int,
            downscale_factor: int,
        ):
        assert log2(downscale_factor) % 1 == 0, "Downscale must be a power of 2"
        downscale_steps = int(log2(downscale_factor))
        layers = []
        c_channel, n_channel = in_channels, hidden_channels // 2
        for _ in range(downscale_steps):
            layers.append(nn.Sequential(
                nn.Conv3d(c_channel, n_channel, 4, stride=2, padding=1),
                nn.BatchNorm3d(n_channel),
                nn.ReLU(inplace=True),
            ))
            c_channel, n_channel = n_channel, hidden_channels
        layers.append(nn.Conv3d(c_channel, n_channel, 3, stride=1, padding=1))
        layers.append(nn.BatchNorm3d(n_channel))
        layers.append(ResidualStack(n_channel, res_channels, nb_res_layers))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)

class Decoder(HelperModule):
    """3D Decoder with transposed convolutions for upsampling."""
    def build(self, 
            in_channels: int, hidden_channels: int, out_channels: int,
            res_channels: int, nb_res_layers: int,
            upscale_factor: int,
        ):
        assert log2(upscale_factor) % 1 == 0, "Upscale must be a power of 2"
        upscale_steps = int(log2(upscale_factor))
        layers = [nn.Conv3d(in_channels, hidden_channels, 3, stride=1, padding=1)]
        layers.append(ResidualStack(hidden_channels, res_channels, nb_res_layers))
        c_channel, n_channel = hidden_channels, hidden_channels // 2
        for _ in range(upscale_steps):
            layers.append(nn.Sequential(
                nn.ConvTranspose3d(c_channel, n_channel, 4, stride=2, padding=1),
                nn.BatchNorm3d(n_channel),
                nn.ReLU(inplace=True),
            ))
            c_channel, n_channel = n_channel, out_channels
        layers.append(nn.Conv3d(c_channel, n_channel, 3, stride=1, padding=1))
        layers.append(nn.BatchNorm3d(n_channel))
        # layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)

"""
    Almost directly taken from https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py
    No reason to reinvent this rather complex mechanism.
    
    Adapted for 3D volumes (brain MRI).

    Essentially handles the "discrete" part of the network, and training through EMA rather than 
    third term in loss function.
"""
class CodeLayer(HelperModule):
    """3D Vector Quantization layer with EMA codebook updates."""
    def build(self, in_channels: int, embed_dim: int, nb_entries: int):
        self.conv_in = nn.Conv3d(in_channels, embed_dim, 1)

        self.dim = embed_dim
        self.n_embed = nb_entries
        self.decay = 0.99
        self.eps = 1e-5

        embed = torch.randn(embed_dim, nb_entries, dtype=torch.float32)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(nb_entries, dtype=torch.float32))
        self.register_buffer("embed_avg", embed.clone())

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, float, torch.LongTensor]:
        # x: (B, C, D, H, W) -> (B, D, H, W, embed_dim)
        x = self.conv_in(x.float()).permute(0, 2, 3, 4, 1)
        flatten = x.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*x.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            # TODO: Replace this? Or can we simply comment out?
            # dist_fn.all_reduce(embed_onehot_sum)
            # dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - x).pow(2).mean()
        quantize = x + (quantize - x).detach()

        return quantize.permute(0, 4, 1, 2, 3), diff, embed_ind

    def embed_code(self, embed_id: torch.LongTensor) -> torch.FloatTensor:
        return F.embedding(embed_id, self.embed.transpose(0, 1))

class Upscaler(HelperModule):
    """3D Upscaler for hierarchical code conditioning."""
    def build(self,
            embed_dim: int,
            scaling_rates: list[int],
        ):

        self.stages = nn.ModuleList()
        for sr in scaling_rates:
            upscale_steps = int(log2(sr))
            layers = []
            for _ in range(upscale_steps):
                layers.append(nn.ConvTranspose3d(embed_dim, embed_dim, 4, stride=2, padding=1))
                layers.append(nn.BatchNorm3d(embed_dim))
                layers.append(nn.ReLU(inplace=True))
            self.stages.append(nn.Sequential(*layers))

    def forward(self, x: torch.FloatTensor, stage: int) -> torch.FloatTensor:
        return self.stages[stage](x)

"""
    Main VQ-VAE-2 Module for 3D volumes, capable of supporting arbitrary number of levels.
    Adapted for brain MRI data.
    
    TODO: A lot of this class could do with a refactor. It works, but at what cost?
    TODO: Add discrete code decoding function
"""
class VQVAE(HelperModule):
    def build(self,
            in_channels: int                = 1,       # 1 for grayscale MRI
            hidden_channels: int            = 128,
            res_channels: int               = 32,
            nb_res_layers: int              = 2,
            nb_levels: int                  = 3,
            embed_dim: int                  = 64,
            nb_entries: int                 = 512,
            scaling_rates: list[int]        = [8, 4, 2]
        ):
        self.nb_levels = nb_levels
        assert len(scaling_rates) == nb_levels, "Number of scaling rates not equal to number of levels!"

        self.encoders = nn.ModuleList([Encoder(in_channels, hidden_channels, res_channels, nb_res_layers, scaling_rates[0])])
        for i, sr in enumerate(scaling_rates[1:]):
            self.encoders.append(Encoder(hidden_channels, hidden_channels, res_channels, nb_res_layers, sr))

        self.codebooks = nn.ModuleList()
        for i in range(nb_levels - 1):
            self.codebooks.append(CodeLayer(hidden_channels+embed_dim, embed_dim, nb_entries))
        self.codebooks.append(CodeLayer(hidden_channels, embed_dim, nb_entries))

        self.decoders = nn.ModuleList([Decoder(embed_dim*nb_levels, hidden_channels, in_channels, res_channels, nb_res_layers, scaling_rates[0])])
        for i, sr in enumerate(scaling_rates[1:]):
            self.decoders.append(Decoder(embed_dim*(nb_levels-1-i), hidden_channels, embed_dim, res_channels, nb_res_layers, sr))

        self.upscalers = nn.ModuleList()
        for i in range(nb_levels - 1):
            self.upscalers.append(Upscaler(embed_dim, scaling_rates[1:len(scaling_rates) - i][::-1]))

    def forward(self, x):
        encoder_outputs = []
        code_outputs = []
        decoder_outputs = []
        upscale_counts = []
        id_outputs = []
        diffs = []

        for enc in self.encoders:
            if len(encoder_outputs):
                encoder_outputs.append(enc(encoder_outputs[-1]))
            else:
                encoder_outputs.append(enc(x))

        for l in range(self.nb_levels-1, -1, -1):
            codebook, decoder = self.codebooks[l], self.decoders[l]

            if len(decoder_outputs): # if we have previous levels to condition on
                # Interpolate decoder output to match encoder output size if needed
                dec_out = decoder_outputs[-1]
                enc_out = encoder_outputs[l]
                if dec_out.shape[2:] != enc_out.shape[2:]:
                    dec_out = F.interpolate(dec_out, size=enc_out.shape[2:], mode='trilinear', align_corners=False)
                code_q, code_d, emb_id = codebook(torch.cat([enc_out, dec_out], axis=1))
            else:
                code_q, code_d, emb_id = codebook(encoder_outputs[l])
            diffs.append(code_d)
            id_outputs.append(emb_id)

            # Upscale previous code outputs and interpolate to match current level size
            upscaled_codes = []
            target_size = code_q.shape[2:]  # Target spatial size for this level
            for i, c in enumerate(code_outputs):
                upscaled = self.upscalers[i](c, upscale_counts[i])
                if upscaled.shape[2:] != target_size:
                    upscaled = F.interpolate(upscaled, size=target_size, mode='trilinear', align_corners=False)
                upscaled_codes.append(upscaled)
            code_outputs = upscaled_codes
            upscale_counts = [u+1 for u in upscale_counts]
            
            decoder_outputs.append(decoder(torch.cat([code_q, *code_outputs], axis=1)))

            code_outputs.append(code_q)
            upscale_counts.append(0)

        # Interpolate final output to match input size if needed
        final_output = decoder_outputs[-1]
        if final_output.shape[2:] != x.shape[2:]:
            final_output = F.interpolate(final_output, size=x.shape[2:], mode='trilinear', align_corners=False)

        return final_output, diffs, encoder_outputs, decoder_outputs, id_outputs

    def decode_codes(self, *cs):
        decoder_outputs = []
        code_outputs = []
        upscale_counts = []

        for l in range(self.nb_levels - 1, -1, -1):
            codebook, decoder = self.codebooks[l], self.decoders[l]
            code_q = codebook.embed_code(cs[l]).permute(0, 3, 1, 2)
            code_outputs = [self.upscalers[i](c, upscale_counts[i]) for i, c in enumerate(code_outputs)]
            upscale_counts = [u+1 for u in upscale_counts]
            decoder_outputs.append(decoder(torch.cat([code_q, *code_outputs], axis=1)))

            code_outputs.append(code_q)
            upscale_counts.append(0)

        return decoder_outputs[-1]

if __name__ == '__main__':
    from helper import get_parameter_count
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test with 3D brain MRI-like input
    nb_levels = 3
    # For 96x112x96 input with scaling_rates=[4, 2, 2]: 
    # Level 0: 96->24, Level 1: 24->12, Level 2: 12->6
    net = VQVAE(
        in_channels=1,           # Grayscale MRI
        hidden_channels=64,      # Reduced for memory
        res_channels=32,
        nb_res_layers=2,
        nb_levels=nb_levels, 
        embed_dim=32,            # Reduced for memory
        nb_entries=512,
        scaling_rates=[4, 2, 2]
    ).to(device)
    print(f"Number of trainable parameters: {get_parameter_count(net)}")

    # Test with 3D volume (B, C, D, H, W)
    x = torch.randn(1, 1, 96, 112, 96).to(device)
    print(f"Input shape: {x.shape}")
    
    recon, diffs, enc_out, dec_out, id_out = net(x)
    print(f"\nReconstruction shape: {recon.shape}")
    print(f"\nEncoder outputs:")
    print('\n'.join(f"  Level {i}: {y.shape}" for i, y in enumerate(enc_out)))
    print(f"\nDecoder outputs:")
    print('\n'.join(f"  Level {i}: {y.shape}" for i, y in enumerate(dec_out)))
    print(f"\nVQ commitment losses (diffs):")
    print('\n'.join(f"  Level {i}: {y:.6f}" for i, y in enumerate(diffs)))
    print(f"\nCode indices shapes:")
    print('\n'.join(f"  Level {i}: {y.shape}" for i, y in enumerate(id_out)))
