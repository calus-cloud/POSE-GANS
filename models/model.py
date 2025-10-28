# models/blocks.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    Basic convolutional block: Conv2d → (Norm) → Activation
    """
    def _init_(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 norm=nn.InstanceNorm2d, activation=nn.ReLU(inplace=True)):
        super()._init_()
        layers = []
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False))
        if norm is not None:
            layers.append(norm(out_ch))
        if activation is not None:
            layers.append(activation)
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResidualBlock(nn.Module):
    """
    A simple residual block: two conv blocks with skip connection.
    """
    def _init_(self, ch, norm=nn.InstanceNorm2d, activation=nn.ReLU(inplace=True)):
        super()._init_()
        self.conv1 = ConvBlock(ch, ch, kernel_size=3, stride=1, padding=1, norm=norm, activation=activation)
        self.conv2 = ConvBlock(ch, ch, kernel_size=3, stride=1, padding=1, norm=norm, activation=None)
        self.act = activation

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return self.act(out + x)


class DownsampleBlock(nn.Module):
    """
    Downsampling by stride-2 conv (or optional choice).
    """
    def _init_(self, in_ch, out_ch, norm=nn.InstanceNorm2d, activation=nn.ReLU(inplace=True)):
        super()._init_()
        self.conv = ConvBlock(in_ch, out_ch, kernel_size=4, stride=2, padding=1, norm=norm, activation=activation)

    def forward(self, x):
        return self.conv(x)


class UpsampleBlock(nn.Module):
    """
    Upsampling by nearest + conv, or transposed conv.
    """
    def _init_(self, in_ch, out_ch, norm=nn.InstanceNorm2d, activation=nn.ReLU(inplace=True), use_transpose=False):
        super()._init_()
        if use_transpose:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)
            self.norm = norm(out_ch) if norm is not None else None
            self.activation = activation
        else:
            # scale up then conv
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
            )
            self.norm = norm(out_ch) if norm is not None else None
            self.activation = activation

    def forward(self, x):
        x = self.up(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

# models/warping.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import helpers (to be implemented in warp_utils)
from utils.warp_utils import warp_feature

class DeformableSkipConnection(nn.Module):
    """
    A skip connection that warps the encoder feature map to align with target pose, then merges into decoder.
    """
    def _init_(self, feat_ch, pose_ch, offset_hidden=64):
        """
        feat_ch: number of feature map channels from encoder
        pose_ch: number of pose map channels (input pose representation)
        offset_hidden: hidden channels for offset prediction
        """
        super()._init_()
        # A small network to predict offsets given encoder features + pose maps
        # You may choose more layers / vary architecture
        self.offset_net = nn.Sequential(
            nn.Conv2d(feat_ch + pose_ch * 2, offset_hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(offset_hidden, 2, kernel_size=3, padding=1)
            # output: (batch, 2, H, W) — flow offsets in x,y directions
        )

    def forward(self, feat_src, src_pose, tgt_pose):
        B, C, Hf, Wf = feat_src.size()
        # resize pose maps
        src_pose_resized = F.interpolate(src_pose, size=(Hf, Wf), mode='bilinear', align_corners=False)
        tgt_pose_resized = F.interpolate(tgt_pose, size=(Hf, Wf), mode='bilinear', align_corners=False)
        x = torch.cat([feat_src, src_pose_resized, tgt_pose_resized], dim=1)
        offsets = self.offset_net(x)
        warped = warp_feature(feat_src, offsets)
        return warped, offsets


class MultiScaleDeformSkip(nn.Module):
    """
    If you want to predict offsets at multiple scales / pyramid levels,
    you may stack several DeformableSkipConnection modules.
    """
    def _init_(self, feat_channels_list, pose_ch, offset_hidden=64):
        """
        feat_channels_list: list of feat channels at different scales (from coarse to fine)
        e.g. [512, 256, 128, 64]
        """
        super()._init_()
        self.deform_skips = nn.ModuleList([
            DeformableSkipConnection(c, pose_ch, offset_hidden=offset_hidden)
            for c in feat_channels_list
        ])

    def forward(self, feat_list, src_pose, tgt_pose):
        """
        feat_list: list of encoder features from multiple levels (coarse->fine)
        Returns: list of warped features aligned with target pose
        """
        warped_feats = []
        offsets_all = []
        for feat, ds in zip(feat_list, self.deform_skips):
            w, off = ds(feat, src_pose, tgt_pose)
            warped_feats.append(w)
            offsets_all.append(off)
        return warped_feats, offsets_all

# models/generator.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import DownsampleBlock, ResidualBlock, UpsampleBlock
from .warping import MultiScaleDeformSkip

class DeformableGenerator(nn.Module):
    def _init_(self, in_ch=3, pose_ch=18, ngf=64):
        super()._init_()

        # ---------- Encoder ----------
        self.enc1 = DownsampleBlock(in_ch, ngf, norm=None)      # -> 64x64
        self.enc2 = DownsampleBlock(ngf, ngf * 2)                # -> 32x32
        self.enc3 = DownsampleBlock(ngf * 2, ngf * 4)            # -> 16x16
        self.enc4 = DownsampleBlock(ngf * 4, ngf * 8)            # -> 8x8
        self.res_blocks = nn.Sequential(
            ResidualBlock(ngf * 8),
            ResidualBlock(ngf * 8)
        )

        # ---------- Deformable skip connections ----------
        self.deform_skips = MultiScaleDeformSkip(
            feat_channels_list=[ngf, ngf * 2, ngf * 4, ngf * 8],
            pose_ch=pose_ch
        )

        # Decoder
        self.ups = nn.ModuleList([
            UpsampleBlock(512, 256),  # stage 0: input from res_blocks
            UpsampleBlock(512, 128),  # stage 1: input from merge_conv[0] output
            UpsampleBlock(256, 64),   # stage 2: input from merge_conv[1] output
            UpsampleBlock(128, 32)    # stage 3: input from merge_conv[2] output
        ])

        self.merge_convs = nn.ModuleList([
            nn.Conv2d(256 + 512, 512, kernel_size=1),  # stage 0
            nn.Conv2d(128 + 256, 256, kernel_size=1),  # stage 1
            nn.Conv2d(64 + 128, 128, kernel_size=1),   # stage 2
            nn.Conv2d(32 + 64, 64, kernel_size=1)      # stage 3
        ])

        # ---------- Final output conv ----------
        self.final_conv = nn.Sequential(
            nn.Conv2d(ngf, in_ch, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, src_img, src_pose, tgt_pose):
        # ----- Encoder -----
        e1 = self.enc1(src_img)   # [B, 64, H/2, W/2]
        e2 = self.enc2(e1)        # [B, 128, H/4, W/4]
        e3 = self.enc3(e2)        # [B, 256, H/8, W/8]
        e4 = self.enc4(e3)        # [B, 512, H/16, W/16]
        x = self.res_blocks(e4)

        # ----- Deformable skips -----
        feats = [e1, e2, e3, e4]
        warped_feats, offsets = self.deform_skips(feats, src_pose, tgt_pose)

        # ----- Decoder -----
        for i, up in enumerate(self.ups):
            x = up(x)

            # get corresponding warped skip
            skip_feat = warped_feats[-(i + 1)]

            # Resize skip to match decoder feature spatially
            if x.shape[2:] != skip_feat.shape[2:]:
                skip_feat = F.interpolate(skip_feat, size=x.shape[2:], mode='bilinear', align_corners=False)

            # Concatenate and reduce with merge conv
            x = torch.cat([x, skip_feat], dim=1)
            x = self.merge_convs[i](x)

        out = self.final_conv(x)
        return out, offsets

# models/discriminator.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator that inputs image + pose, and outputs patch realism map.
    """
    def _init_(self, in_ch_img=3, pose_ch=18, base_channel=64, n_layers=4):
        super()._init_()
        # input channels = img + pose
        ch = base_channel
        layers = []
        layers.append(nn.Conv2d(in_ch_img + pose_ch, ch, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        curr_ch = ch
        for n in range(1, n_layers):
            next_ch = min(curr_ch * 2, base_channel * 8)
            layers.append(nn.Conv2d(curr_ch, next_ch, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(next_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            curr_ch = next_ch

        # After downsampling, one more conv with stride=1
        layers.append(nn.Conv2d(curr_ch, curr_ch * 2, kernel_size=4, stride=1, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(curr_ch * 2))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # final output layer
        layers.append(nn.Conv2d(curr_ch * 2, 1, kernel_size=4, stride=1, padding=1))
        # No sigmoid — we'll use LSGAN or hinge loss directly on output

        self.main = nn.Sequential(*layers)

    def forward(self, img, pose):
        """
        img: (B, 3, H, W)
        pose: (B, P, H, W)
        returns: patch score map (B, 1, H', W')
        """
        x = torch.cat([img, pose], dim=1)
        return self.main(x)
