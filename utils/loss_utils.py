#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def l1_loss(network_output, gt, mask=None):
    if mask is not None:
        return (torch.abs((network_output - gt)) * mask).sum() / (mask.sum() + 1e-6)
    else:
        return torch.abs((network_output - gt)).mean()


def kl_divergence(rho, rho_hat):
    rho_hat = torch.mean(torch.sigmoid(rho_hat), 0)
    rho = torch.tensor([rho] * len(rho_hat)).cuda()
    return torch.mean(
        rho * torch.log(rho / (rho_hat + 1e-5)) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat + 1e-5)))


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, mask=None):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, mask)


def _ssim(img1, img2, window, window_size, channel, mask=None):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if mask is not None:
        ssim_map = (ssim_map * mask).sum() / (3 * mask.sum() + 1e-6)
    else:
        ssim_map = ssim_map.mean()
    return ssim_map


def contrastive_loss(
        features: torch.Tensor, 
        instance_labels: torch.Tensor, 
        temperature: float, 
    ):
    '''
    Args:
        features: (N, D), features of the sampled pixels
        instance_labels: (N, ) or (N, L), integer labels or multi-hot bool labels
        temperature: temperature for the softmax
        weight: (N, ), weight for each sample in the final loss
    '''
    bsize = features.size(0) # N
    sim_masks = instance_labels.view(-1, 1).repeat(1, bsize).eq_(instance_labels.clone()) # (N, N)
    sim_masks = sim_masks.fill_diagonal_(0, wrap=False) # (N, N)
    # compute similarity matrix based on Euclidean distance
    distance_sq = torch.cdist(features, features, p=2).pow(2) # (N, N)
    # temperature = torch.ones_like(distance_sq) * 10 # (N, N)
    # temperature = torch.where(sim_masks==1, temperature, torch.ones_like(temperature)) # (N, N)
    # prob = F.log_softmax(-distance_sq/temperature, -1) # (N, N)

    # # features = F.normalize(features, p=2, dim=-1)
    # # similarity = torch.matmul(features, features.t()) # (N, N)
    # # prob = F.log_softmax(similarity/0.07, -1) # (N, N)

    # prob = torch.mul(prob, sim_masks).sum(dim=-1) / (sim_masks.sum(-1) + 1e-6) # (N,), numer
    # prob_masked = torch.masked_select(prob, prob.ne(0)) # (N,)
    # loss = - prob_masked.mean() # (1,)

    # temperature = 1 for positive pairs and temperature (100) for negative pairs
    temperature = torch.ones_like(distance_sq) * temperature # (N, N)
    temperature = torch.where(sim_masks==1, temperature, torch.ones_like(temperature)) # (N, N)
    similarity_kernel = torch.exp(-distance_sq/temperature) # (N, N)
    prob_before_norm = torch.exp(similarity_kernel) # (N, N)

    Z = prob_before_norm.sum(dim=-1) # (N,), denom
    p = torch.mul(prob_before_norm, sim_masks).sum(dim=-1) # (N,), numer
    prob = torch.div(p, Z) # (N,)
    prob_masked = torch.masked_select(prob, prob.ne(0)) # (N,)
    loss = - prob_masked.log().sum()/bsize # (1,)

    return loss


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, cfg):
        super(ContrastiveLoss, self).__init__()
        self.cfg = cfg
        self.n_samples = 4096
        self.temperature = 100

    def forward(self, gt_masks, render_features, weight_image=None):
        # Extract the last a few channels as the instance feature field
        render_features = render_features[-16:, :, :] # (D, H, W)
        render_features = render_features.permute(1, 2, 0).contiguous() # (H, W, D)
        if weight_image is not None:
            # Input weight image should be (1, H, W)
            assert weight_image.shape[0] == 1 and weight_image.dim() == 3, f"weight_image.shape: {weight_image.shape}"
            # Resize the weight image to the same size as the render features
            weight_image = F.interpolate(
                weight_image.unsqueeze(0), 
                size=render_features.shape[:2],
                mode='bilinear', 
                align_corners=False
            ).squeeze(0) # (1, H, W)

        return self.contr_loss(gt_masks, render_features, weight_image)

    def contr_loss(self, gt_mask_image, render_features, weight_image=None):
        valid_mask1 = (gt_mask_image > 0) 
        valid_indices1 = torch.nonzero(valid_mask1, as_tuple=False).to(render_features.device) # (N, 2)
        valid_indices_flatten1 = valid_indices1[:, 0] * valid_mask1.shape[1] + valid_indices1[:, 1] # (N, )
        sample_idx1 = torch.randperm(valid_indices_flatten1.shape[0])[:self.n_samples] # (N_sample, )
        sample_idx1 = valid_indices_flatten1[sample_idx1] # (N_sample, )
        # valid_mask2 = (gt_mask_image < gt_mask_image.max())
        # valid_indices2 = torch.nonzero(valid_mask2, as_tuple=False).to(render_features.device) # (N, 2)
        # valid_indices_flatten2 = valid_indices2[:, 0] * valid_mask2.shape[1] + valid_indices2[:, 1] # (N, )
        # sample_idx2 = torch.randperm(valid_indices_flatten2.shape[0])[:self.n_samples // 2]
        # sample_idx2 = valid_indices_flatten2[sample_idx2] # (N_sample, )
        # sample_idx = torch.cat([sample_idx1, sample_idx2], dim=0)
        sample_idx = sample_idx1

        render_features_flatten = render_features.reshape(-1, render_features.shape[-1]) # (H * W, D)

        mask_flatten = gt_mask_image.reshape(-1) # (H * W)

        # If there are too few pixels sampled, skip contrastive loss for now
        if len(sample_idx) < 2 ** 3:
            return torch.tensor(0.0)

        sampled_mask_flatten = mask_flatten[sample_idx] # (N, ) or (N, L)
        feats = render_features_flatten[sample_idx]
        # Contrastive loss on the rendered features
        loss_contr = contrastive_loss(
            feats,
            sampled_mask_flatten,
            temperature = self.temperature,
        )

        return loss_contr
    