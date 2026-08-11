import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


def depth2normal(depth:torch.Tensor, focal:float=None):
    if depth.dim() == 2:
        depth = depth[None, None]
    elif depth.dim() == 3:
        depth = depth.squeeze()[None, None]
    if focal is None:
        focal = depth.shape[-1] / 2 / np.tan(torch.pi/6)
    depth = torch.cat([depth[:, :, :1], depth, depth[:, :, -1:]], dim=2)
    depth = torch.cat([depth[..., :1], depth, depth[..., -1:]], dim=3)
    kernel = torch.tensor([[[  0,   0,  0],
                            [-.5,   0, .5],
                            [  0,   0,  0]],
                           [[  0, -.5,  0],
                            [  0,   0,  0],
                            [  0,  .5,  0]]], device=depth.device, dtype=depth.dtype)[:, None]
    normal = torch.nn.functional.conv2d(depth, kernel, padding='valid')[0].permute(1, 2, 0)
    normal = normal / (depth[0, 0, 1:-1, 1:-1, None] + 1e-10) * focal
    normal = torch.cat([normal, torch.ones_like(normal[..., :1])], dim=-1)
    normal = normal / normal.norm(dim=-1, keepdim=True)
    return normal.permute(2, 0, 1)


def vis_depth(depth, path):
    depth = depth.squeeze()
    if type(depth) == torch.Tensor:
        depth = depth.cpu().numpy()
    plt.imsave(path, depth, cmap='Spectral')