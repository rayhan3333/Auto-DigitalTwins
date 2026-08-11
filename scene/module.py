import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

import torch
import torch.nn as nn
import tinycudann as tcnn


def gumbel_softmax(logits, tau=1., hard=True, dim=-1, is_training=False):
    # modified from torch.nn.functional.gumbel_softmax
    if is_training:
        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )  # ~Gumbel(0,1)
        logits = (logits + gumbels)
    logits = logits / tau
    y_soft = logits.softmax(dim)
    index = y_soft.max(dim, keepdim=True)[1]
    if hard:
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        # Straight through estimator.
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft
    return y, index.squeeze(dim)


def get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


class ProgressiveBandHashGrid(nn.Module):
    def __init__(self, in_channels, start_level=6, n_levels=12, start_step=1000, update_steps=1000, dtype=torch.float32):
        super().__init__()

        encoding_config = {
            "otype": "Grid",
            "type": "Hash",
            "n_levels": n_levels,  # 16 for complex motions
            "n_features_per_level": 2,
            "log2_hashmap_size": 19,
            "base_resolution": 16,
            "per_level_scale": 2.0,
            "interpolation": "Linear",
            "start_level": start_level,
            "start_step": start_step,
            "update_steps": update_steps,
        }

        self.n_input_dims = in_channels
        with torch.cuda.device(get_rank()):
            self.encoding = tcnn.Encoding(in_channels, encoding_config, dtype=dtype)
        self.n_output_dims = self.encoding.n_output_dims
        self.n_level = encoding_config["n_levels"]
        self.n_features_per_level = encoding_config["n_features_per_level"]
        self.start_level, self.start_step, self.update_steps = (
            encoding_config["start_level"],
            encoding_config["start_step"],
            encoding_config["update_steps"],
        )
        self.current_level = self.start_level
        self.mask = torch.zeros(
            self.n_level * self.n_features_per_level,
            dtype=torch.float32,
            device=get_rank(),
        )
        self.mask[: self.current_level * self.n_features_per_level] = 1.0

    def forward(self, x):
        enc = self.encoding(x)
        enc = enc * self.mask + enc.detach() * (1 - self.mask)
        return enc

    def update_step(self, global_step):
        current_level = min(
            self.start_level
            + max(global_step - self.start_step, 0) // self.update_steps,
            self.n_level,
        )
        if current_level > self.current_level:
            # print(f"Update current level of HashGrid to {current_level}")
            self.current_level = current_level
            self.mask[: self.current_level * self.n_features_per_level] = 1.0


def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
    

class TimePrediction(nn.Module):
    def __init__(self, out_ch, hid_dim, num_joints, emb_freqs=10):
        super().__init__()
        self.encoder, time_dim = get_embedder(emb_freqs, 1)
        out_layer_list = []
        # self.encoder = GaborNet(in_size=1,
        #                             hidden_size=256,
        #                             n_layers=2,
        #                             alpha=4.5,
        #                             out_size=hid_dim)
        for _ in range(num_joints):
            out_layer = nn.Sequential(
                nn.Linear(time_dim, hid_dim),
                nn.ReLU(),
                nn.Linear(hid_dim, hid_dim),
                nn.ReLU(),
                nn.Linear(hid_dim, out_ch)
            )
            nn.init.zeros_(out_layer[-1].weight)
            nn.init.zeros_(out_layer[-1].bias)
            out_layer_list.append(out_layer)
        self.out_layer_list = nn.ModuleList(out_layer_list)

    
    def forward(self, t):
        feat = self.encoder(t)
        return torch.stack([self.out_layer_list[joint_idx](feat) for joint_idx in range(len(self.out_layer_list))])
    
    # def forward(self, t, joint_idx):
    #     return self.out_layer_list[joint_idx](self.encoder(t))

    def reg_loss(self):
        t = torch.linspace(0, 1, 60).to(self.out_layer_list[0][0].weight.device)[:, None]
        feat = self.encoder(t)
        thetas = torch.stack([self.out_layer_list[joint_idx](feat) for joint_idx in range(len(self.out_layer_list))], 1) # [n, num_joints, 1]
        velocities = torch.diff(thetas, dim=0) / (t[1] - t[0]) # [n-1, num_joints, 1]
        accs = torch.diff(velocities, dim=0) / (t[2] - t[0]) # [n-2, num_joints, 1]
        reg_loss = torch.mean(torch.norm(velocities, dim=-1)) + torch.mean(torch.norm(accs, dim=-1))
        return reg_loss * 0.001
    

class TimeInterpolation(nn.Module):
    def __init__(self, output_dim, num_joints, num_control_points=240, interpolation='linear'):
        super().__init__()
        self.num_joints = num_joints
        self.output_dim = output_dim
        self.num_control_points = num_control_points
        self.interpolation = interpolation
        self.control_points = nn.Parameter(
            torch.zeros(num_control_points, num_joints, output_dim) * 0.01
        )

    def forward(self, t):
        float_idx = t * (self.num_control_points)
        idx_left = torch.floor(float_idx).long()
        idx_right = torch.ceil(float_idx).long()
        idx_left = torch.clamp(idx_left, 0, self.num_control_points - 1)
        idx_right = torch.clamp(idx_right, 0, self.num_control_points - 1)

        points_left = self.control_points[idx_left].squeeze(1) # [T, num_joints, output_dim]
        points_right = self.control_points[idx_right].squeeze(1) # [T, num_joints, output_dim]
        alpha = (float_idx - idx_left.float())[:, None]
        interpolated = (1.0 - alpha) * points_left + alpha * points_right # [T, num_joints, output_dim]
        return interpolated.transpose(0, 1) # [num_joints, T, output_dim]
    
    def reg_loss(self):
        velocities = torch.diff(self.control_points, dim=0)
        accs = torch.diff(velocities, dim=0)
        reg_loss = torch.mean(torch.norm(velocities, dim=-1)) + torch.mean(torch.norm(accs, dim=-1))
        return reg_loss * 0.01


class PolynomialEmbedding(nn.Module):
    def __init__(self, output_dim, num_joints, degree=6):
        super().__init__()
        self.output_dim = output_dim
        self.num_joints = num_joints
        self.degree = degree
        
        self.coefficients = nn.Parameter(
            torch.randn(num_joints, output_dim, degree) * 0.01
        )
        
    def forward(self, t):
        outputs = []
        t_powers = (t[:, None]) ** torch.arange(self.degree).unsqueeze(0).to(t.device)
        for joint_idx in range(self.num_joints):
            output = torch.sum(self.coefficients[joint_idx] * t_powers, dim=-1)
            outputs.append(output)
        return torch.stack(outputs, dim=1)