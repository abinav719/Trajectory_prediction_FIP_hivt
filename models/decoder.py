# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import init_weights
import math



class GRUDecoder(nn.Module):

    def __init__(self,
                 local_channels: int,
                 global_channels: int,
                 future_steps: int,
                 num_modes: int,
                 uncertain: bool = True,
                 min_scale: float = 1e-3) -> None:
        super(GRUDecoder, self).__init__()
        self.input_size = global_channels
        self.hidden_size = local_channels
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.uncertain = uncertain
        self.min_scale = min_scale

        self.gru = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=1,
                          bias=True,
                          batch_first=False,
                          dropout=0,
                          bidirectional=False)
        self.loc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 2))
        if uncertain:
            self.scale = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, 2))
        self.pi = nn.Sequential(
            nn.Linear(self.hidden_size + self.input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1))
        self.apply(init_weights)

    def forward(self,
                local_embed: torch.Tensor,
                global_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pi = self.pi(torch.cat((local_embed.expand(self.num_modes, *local_embed.shape),
                                global_embed), dim=-1)).squeeze(-1).t()
        global_embed = global_embed.reshape(-1, self.input_size)  # [F x N, D]
        global_embed = global_embed.expand(self.future_steps, *global_embed.shape)  # [H, F x N, D]
        local_embed = local_embed.repeat(self.num_modes, 1).unsqueeze(0)  # [1, F x N, D]
        out, _ = self.gru(global_embed, local_embed)
        out = out.transpose(0, 1)  # [F x N, H, D]
        loc = self.loc(out)  # [F x N, H, 2]
        if self.uncertain:
            scale = F.elu_(self.scale(out), alpha=1.0) + 1.0 + self.min_scale  # [F x N, H, 2]
            return torch.cat((loc, scale),
                             dim=-1).view(self.num_modes, -1, self.future_steps, 4), pi  # [F, N, H, 4], [N, F]
        else:
            return loc.view(self.num_modes, -1, self.future_steps, 2), pi  # [F, N, H, 2], [N, F]


class MLPDecoder(nn.Module):

    def __init__(self,
                 local_channels: int,
                 global_channels: int,
                 future_steps: int,
                 num_modes: int,
                 uncertain: bool = True,
                 min_scale: float = 1e-3) -> None:
        super(MLPDecoder, self).__init__()
        self.input_size = global_channels #One global channel (Vector) per agent 128
        self.hidden_size = local_channels #One local channel (Vector) per agent 128
        self.future_steps = future_steps #30 steps preidciton
        self.num_modes = num_modes #6 modes
        self.uncertain = uncertain
        self.min_scale = min_scale
        #This is aggregating the global and local embedding for prediction
        self.aggr_embed = nn.Sequential(
            nn.Linear(self.input_size + self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True))
        #Predictions
        self.loc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.future_steps * 2))
        if uncertain:
            self.scale = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.future_steps * 2))
        #Self.pi gives mode probability
        self.pi = nn.Sequential(
            nn.Linear(self.hidden_size + self.input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1))
        self.apply(init_weights)

    def forward(self,
                local_embed: torch.Tensor,
                global_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        #global embedding is like embed_dimension x number of modes.

        #after self.pi we would get mode probability for each mode. It requires local and a different 128 global dim to make it.
        # N here denotes batch dimension which is number of agents. F is modes and H is future timesteps for prediction
        pi = self.pi(torch.cat((local_embed.expand(self.num_modes, *local_embed.shape),
                                global_embed), dim=-1)).squeeze(-1).t()
        out = self.aggr_embed(torch.cat((global_embed, local_embed.expand(self.num_modes, *local_embed.shape)), dim=-1))
        loc = self.loc(out).view(self.num_modes, -1, self.future_steps, 2)  # [F, N, H, 2]
        if self.uncertain:
            scale = F.elu_(self.scale(out), alpha=1.0).view(self.num_modes, -1, self.future_steps, 2) + 1.0
            scale = scale + self.min_scale  # [F, N, H, 2]
            return torch.cat((loc, scale), dim=-1), pi  # [F, N, H, 4], [N, F]
        else:
            return loc, pi  # [F, N, H, 2], [N, F]


class TransformerDecoder(nn.Module):
# The main idea is to replace the MLP decoder with transformer based attention.
# 1) Attention based mode querys which interact with each other.
# 2) Time based self attention.
    def __init__(self,
                 local_channels: int,
                 global_channels: int,
                 future_steps: int,
                 num_modes: int,
                 uncertain: bool = True,
                 min_scale: float = 1e-3,
                 num_heads: int = 8,
                 mode_layers: int = 1,
                 time_layers: int = 2,
                 dropout: float = 0.1) -> None:

        super(TransformerDecoder, self).__init__()
        self.input_size = global_channels #One global channel (Vector) per agent 128
        self.hidden_size = local_channels #One local channel (Vector) per agent 128
        self.future_steps = future_steps #30 steps preidciton
        self.num_modes = num_modes #6 modes
        self.uncertain = uncertain
        self.min_scale = min_scale

        #This is aggregating the global and local embedding for prediction
        self.aggr_embed = nn.Sequential(
            nn.Linear(self.input_size + self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True))

        #Self attention across modes
        mode_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size, nhead=num_heads,
            dim_feedforward=self.hidden_size * 4, dropout=dropout,
            activation='relu')
        self.mode_attn = nn.TransformerEncoder(mode_layer, num_layers=mode_layers)

        self.pi = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1))

        #Time-based attention
        self.time_queries = nn.Parameter(torch.empty(self.future_steps, self.hidden_size))
        nn.init.normal_(self.time_queries, mean=0.0, std=0.02)

        time_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size, nhead=num_heads,
            dim_feedforward=self.hidden_size * 4, dropout=dropout,
            activation='relu')
        self.time_attn = nn.TransformerEncoder(time_layer, num_layers=time_layers)

        # Per-timestep readout heads.
        self.loc = nn.Sequential(
            nn.LayerNorm(self.hidden_size), nn.Linear(self.hidden_size, 2))
        if uncertain:
            self.scale = nn.Sequential(
                nn.LayerNorm(self.hidden_size), nn.Linear(self.hidden_size, 2))



    def forward(self,
                local_embed: torch.Tensor,
                global_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        #global embedding is like embed_dimension x number of modes.

        #after self.pi we would get mode probability for each mode. It requires local and a different 128 global dim to make it.
        # N here denotes batch dimension which is number of agents. F is modes and H is future timesteps for prediction

        F_, N = self.num_modes, local_embed.shape[0]
        tokens = self.aggr_embed(
            torch.cat((global_embed,
                       local_embed.expand(self.num_modes, *local_embed.shape)),
                      dim=-1))  # [F, N, D]

        # ---- Stage 1: modes attend to each other, per agent ----
        # Put agents in the batch dim, modes as the sequence: [N, F, D].
        mode_seq = self.mode_attn(tokens)  # [F, N, D]
        pi = self.pi(mode_seq).squeeze(-1).t()  # [F, N] -> [N, F]


        # ---- Stage 2: build a length-H sequence per (mode, agent), attend over time ----
        # context: [F, N, D] -> [1, F*N, D];  time_queries: [H, D] -> [H, 1, D]
        context = mode_seq.reshape(1, F_ * N, self.hidden_size)  # [1, F*N, D]
        time_tokens = context + self.time_queries.unsqueeze(1)  # [H, F*N, D]
        seq = self.time_attn(time_tokens)  # [H, F*N, D]

        # [H, F*N, 2] -> [F*N, H, 2] -> [F, N, H, 2]
        loc = self.loc(seq).transpose(0, 1).reshape(F_, N, self.future_steps, 2)

        if self.uncertain:
            scale = F.elu_(self.scale(seq), alpha=1.0).transpose(0, 1).reshape(
                F_, N, self.future_steps, 2) + 1.0
            scale = scale + self.min_scale  # [F, N, H, 2]
            return torch.cat((loc, scale), dim=-1), pi  # [F, N, H, 4], [N, F]
        else:
            return loc, pi  # [F, N, H, 2], [N, F]


class KinematicTransformerDecoder(nn.Module):
    def __init__(self, local_channels, global_channels, future_steps, num_modes,
                 uncertain=True, min_scale=1e-3, num_heads=8, mode_layers=1,
                 time_layers=2, dropout=0.1,
                 dt=0.1, max_accel=8.0, max_yaw_rate=1.0):

        super().__init__()
        self.input_size  = global_channels
        self.hidden_size = local_channels
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.uncertain = uncertain
        self.min_scale = min_scale
        self.dt = dt
        self.max_accel = max_accel
        self.max_yaw_rate = max_yaw_rate

        self.aggr_embed = nn.Sequential(nn.Linear(self.input_size + self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size), nn.ReLU(inplace=True))

        mode_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size, nhead=num_heads,
            dim_feedforward=self.hidden_size * 4, dropout=dropout, activation='relu')
        self.mode_attn = nn.TransformerEncoder(mode_layer, num_layers=mode_layers)

        self.pi = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1))

        self.time_queries = nn.Parameter(torch.empty(self.future_steps, self.hidden_size))
        nn.init.normal_(self.time_queries, mean=0.0, std=0.02)
        time_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size, nhead=num_heads,
            dim_feedforward=self.hidden_size * 4, dropout=dropout, activation='relu')
        self.time_attn = nn.TransformerEncoder(time_layer, num_layers=time_layers)


        # per-timestep controls: acceleration a_t and yaw-rate omega_t
        self.control = nn.Sequential(
            nn.LayerNorm(self.hidden_size), nn.Linear(self.hidden_size, 2))

        if uncertain:
            self.scale = nn.Sequential(
                nn.LayerNorm(self.hidden_size), nn.Linear(self.hidden_size, 2))

    def _rollout(self, v0, theta0, a, omega):
        # v0, theta0: [F, N]   a, omega: [F, N, H]  ->  loc: [F, N, H, 2]
        dt = self.dt
        theta = theta0.unsqueeze(-1) + torch.cumsum(omega * dt, dim=-1)  # [F,N,H]
        v     = v0.unsqueeze(-1)     + torch.cumsum(a     * dt, dim=-1)   # [F,N,H]
        x = torch.cumsum(v * torch.cos(theta) * dt, dim=-1)
        y = torch.cumsum(v * torch.sin(theta) * dt, dim=-1)
        return torch.stack((x, y), dim=-1)                               # [F,N,H,2]

    def forward(self, local_embed,global_embed, vel_prior):
        F_, N = self.num_modes, local_embed.shape[0]
        tokens = self.aggr_embed(
            torch.cat((global_embed,
                       local_embed.expand(self.num_modes, *local_embed.shape)),
                      dim=-1))
        mode_seq = self.mode_attn(tokens)                                # [F, N, D]
        context = mode_seq.reshape(1, F_ * N, self.hidden_size)          # [1, F*N, D]
        time_tokens = context + self.time_queries.unsqueeze(1)           # [H, F*N, D]
        seq = self.time_attn(time_tokens)                                # [H, F*N, D]

        # initial state: per agent, broadcast across modes
        # Need to give the correct values here
        v0     = vel_prior.expand(F_, N)                 # [F, N]
        theta0 = torch.zeros(F_, N, device=local_embed.device, dtype=local_embed.dtype)                # [F, N]

        # per-step controls, bounded so they can't blow up early in training
        ctrl = self.control(seq).transpose(0, 1).reshape(
            F_, N, self.future_steps, 2)                                 # [F,N,H,2]
        a     = torch.tanh(ctrl[..., 0]) * self.max_accel                # [F,N,H]
        omega = torch.tanh(ctrl[..., 1]) * self.max_yaw_rate             # [F,N,H]

        loc = self._rollout(v0, theta0, a, omega)                        # [F,N,H,2]

        # mode probability is per mode -> read from mode features, not the time seq
        pi = self.pi(mode_seq).squeeze(-1).t()                           # [F,N] -> [N,F]

        if self.uncertain:
            scale = F.elu_(self.scale(seq), alpha=1.0).transpose(0, 1).reshape(
                F_, N, self.future_steps, 2) + 1.0
            scale = scale + self.min_scale
            return torch.cat((loc, scale), dim=-1), pi                   # [F,N,H,4], [N,F]
        return loc, pi                                                   # [F,N,H,2], [N,F]

"""
Flow-matching transformer decoder for HiVT (TrajFlow-style).

Drop-in replacement for HiVT's `MLPDecoder`:
    MLPDecoder.forward(local_embed, global_embed)             -> (y_hat [F,N,H,4], pi [N,F])
    FlowMatchingDecoder.forward(local_embed, global_embed,
                                x_t, t)                       -> (y_hat [F,N,H,4], pi [N,F])
    FlowMatchingDecoder.sample(local_embed, global_embed)     -> (y_hat [F,N,H,4], pi [N,F])

Shapes (HiVT conventions):
    N = number of agents (nodes), F = num_modes, H = future_steps, D = embed_dim
    local_embed : [N, D_local]      output of LocalEncoder
    global_embed: [F, N, D_global]  output of GlobalInteractor (already split per mode)
    x_t         : [F, N, H, 2]      noisy trajectories in *normalised* space (y / traj_scale)
    t           : [N] or [F, N]     flow time in [0, 1]; t=0 noise, t=1 data

Query construction  : q = TrajMLP(x_t) + TimeMLP(t) + ModeEmbedding   (learnt per-mode embedding)
Per layer           : self-attention over the F mode queries of an agent
                      -> cross-attention to the agent's context tokens [local, global_1..F, (extra)]
                      -> FFN; every sub-block is adaLN-modulated by the time embedding.
Heads               : loc (+ Laplace scale) and pi, exactly like HiVT, so the existing
                      LaplaceNLLLoss / SoftTargetCrossEntropyLoss and metrics are unchanged.
"""


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding of t in [0, 1]. t: [...] -> [..., dim]."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) *
                      torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = (t.float() * 1000.0).unsqueeze(-1) * freqs
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb

class FlowMatchingDecoder(nn.Module):
    def __init__(self,
                 local_channels: int,
                 global_channels: int,
                 future_steps: int,
                 num_modes: int,
                 uncertain: bool = True,
                 min_scale: float = 1e-3,
                 num_layers: int = 3,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 traj_scale: float = 10.0) -> None:
        super(FlowMatchingDecoder, self).__init__()
        self.embed_dim = global_channels        # D = 128
        self.future_steps = future_steps        # H = 30
        self.num_modes = num_modes              # F = 6
        self.uncertain = uncertain
        self.min_scale = min_scale
        self.traj_scale = traj_scale            # metres -> ~unit scale, so traj and noise have similar size
        D, H = self.embed_dim, future_steps

        # ---------------- 1. Query = noisy traj + time + learnt mode embedding ----------------
        self.traj_embed = nn.Sequential(        # [H*2] -> [D]
            nn.Linear(H * 2, D),
            nn.LayerNorm(D),
            nn.ReLU(inplace=True),
            nn.Linear(D, D))
        self.mode_query = nn.Embedding(num_modes, D)   # one learnt query per mode
        self.query_fuse = nn.Sequential(        # outer MLP: [3D] -> [D]
            nn.Linear(3 * D, D),
            nn.LayerNorm(D),
            nn.ReLU(inplace=True),
            nn.Linear(D, D))


        self.context_embed = nn.Embedding(1 + num_modes, D)   # 0: local, 1..F: which mode

        # ---------------- 3. Transformer layers: self-attn -> cross-attn -> FFN ----------------
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=D, nhead=num_heads, dim_feedforward=4 * D,
                                       dropout=dropout)
            for _ in range(num_layers)])
        self.norm_out = nn.LayerNorm(D)

        # ---------------- 4. Output heads (same as MLPDecoder) ----------------
        self.loc = nn.Sequential(
            nn.Linear(D, D),
            nn.LayerNorm(D),
            nn.ReLU(inplace=True),
            nn.Linear(D, H * 2))
        if uncertain:
            self.scale = nn.Sequential(
                nn.Linear(D, D),
                nn.LayerNorm(D),
                nn.ReLU(inplace=True),
                nn.Linear(D, H * 2))
        self.pi = nn.Sequential(
            nn.Linear(D, D),
            nn.LayerNorm(D),
            nn.ReLU(inplace=True),
            nn.Linear(D, D),
            nn.LayerNorm(D),
            nn.ReLU(inplace=True),
            nn.Linear(D, 1))
        self.apply(init_weights)

    # ------------------------------------------------------------------------------------
    # Training: mix GT future with noise.  x_t = (1 - t) * noise + t * future
    # ------------------------------------------------------------------------------------
    def add_noise(self,
                  y: torch.Tensor,                          # [N, H, 2] GT future (agent frame, metres)
                  reg_mask: Optional[torch.Tensor] = None   # [N, H] True = valid step
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        x1 = y / self.traj_scale
        if reg_mask is not None:
            x1 = x1 * reg_mask.unsqueeze(-1)                # padded steps -> 0
        x1 = x1.unsqueeze(0).expand(self.num_modes, *x1.shape)        # [F, N, H, 2]
        x0 = torch.randn_like(x1)                                     # noise, different per mode
        t = torch.sigmoid(torch.randn(self.num_modes, y.size(0), device=y.device))  # [F, N], logit-normal
        x_t = (1.0 - t)[..., None, None] * x0 + t[..., None, None] * x1
        return x_t, t

    # ------------------------------------------------------------------------------------
    # One network pass: (noisy traj, t, context) -> clean traj prediction
    # ------------------------------------------------------------------------------------
    def forward(self,
                local_embed: torch.Tensor,     # [N, D_local]
                global_embed: torch.Tensor,    # [F, N, D]
                x_t: torch.Tensor,             # [F, N, H, 2] noisy traj (normalised)
                t: torch.Tensor                # [F, N], [N] or scalar
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        F_, N, H = self.num_modes, local_embed.size(0), self.future_steps
        if t.dim() == 0:
            t = t.expand(N)
        if t.dim() == 1:
            t = t.unsqueeze(0).expand(F_, N)                           # [F, N]

        # ---- build queries [N, F, D] ----
        x = x_t.permute(1, 0, 2, 3).reshape(N, F_, H * 2)             # [N, F, H*2]
        traj_feat = self.traj_embed(x)  # MLP(Y_t)   [N, F, D]
        pt = timestep_embedding(t.t(), self.embed_dim)  # PT_t       [N, F, D]
        pq = self.mode_query.weight.unsqueeze(0).expand(N, -1, -1)  # PQ_i       [N, F, D]
        query = self.query_fuse(torch.cat((traj_feat, pt, pq), dim=-1))


        # ---- build context [N, 1+F, D] ----
        local_tok = local_embed.unsqueeze(1)
        mode_tok = global_embed.transpose(0, 1)
        #Self.context embedding is can be removed as well
        context = torch.cat((local_tok, mode_tok), dim=1) + self.context_embed.weight.unsqueeze(0)

        # ---- self-attn over modes, cross-attn to context ----
        query = query.transpose(0, 1)                                  # [F, N, D]
        context = context.transpose(0, 1)                              # [1+F, N, D]
        pq_s = pq.transpose(0, 1)                                      # [F, N, D]
        for layer in self.layers:
            query = layer(query + pq_s, context)                       # re-inject time each layer
        query = self.norm_out(query.transpose(0, 1))                   # back to [N, F, D]

        # ---- heads -> HiVT format ----
        loc = self.loc(query).view(N, F_, H, 2).transpose(0, 1) * self.traj_scale   # [F, N, H, 2] metres
        pi = self.pi(query).squeeze(-1)                                              # [N, F]
        if self.uncertain:
            scale = F.elu(self.scale(query), alpha=1.0).view(N, F_, H, 2).transpose(0, 1) + 1.0
            scale = scale + self.min_scale                                           # [F, N, H, 2]
            return torch.cat((loc, scale), dim=-1), pi                               # [F, N, H, 4], [N, F]
        return loc, pi

    # ------------------------------------------------------------------------------------
    # Inference: Euler integration from noise (t=0) to trajectory (t=1)
    # ------------------------------------------------------------------------------------
    @torch.no_grad()
    def sample(self,
               local_embed: torch.Tensor,
               global_embed: torch.Tensor,
               num_steps: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        N = local_embed.size(0)
        x = torch.randn(self.num_modes, N, self.future_steps, 2, device=local_embed.device)
        ts = torch.linspace(0.0, 1.0, num_steps + 1, device=local_embed.device)
        for i in range(num_steps):
            out, pi = self(local_embed, global_embed, x, ts[i])
            if i == num_steps - 1:
                break
            x1_hat = out[..., :2] / self.traj_scale
            v = (x1_hat - x) / (1.0 - ts[i])          # velocity pointing to predicted clean traj
            x = x + (ts[i + 1] - ts[i]) * v
        return out, pi


