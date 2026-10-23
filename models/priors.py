import numpy as np
import torch

class Priors ():
    def __init__(self):
        self.W = 5
        self.dt = 0.1
        self.tau = (np.arange(self.W) - (self.W - 1)) * self.dt
        self.A = np.stack([np.ones(self.W), self.tau, self.tau ** 2], axis=1)  # [W, 3]
        self.Ap = np.linalg.pinv(self.A)  # the only "solve", done once
        self.VEL_K = torch.tensor(self.Ap[1], dtype=torch.float)

    def vel(self,data):
        p = data.positions[:, 15:20]  # [N, 5, 2] last 5 history samples
        k = self.VEL_K.to(p)
        v_vec = torch.einsum('w,nwc->nc', k, p)  # [N, 2]  velocity at step 19
        speed19 = v_vec.norm(dim=-1)  # [N]     what you want
        pad_win = data.padding_mask[:, 15:20]
        speed19[pad_win.any(dim=1)] = 0.0
        return speed19