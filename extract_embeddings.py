import argparse, numpy as np, torch
from torch_geometric.data import DataLoader
from datasets import ArgoverseV1Dataset
from models.hivt import HiVT
import os
from tqdm import tqdm

@torch.no_grad()
def run(model, loader, device, out_path):
    buf = {k: [] for k in ['local', 'global_mean', 'global_modes', 'ade', 'fde',
                           'fde_modes', 'pi', 'seq_id', 'city']}
    i = 0
    for data in loader:
        i +=1
        print(i,len(loader))
        data = data.to(device)
        y_hat, pi, local_embed, global_embed = model(data)#, return_embed=True)
        idx = data['agent_index']                          # focal agent per scene
        B = idx.numel()
        ar = torch.arange(B, device=device)

        # same metric definition as validation_step (best mode by FDE)
        y_hat_a = y_hat[:, idx, :, :2]                     # [F, B, T, 2]
        y_a = data.y[idx]                                  # [B, T, 2] (already rotated in forward)
        disp = torch.norm(y_hat_a - y_a.unsqueeze(0), dim=-1)  # [F, B, T]
        fde_k = disp[:, :, -1]                             # [F, B]
        best = fde_k.argmin(0)
        ade = disp.mean(-1)[best, ar]
        fde = fde_k[best, ar]

        g = global_embed[:, idx]                           # [F, B, D]  (check shape!)
        buf['local'].append(local_embed[idx].half().cpu())
        buf['global_mean'].append(g.mean(0).half().cpu())
        buf['global_modes'].append(g.permute(1, 0, 2).half().cpu())   # [B, F, D]
        buf['ade'].append(ade.cpu()); buf['fde'].append(fde.cpu())
        buf['fde_modes'].append(fde_k.t().cpu())           # [B, F]
        buf['pi'].append(torch.softmax(pi[idx], -1).cpu())
        buf['seq_id'].append(torch.as_tensor(data['seq_id']).view(-1).cpu())
        buf['city'] += list(data['city']) if 'city' in data else []

    out = {k: (torch.cat(v).numpy() if k != 'city' else np.array(v)) for k, v in buf.items() if len(v)}
    file_path = out_path
    np.savez(file_path, **out)
    print(out_path, {k: v.shape for k, v in out.items()},
                  'minADE', out['ade'].mean(), 'minFDE', out['fde'].mean())

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--split', default='val')      # 'train' or 'val'
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--out', required=True)
    a = p.parse_args()

    # check the output location before doing any work
    os.makedirs(a.out, exist_ok=True)
    file_path = os.path.join(a.out, f'{a.split}_embed.npz')
    np.savez(file_path, test=np.zeros(1))
    print('Output file will be:', os.path.abspath(file_path))

    device = 'cuda'
    model = HiVT.load_from_checkpoint(a.ckpt).to(device).eval()
    model.return_embed = True
    ds = ArgoverseV1Dataset(root=a.root, split=a.split, local_radius=model.hparams.local_radius)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=8)
    torch.manual_seed(0)
    run(model, loader, device, file_path)