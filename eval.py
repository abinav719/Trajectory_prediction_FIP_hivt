from argparse import ArgumentParser
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.data import DataLoader
from argoverse.map_representation.map_api import ArgoverseMap
from datasets import ArgoverseV1Dataset
from models.hivt import HiVT
import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import List




def to_np(t):
    return t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)

def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

@torch.no_grad()
def plot_scene(data, out, pi, avm, scene=0, rotate=True,
               map_range=70, all_modes_for_all=False, save=None):
    """
    data : HiVT batch (TemporalData / PyG Batch)
    out  : [num_modes, N_total, 30, >=2] predictions (agent-local frame, real units)
    pi   : [N_total, num_modes] (or [num_modes, N_total]) mode logits
    """
    # node range of this scene in the batch
    if hasattr(data, "ptr") and data.ptr is not None:
        n_scenes = len(data.ptr) - 1
        if scene >= n_scenes:
            print(f"Batch has only {n_scenes} scene(s); skipping scene={scene}")
            return
        s, e = int(data.ptr[scene]), int(data.ptr[scene + 1])
    else:
        if scene > 0:
            print("Batch has only 1 scene; skipping")
            return
        s, e = 0, data.num_nodes

    origin = to_np(data["origin"]).reshape(-1, 2)[scene]
    theta = float(to_np(data["theta"]).reshape(-1)[scene])
    city = data["city"][scene] if isinstance(data["city"], (list, tuple)) else data["city"]
    R_scene = rot(theta)

    pos = to_np(data["positions"][s:e])          # [n, 50, 2] scene frame
    pad = to_np(data["padding_mask"][s:e])       # [n, 50] True = missing
    focal = int(data["agent_index"][scene]) - s  # agent_index is batch-global
    av = int(data["av_index"][scene]) - s

    pred = to_np(out[..., :2])[:, s:e]           # [M, n, 30, 2]
    pi = to_np(pi)
    pi = pi[s:e] if pi.shape[0] == out.shape[1] else pi[:, s:e].T   # -> [n, M]
    prob = np.exp(pi - pi.max(1, keepdims=True))
    prob /= prob.sum(1, keepdims=True)

    # agent frame -> scene frame
    if rotate:
        ang = to_np(data["rotate_angles"][s:e])
        Ra = np.stack([rot(a) for a in ang])                 # [n, 2, 2]
        pred = np.einsum("mntj,nkj->mntk", pred, Ra)
    pred = pred + pos[None, :, 19:20, :]

    # scene frame -> city frame
    to_city = lambda xy: xy @ R_scene.T + origin
    pos_c, pred_c = to_city(pos), to_city(pred)

    # ---- map ----
    cx, cy = pos_c[focal, 19]
    bbox = [cx - map_range, cx + map_range, cy - map_range, cy + map_range]
    fig, ax = plt.subplots(figsize=(9, 9))
    for poly in avm.find_local_driveable_areas(bbox, city):
        ax.fill(poly[:, 0], poly[:, 1], color="#EDEDED", zorder=0)
    for poly in avm.find_local_lane_polygons(bbox, city):
        ax.plot(poly[:, 0], poly[:, 1], color="#BBBBBB", lw=0.8, zorder=1)

    # ---- agents ----
    for i in range(e - s):
        if pad[i, 19]:                      # not present at current step -> skip
            continue
        is_focal, is_av = i == focal, i == av
        hist = ~pad[i, :20]
        fut = ~pad[i, 20:]
        past_col = "#ECA154" if is_focal else ("#4C72B0" if is_av else "#8C8C8C")

        ax.plot(*pos_c[i, :20][hist].T, color=past_col,
                lw=3 if is_focal else 1.5, zorder=15)
        ax.scatter(*pos_c[i, 19], color=past_col, s=40 if is_focal else 15, zorder=16)

        gt = np.concatenate([pos_c[i, 19:20], pos_c[i, 20:][fut]])
        ax.plot(*gt.T, color="#d33e4c", ls="--",
                lw=2.5 if is_focal else 1, zorder=20)

        modes = range(pred_c.shape[0]) if (is_focal or all_modes_for_all) \
            else [int(prob[i].argmax())]
        for m in modes:
            tr = np.concatenate([pos_c[i, 19:20], pred_c[m, i]])
            ax.plot(*tr.T, color="#007672",
                    lw=2.5 if is_focal else 1,
                    alpha=float(0.3 + 0.7 * prob[i, m] / prob[i].max()),
                    zorder=18)
            ax.scatter(*tr[-1], color="#007672", s=12 if is_focal else 5, zorder=19)

    # legend without duplicates
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], color="#ECA154", lw=3, label="Focal past"),
        Line2D([], [], color="#4C72B0", lw=2, label="AV past"),
        Line2D([], [], color="#8C8C8C", lw=2, label="Other past"),
        Line2D([], [], color="#d33e4c", lw=2, ls="--", label="Ground truth"),
        Line2D([], [], color="#007672", lw=2, label="Prediction"),
    ], loc="upper right")
    ax.set_xlim(bbox[0], bbox[1]); ax.set_ylim(bbox[2], bbox[3])
    ax.set_aspect("equal"); ax.set_title(f"{city}  scene {scene}")
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    plt.show()



if __name__ == '__main__':
    pl.seed_everything(2022)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1) #32
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--decoder_type', type=str, default=None)
    args = parser.parse_args()

    trainer = pl.Trainer.from_argparse_args(args)
    model = HiVT.load_from_checkpoint(checkpoint_path=args.ckpt_path, parallel=True)
    val_dataset = ArgoverseV1Dataset(root=args.root, split='val', local_radius=model.hparams.local_radius)
    dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)
    #trainer.validate(model, dataloader)

    model.eval()
    model.cuda()
    avm = ArgoverseMap()
    i=0
    for batch in dataloader:
        batch = batch.to('cuda')
        i+=1
        with torch.no_grad():
            y_hat, pi = model(batch)
            plot_scene(batch,y_hat, pi, avm, scene=0)
            if i==6:
                k=0
                city_name = batch.city[0]
                sample_past_trajectory = batch.x[k].cpu().numpy()
                sample_groundtruth = batch.y[k].cpu().numpy()
                sample_forecasted_trajectories = [y_hat[i,k,:,:2].cpu().numpy() for i in range(y_hat.shape[0])]
                sample_city_name = batch.city[0]
                # plot_single_vehicle(
                #     avm=avm,
                #     sample_past_trajectory=sample_past_trajectory,
                #     sample_groundtruth=sample_groundtruth,
                #     sample_forecasted_trajectories=sample_forecasted_trajectories,
                #     sample_city_name=sample_city_name,
                #     ls='-'
                # )
                break