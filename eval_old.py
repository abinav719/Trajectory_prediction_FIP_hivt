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
from argparse import ArgumentParser

import pytorch_lightning as pl
from torch_geometric.data import DataLoader
from argoverse.map_representation.map_api import ArgoverseMap
from datasets import ArgoverseV1Dataset
from models.hivt import HiVT
import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import List


def plot_single_vehicle(
        avm: ArgoverseMap,
        sample_past_trajectory: np.ndarray,  # (20, 2)
        sample_groundtruth: np.ndarray,  # (20, 2)
        sample_forecasted_trajectories: List[np.ndarray],  # List[(30, 2)]
        sample_city_name: str,
        ls: str):
    sample_groundtruth = np.concatenate((np.expand_dims(sample_past_trajectory[-1], axis=0), sample_groundtruth),
                                        axis=0)
    for sample_forecasted_trajectory in sample_forecasted_trajectories:
        sample_forecasted_trajectory = np.concatenate(
            (np.expand_dims(sample_past_trajectory[-1], axis=0), sample_forecasted_trajectory), axis=0)

    ## Plot history
    obs_len = sample_past_trajectory.shape[0]
    pred_len = sample_groundtruth.shape[0]
    plt.plot(
        sample_past_trajectory[:, 0],
        sample_past_trajectory[:, 1],
        color="#ECA154",
        label="Past Trajectory",
        alpha=1,
        linewidth=3,
        zorder=15,
        ls=ls
    )

    ## Plot future
    plt.plot(
        sample_groundtruth[:, 0],
        sample_groundtruth[:, 1],
        color="#d33e4c",
        label="Ground Truth",
        alpha=1,
        linewidth=3.0,
        zorder=20,
        ls="--"
    )

    ## Plot prediction
    for j in range(len(sample_forecasted_trajectories)):
        plt.plot(
            sample_forecasted_trajectories[j][:, 0],
            sample_forecasted_trajectories[j][:, 1],
            color="#007672",
            label="Forecasted Trajectory",
            alpha=1,
            linewidth=3.2,
            zorder=15,
            ls="--"
        )

        # Plot the end marker for forcasted trajectories
        # plt.arrow(
        #     sample_forecasted_trajectories[j][-2, 0],
        #     sample_forecasted_trajectories[j][-2, 1],
        #     sample_forecasted_trajectories[j][-1, 0] - sample_forecasted_trajectories[j][-2, 0],
        #     sample_forecasted_trajectories[j][-1, 1] - sample_forecasted_trajectories[j][-2, 1],
        #     color="#007672",
        #     label="Forecasted Trajectory",
        #     alpha=1,
        #     linewidth=3.2,
        #     zorder=15,
        #     head_width=1.1,
        # )

    ## Plot the end marker for history
    # plt.arrow(
    #     sample_past_trajectory[-2, 0],
    #     sample_past_trajectory[-2, 1],
    #     sample_past_trajectory[-1, 0] - sample_past_trajectory[-2, 0],
    #     sample_past_trajectory[-1, 1] - sample_past_trajectory[-2, 1],
    #     color="#ECA154",
    #     label="Past Trajectory",
    #     alpha=1,
    #     linewidth=3,
    #     zorder=25,
    #     head_width=1.0,
    # )

    ## Plot the end marker for future
    # plt.arrow(
    #     sample_groundtruth[-2, 0],
    #     sample_groundtruth[-2, 1],
    #     sample_groundtruth[-1, 0] - sample_groundtruth[-2, 0],
    #     sample_groundtruth[-1, 1] - sample_groundtruth[-2, 1],
    #     color="#d33e4c",
    #     label="Ground Truth",
    #     alpha=1,
    #     linewidth=3.0,
    #     zorder=25,
    #     head_width=1.0,
    # )

    ## Plot history context
    for j in range(obs_len):
        lane_ids = avm.get_lane_ids_in_xy_bbox(
            sample_past_trajectory[j, 0],
            sample_past_trajectory[j, 1],
            sample_city_name,
            query_search_range_manhattan=65,
        )
        [avm.draw_lane(lane_id, sample_city_name) for lane_id in lane_ids]

    ## Plot future context
    for j in range(pred_len):
        lane_ids = avm.get_lane_ids_in_xy_bbox(
            sample_groundtruth[j, 0],
            sample_groundtruth[j, 1],
            sample_city_name,
            query_search_range_manhattan=65,
        )
        [avm.draw_lane(lane_id, sample_city_name) for lane_id in lane_ids]
    # Finally, display the plot
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
    args = parser.parse_args()

    trainer = pl.Trainer.from_argparse_args(args)
    model = HiVT.load_from_checkpoint(checkpoint_path=args.ckpt_path, parallel=True)
    val_dataset = ArgoverseV1Dataset(root=args.root, split='val', local_radius=model.hparams.local_radius)
    dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)
    trainer.validate(model, dataloader)

    # model.eval()
    # model.cuda()
    # avm = ArgoverseMap()
    # for batch in dataloader:
    #     batch = batch.to('cuda')
    #
    #     with torch.no_grad():
    #         y_hat, pi = model(batch)
    #     city_name = batch.city[0]  # use city from batch
    #     sample_past_trajectory = batch.x[0].cpu().numpy()
    #     sample_groundtruth = batch.y[0].cpu().numpy()
    #     sample_forecasted_trajectories = [y_hat[i,0,:,:2].cpu().numpy() for i in range(y_hat.shape[0])]
    #     sample_city_name = batch.city[0]
    #     plot_single_vehicle(
    #         avm=avm,
    #         sample_past_trajectory=sample_past_trajectory,
    #         sample_groundtruth=sample_groundtruth,
    #         sample_forecasted_trajectories=sample_forecasted_trajectories,
    #         sample_city_name=sample_city_name,
    #         ls='-'
    #     )
    #     break



