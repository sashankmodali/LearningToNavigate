#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch

import habitat
import numpy as np

from habitat import logger
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.datasets.registration import make_dataset

from habitat_baselines.config.default import get_config as cfg_baseline
from habitat.config.default import get_config as cfg_env

from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from ppo_model import PPO, Policy, batch_obs



def eval_ppo(args,model_path,sim_gpu_id,pth_gpu_id,num_processes,count_test_episodes=100,hidden_size=512,sensors="RGB_SENSOR,DEPTH_SENSOR",task_config="habitat-lab/configs/tasks/pointnav.yaml"):
    
    device = torch.device("cuda:{}".format(pth_gpu_id))

    env_configs = []
    baseline_configs = []
    basic_config = cfg_env(config_paths=args.task_config, opts=args.opts)
    basic_config.defrost()
    basic_config.DATASET.SPLIT = 'val'#'train'#
    basic_config.DATASET.DATA_PATH = ("data/datasets/pointnav/gibson/v1/{split}/{split}.json.gz")
    basic_config.DATASET.TYPE = "PointNavDataset-v1"
    basic_config.freeze()
    dataset = PointNavDatasetV1(basic_config.DATASET)
    scenes = dataset.get_scenes_to_load(basic_config.DATASET)

    if len(scenes) > 0:

        assert len(scenes) >= args.num_processes, (
            "reduce the number of processes as there "
            "aren't enough number of scenes"
        )
        scene_split_size = int(np.floor(len(scenes) / args.num_processes))

    scene_splits = [[] for _ in range(args.num_processes)]
    for j, s in enumerate(scenes):
        scene_splits[j % len(scene_splits)].append(s)

    assert sum(map(len, scene_splits)) == len(scenes)
    args_list=[]

    for i in range(args.num_processes):
        config_env = cfg_env(config_paths=args.task_config, opts=args.opts)
        config_env.defrost()
        config_env.DATASET.SPLIT = 'val'#'train'#
        config_env.DATASET.DATA_PATH = (
        "data/datasets/pointnav/gibson/v1/{split}/{split}.json.gz")
        config_env.DATASET.TYPE = "PointNavDataset-v1"
        if len(scenes) > 0:
            config_env.DATASET.CONTENT_SCENES = scene_splits[i]

        config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args.sim_gpu_id

        config_env.ENVIRONMENT.MAX_EPISODE_STEPS = args.max_episode_length

        config_env.SIMULATOR.RGB_SENSOR.WIDTH = args.env_frame_width
        config_env.SIMULATOR.RGB_SENSOR.HEIGHT = args.env_frame_height
        config_env.SIMULATOR.RGB_SENSOR.HFOV = args.hfov
        config_env.SIMULATOR.RGB_SENSOR.POSITION = [0, args.camera_height, 0]

        config_env.SIMULATOR.DEPTH_SENSOR.WIDTH = args.env_frame_width
        config_env.SIMULATOR.DEPTH_SENSOR.HEIGHT = args.env_frame_height
        config_env.SIMULATOR.DEPTH_SENSOR.HFOV = args.hfov
        config_env.SIMULATOR.DEPTH_SENSOR.POSITION = [0, args.camera_height, 0]

        agent_sensors = args.sensors.strip().split(",")
        for sensor in agent_sensors:
            assert sensor in ["RGB_SENSOR", "DEPTH_SENSOR"]
        config_env.SIMULATOR.AGENT_0.SENSORS = agent_sensors
        config_env.freeze()
        env_configs.append(config_env)
        config_baseline = cfg_baseline(opts=['BASE_TASK_CONFIG_PATH',args.task_config])
        baseline_configs.append(config_baseline)
        args_list.append(args)

        # logger.info("config_env: {}".format(config_env))
    assert len(baseline_configs) > 0, "empty list of datasets"

    envs = habitat.VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(
            tuple(
                zip(args_list, env_configs, baseline_configs, range(args.num_processes))
            )
        ),
    )

    ckpt = torch.load(model_path, map_location=device)

    actor_critic = Policy(
        observation_space=envs.observation_spaces[0],
        action_space=envs.action_spaces[0],
        hidden_size=512,
        goal_sensor_uuid=env_configs[0].TASK.GOAL_SENSOR_UUID,args=args
    )
    actor_critic.to(device)

    ppo = PPO(
        actor_critic=actor_critic,
        clip_param=0.1,
        ppo_epoch=4,
        num_mini_batch=32,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        lr=2.5e-4,
        eps=1e-5,
        max_grad_norm=0.5,
    )

    ppo.load_state_dict(ckpt["state_dict"])

    actor_critic = ppo.actor_critic

    observations,info = envs.reset()
    batch = batch_obs(observations)
    for sensor in batch:
        batch[sensor] = batch[sensor].to(device)

    episode_rewards = torch.zeros(envs.num_envs, 1, device=device)
    episode_spls = torch.zeros(envs.num_envs, 1, device=device)
    episode_success = torch.zeros(envs.num_envs, 1, device=device)
    episode_counts = torch.zeros(envs.num_envs, 1, device=device)
    current_episode_reward = torch.zeros(envs.num_envs, 1, device=device)

    test_recurrent_hidden_states = torch.zeros(
        num_processes, hidden_size, device=device
    )
    not_done_masks = torch.zeros(num_processes, 1, device=device)

    while episode_counts.sum() < count_test_episodes:
        with torch.no_grad():
            if args.perception_model!=2:
                _, actions, _, test_recurrent_hidden_states = actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    not_done_masks,
                    deterministic=False,
                )
            else:
                _, actions, _, test_recurrent_hidden_states,_ = actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    not_done_masks,
                    deterministic=False,
                )

        observations, rewards, dones, infos  = envs.step([a[0].item() for a in actions])
        batch = batch_obs(observations)
        for sensor in batch:
            batch[sensor] = batch[sensor].to(device)

        not_done_masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float,
            device=device,
        )
        for i in range(not_done_masks.shape[0]):
            if not_done_masks[i].item() == 0:
                episode_spls[i] += infos[i]["spl"]
                if infos[i]["spl"] > 0:
                    episode_success[i] += 1

        rewards = torch.tensor(
            rewards, dtype=torch.float, device=device
        ).unsqueeze(1)
        current_episode_reward += rewards
        episode_rewards += (1 - not_done_masks) * current_episode_reward
        episode_counts += 1 - not_done_masks
        current_episode_reward *= not_done_masks

    episode_reward_mean = (episode_rewards / episode_counts).mean().item()
    episode_spl_mean = (episode_spls / episode_counts).mean().item()
    episode_success_mean = (episode_success / episode_counts).mean().item()

    return episode_reward_mean , episode_spl_mean, episode_success_mean


# if __name__ == "__main__":
#     pass

from to_train_ppo import NavRLEnv, make_env_fn