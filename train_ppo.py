#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os, sys
import contextlib
import random
from collections import deque
import gym
from time import time,sleep

import numpy as np
import torch
from torchvision import transforms
from PIL import Image

import matplotlib.pyplot as plt

from signal import signal, SIGINT
from contextlib import contextmanager

import habitat
from habitat_baselines.config.default import get_config as cfg_baseline
from habitat import logger
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.config.default import get_config as cfg_env
from habitat.datasets.registration import make_dataset
from ppo_utils import PPO,Policy, RolloutStorage,batch_obs, ppo_args, update_linear_schedule
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from evaluate_ppo_fn import eval_ppo

# with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
#     print("This won't be printed.")
#     sleep(20)
#     print("BLue blah blah")

def _preprocess_depth(depth):
    depth = depth[:, :, 0]*1
    mask2 = depth > 0.99
    depth[mask2] = 0.

    for i in range(depth.shape[1]):
        depth[:,i][depth[:,i] == 0.] = depth[:,i].max()

    mask1 = depth == 0
    depth[mask1] = np.NaN
    depth = depth*1000.
    return depth

class NavRLEnv(habitat.RLEnv):
    def __init__(self, args,config_env, config_baseline, dataset):
        self._config_env = config_env.TASK
        # print("Displaying config")
        # print(config_env)
        self.args = args
        self._config_baseline = config_baseline
        self._previous_target_distance = None
        self._previous_action = None
        self._episode_distance_covered = None

        self.num_actions = 4

        self.action_space = gym.spaces.Discrete(self.num_actions)

        print(gym.spaces.Discrete(self.num_actions))

        

        self.res = transforms.Compose([transforms.ToPILImage(),
                    transforms.Resize((args.frame_height, args.frame_width),
                                      interpolation = Image.NEAREST)])

        # print(self.observation_space)

        super().__init__(config_env, dataset)

        # print("\n\n\n\n\n\n\n\n\nDisplaying this shit\n\n\n\n\n\n\n")

        # self.observation_space = {"depth" : gym.spaces.Box(0, 255,
        #                                         (1, args.frame_height,
        #                                             args.frame_width),
        #                                         dtype='uint8') ,\
        #                             "pointgoal_with_gps_compass" : self.observation_space["pointgoal_with_gps_compass"], "rgb" : gym.spaces.Box(0, 255,
        #                                         (3, args.frame_height,
        #                                             args.frame_width),
        #                                         dtype='uint8')}
        # self.observation_space.spaces["depth"] = gym.spaces.Box(0, 255,
        #                                         (1, args.frame_height,
        #                                             args.frame_width),
        #                                         dtype='uint8')

        # self.observation_space.spaces["rgb"] = gym.spaces.Box(0, 255,
        #                                         (3, args.frame_height,
        #                                             args.frame_width),
        #                                         dtype='uint8')

        # print(self.observation_space,self.observation_space.spaces,self.action_space)

    def reset(self):
        self._previous_action = None
        observations = super().reset()

        # print(observations)

        rgb = observations['rgb'].astype(np.uint8)
        depth_ = observations['depth'].astype(np.uint8)
        self.obs = rgb # For visualization
        
        if self.args.frame_width != self.args.env_frame_width:
            rgb = np.asarray(self.res(rgb))
            depth_ = np.asarray(self.res(depth_))
            depth_ = np.expand_dims(depth_, axis=0)
        #state = rgb.transpose(2, 0, 1)
        # state = np.concatenate((rgb.transpose(2, 0, 1), depth_))

        self._previous_target_distance = self.habitat_env.current_episode.info[
            "geodesic_distance"
        ]
        infos= self.get_info(observations)

        obs = {}
        obs["rgb"] = rgb
        obs["depth"] = depth_
        obs["pointgoal_with_gps_compass"] = observations["pointgoal_with_gps_compass"]
        return observations, infos

    def step(self, action):
        self._previous_action = action
        observations, reward, done, info = super().step(action)

        rgb = observations['rgb'].astype(np.uint8)
        depth_ = observations['depth'].astype(np.uint8)
        self.obs = rgb # For visualization
        
        if self.args.frame_width != self.args.env_frame_width:
            rgb = np.asarray(self.res(rgb))
            depth_ = np.asarray(self.res(depth_))
            depth_ = np.expand_dims(depth_, axis=0)

        obs = {}
        obs["rgb"] = rgb
        obs["depth"] = depth_
        obs["pointgoal_with_gps_compass"] = observations["pointgoal_with_gps_compass"]

        return observations, reward,done,info


    def get_reward_range(self):
        return (
            self._config_baseline.RL.SLACK_REWARD - 1.0,
            self._config_baseline.RL.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = self._config_baseline.RL.SLACK_REWARD

        current_target_distance = self._distance_target()
        reward += self._previous_target_distance - current_target_distance
        self._previous_target_distance = current_target_distance

        if self._episode_success():
            reward += self._config_baseline.RL.SUCCESS_REWARD

        return reward

    def _distance_target(self):
        current_position = self._env.sim.get_agent_state().position.tolist()
        target_position = self._env.current_episode.goals[0].position
        distance = self._env.sim.geodesic_distance(
            current_position, target_position
        )
        return distance

    def _episode_success(self):
        if (
            self._previous_action == HabitatSimActions.STOP
            and self._distance_target() < self._config_env.SUCCESS_DISTANCE
        ):
            return True
        return False

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


def make_env_fn(args, config_env, config_baseline, rank):
    dataset = PointNavDatasetV1(config_env.DATASET)
    config_env.defrost()
    config_env.SIMULATOR.SCENE = dataset.episodes[0].scene_id
    config_env.freeze()
    env = NavRLEnv(args = args,
        config_env=config_env, config_baseline=config_baseline, dataset=dataset
    )
    env.seed(rank)
    return env


def construct_envs(args):
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
        random.shuffle(scenes)

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
        config_env.DATASET.SPLIT = args.split#'train'#
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

    envs = habitat.VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(
            tuple(
                zip(args_list, env_configs, baseline_configs, range(args.num_processes))
            )
        ),
    )

    return envs

def run_training():
    plt.figure()
    parser = ppo_args()
    args = parser.parse_args()

    random.seed(args.seed)

    device = torch.device("cuda:{}".format(args.pth_gpu_id))

    logger.add_filehandler(args.log_file)

    if not os.path.isdir(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder)

    for p in sorted(list(vars(args))):
        logger.info("{}: {}".format(p, getattr(args, p)))
    envs = construct_envs(args)
    task_cfg = cfg_env(config_paths=args.task_config)
    actor_critic = Policy(
        observation_space=envs.observation_spaces[0],
        action_space=envs.action_spaces[0],
        hidden_size=args.hidden_size,
        goal_sensor_uuid=task_cfg.TASK.GOAL_SENSOR_UUID,
    )
    actor_critic.to(device)

    agent = PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
    )

    logger.info(
        "agent number of parameters: {}".format(
            sum(param.numel() for param in agent.parameters())
        )
    )

    # logger.info(
    #     "number of episodes: {}".format(args.num_episodes
    #     )
    # )

    observations,infos = envs.reset()

    # print(envs.observation_spaces,envs.action_spaces)

    batch = batch_obs(observations)
    rollouts = RolloutStorage(
        args.num_steps,
        envs.num_envs,
        envs.observation_spaces[0],
        envs.action_spaces[0],
        args.hidden_size,
    )
    for sensor in rollouts.observations:
        rollouts.observations[sensor][0].copy_(batch[sensor])
    rollouts.to(device)

    episode_rewards = torch.zeros(envs.num_envs, 1)
    episode_counts = torch.zeros(envs.num_envs, 1)
    current_episode_reward = torch.zeros(envs.num_envs, 1)
    window_episode_reward = deque()
    window_episode_counts = deque()

    t_start = time()
    env_time = 0
    pth_time = 0
    count_steps = 0
    count_checkpoints = 0
    update = 0 if args.load_train == -1 else args.load_train
    # dones_prev = [False for i in range(envs.num_envs)]
    test_rewards=[]

    if args.load_train>0:
        # print(agent.state_dict)
        # print("\n\n\n Loading state on update {} for training\n\n\n".format(args.load_train))
        load_path = "ckpt.{}.pth".format(args.load_train)
        # print(torch.load(os.path.join(args.checkpoint_folder,load_path)))
        if not os.path.exists(os.path.join(args.checkpoint_folder,load_path)):
            raise Exception("load path {} doesn't exist".format(load_path))
        agent.load_state_dict(torch.load(os.path.join(args.checkpoint_folder,load_path))["state_dict"])
        if os.path.exists("./latest-train-results-from-train-ppo.npy"):
            print("Loading rewards array")
            test_rewards = np.load("./latest-train-results-from-train-ppo.npy")
            # print(test_rewards[-1,:])
            episode_counts = torch.tensor([[test_rewards[-1,-1]//envs.num_envs]]).repeat(envs.num_envs,1)
            episode_counts[-1,0] += test_rewards[-1,-1]-episode_counts.sum()
            test_rewards = test_rewards.tolist()

    def handler(signal_received, frame):
        # Handle any cleanup here
        print('SIGINT or CTRL-C detected. Exiting gracefully. Saved checkpoint {}.'.format(update))
        checkpoint = {"state_dict": agent.state_dict()}
        torch.save(
            checkpoint,
            os.path.join(
                args.checkpoint_folder,
                "ckpt.{}.pth".format(update),
            ),
        )

        np.save("latest-train-results-from-train-ppo",test_rewards)

        # # print(test_rewards)
        if test_rewards!=[]:
            temp_arr = test_rewards
            test_rewards=np.array(temp_arr)

            plt.figure
            ax1 = plt.subplot(3,1,1)
            ax1.set_title("Avg reward per episode vs episode")
            plt.plot(test_rewards[:,-1],test_rewards[:,0])
            ax2 = plt.subplot(3,1,2)
            ax2.set_title("Avg spl per episode vs episode")
            plt.plot(test_rewards[:,-1],test_rewards[:,1])
            ax3 = plt.subplot(3,1,3)
            ax3.set_title("Avg success rate per episode vs episode")
            plt.xlabel("Episodes")
            plt.plot(test_rewards[:,-1],test_rewards[:,2])
            plt.tight_layout()
            plt.savefig("latest-train-results-from-train-ppo.png")
        sys.exit(0)
    signal(SIGINT, handler)
    # print("Number Episodes : {}".format(args.num_episodes))

    while update < args.num_updates:
        if episode_counts.sum() == 0:
            print("\n\nStarted Training")

        # print("Epsiodes : {}".format(episode_counts))
        if args.use_linear_lr_decay:
            update_linear_schedule(
                agent.optimizer, update, args.num_updates, args.lr
            )

        agent.clip_param = args.clip_param * (1 - update / args.num_updates)

        for step in range(args.num_steps):
            t_sample_action = time()
            # sample actions
            with torch.no_grad():
                step_observation = {
                    k: v[step] for k, v in rollouts.observations.items()
                }
                (
                    values,
                    actions,
                    actions_log_probs,
                    recurrent_hidden_states,
                ) = actor_critic.act(
                    step_observation,
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step],
                )
            pth_time += time() - t_sample_action

            t_step_env = time()
            observations, rewards, dones, infos = envs.step([a[0].item() for a in actions])
            env_time += time() - t_step_env

            t_update_stats = time()
            batch = batch_obs(observations)
            rewards = torch.tensor(rewards, dtype=torch.float)
            rewards = rewards.unsqueeze(1)

            masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones], dtype=torch.float
            )
            # for i in range(len(dones)):
            #     dones_prev[i] = dones[i]

            current_episode_reward += rewards
            episode_rewards += (1 - masks) * current_episode_reward
            episode_counts += 1 - masks
            current_episode_reward *= masks

            rollouts.insert(
                batch,
                recurrent_hidden_states,
                actions,
                actions_log_probs,
                values,
                rewards,
                masks,
            )

            count_steps += envs.num_envs
            pth_time += time() - t_update_stats

        if len(window_episode_reward) == args.reward_window_size:
            window_episode_reward.popleft()
            window_episode_counts.popleft()
        window_episode_reward.append(episode_rewards.clone())
        window_episode_counts.append(episode_counts.clone())

        t_update_model = time()
        with torch.no_grad():
            last_observation = {
                k: v[-1] for k, v in rollouts.observations.items()
            }
            next_value = actor_critic.get_value(
                last_observation,
                rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1],
            ).detach()

        rollouts.compute_returns(
            next_value, args.use_gae, args.gamma, args.tau
        )

        value_loss, action_loss, dist_entropy = agent.update(rollouts,update,test_rewards,args)

        rollouts.after_update()
        pth_time += time() - t_update_model

        # log stats
        if update > 0 and update % args.log_interval == 0:
            logger.info(
                "update: {}\tfps: {:.3f}\t".format(
                    update, count_steps / (time() - t_start)
                )
            )

            logger.info(
                "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                "frames: {}".format(update, env_time, pth_time, count_steps)

            )
            logger.info("episodes : {} | test_rewards_length : {}".format(episode_counts,len(test_rewards)))

            window_rewards = (
                window_episode_reward[-1] - window_episode_reward[0]
            ).sum()
            window_counts = (
                window_episode_counts[-1] - window_episode_counts[0]
            ).sum()

            if window_counts > 0:
                logger.info(
                    "Average window size {} reward: {:3f}".format(
                        len(window_episode_reward),
                        (window_rewards / window_counts).item(),
                    )
                )
            else:
                logger.info("No episodes finish in current window")

        # checkpoint model
        if update % args.checkpoint_interval == 0:
            # print("saving {}".format(count_checkpoints))
            checkpoint = {"state_dict": agent.state_dict()}
            torch.save(
                checkpoint,
                os.path.join(
                    args.checkpoint_folder,
                    "ckpt.{}.pth".format(update),
                ),
            )
            # model_path,sim_gpu_id,pth_gpu_id,num_processes,hidden_size=512,count_test_episodes=100,sensors="RGB_SENSOR,DEPTH_SENSOR",task_config="habitat-lab/configs/tasks/pointnav.yaml"

            episode_reward_mean , episode_spl_mean, episode_success_mean = eval_ppo(model_path=os.path.join(args.checkpoint_folder,"ckpt.{}.pth".format(update)), sim_gpu_id=args.sim_gpu_id, pth_gpu_id=args.pth_gpu_id, num_processes=args.num_processes,count_test_episodes=30)

            test_rewards.append([episode_reward_mean , episode_spl_mean, episode_success_mean,episode_counts.sum()])

            test_rewards_arr = np.array(test_rewards)
            np.save("latest-train-results-from-train-ppo",test_rewards_arr)

            # # print(test_rewards)
            # ::max(len(test_rewards_arr)//1000,-1,1)
            plt.clf()
            ax1 = plt.subplot(3,1,1)
            ax1.set_title("Avg reward per episode vs episode")
            plt.plot(test_rewards_arr[:,-1],test_rewards_arr[:,0])
            ax2 = plt.subplot(3,1,2)
            ax2.set_title("Avg spl per episode vs episode")
            plt.plot(test_rewards_arr[:,-1],test_rewards_arr[:,1])
            ax3 = plt.subplot(3,1,3)
            ax3.set_title("Avg success rate per episode vs episode")
            plt.xlabel("Episodes")
            plt.plot(test_rewards_arr[:,-1],test_rewards_arr[:,2])
            plt.tight_layout()
            plt.savefig("latest-train-results-from-train-ppo.png")

            # print("Average episode reward: {:.6f}".format(episode_reward_mean))
            # print("Average episode success: {:.6f}".format(episode_success_mean))
            # print("Average episode spl: {:.6f}".format(episode_spl_mean))

            count_checkpoints += 1
        update += 1
        # if update > args.num_updates:
        #     break
    # print(test_rewards)
    test_rewards = np.array(test_rewards)
    np.save("latest-train-results-from-train-ppo",test_rewards)

    # # print(test_rewards)

    plt.clf()
    ax1 = plt.subplot(3,1,1)
    ax1.set_title("Avg reward per episode vs episode")
    plt.plot(test_rewards[:,-1],test_rewards[:,0])
    ax2 = plt.subplot(3,1,2)
    ax2.set_title("Avg spl per episode vs episode")
    plt.plot(test_rewards[:,-1],test_rewards[:,1])
    ax3 = plt.subplot(3,1,3)
    ax3.set_title("Avg success rate per episode vs episode")
    plt.xlabel("Episodes")
    plt.plot(test_rewards[:,-1],test_rewards[:,2])
    plt.tight_layout()
    plt.savefig("latest-train-results-from-train-ppo.png")

if __name__ == "__main__":
    run_training()