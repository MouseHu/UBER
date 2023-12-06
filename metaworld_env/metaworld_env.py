# from metaworld.envs.mujoco.env_dict import ALL_V1_ENVIRONMENTS

from gym import Wrapper, ObservationWrapper, RewardWrapper
from gym.spaces import Box

import time

import gym
import numpy as np
from .doordrawer_env import SawyerDoorDrawerEnv, SawyerDoorDrawerEnv2


class EpisodeMonitor(gym.ActionWrapper):
    """A class that computes episode returns and lengths."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action: np.ndarray):
        observation, reward, done, info = self.env.step(action)

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if done:
            info['episode'] = {}
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

            if hasattr(self, 'get_normalized_score'):
                info['episode']['return'] = self.get_normalized_score(
                    info['episode']['return']) * 100.0

        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        self._reset_stats()
        return self.env.reset()


class DoneWrapper(Wrapper):
    def __init__(self, env, max_length=500):
        super().__init__(env)
        self.env = env
        self.max_length = max_length
        self.max_episode_steps = max_length
        self.num_steps = 0

    def reset(self):
        self.num_steps = 0
        return self.env.reset()

    def step(self, action):
        self.num_steps += 1
        next_state, reward, done, info = self.env.step(action)
        done = done or self.num_steps >= self.max_length
        return next_state, reward, done, info


class SparseRewardWrapper(Wrapper):
    def __init__(self, env):
        super(SparseRewardWrapper, self).__init__(env)

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        reward = float(info['success'])
        return next_state, reward, done, info


class MetaWorldWrapper(ObservationWrapper):
    def __init__(self, env, env_id):
        super().__init__(env)
        self.id = np.zeros(4)
        assert 0 <= env_id < 4
        self.id[env_id] = 1
        self.observation_space = self.get_observation_space()

    def observation(self, observation):
        # tmp = observation[3:6].copy()
        # observation[3:6] = observation[6:9]
        # observation[6:9] = tmp  # swap place
        observation = np.concatenate([observation, self.id])
        return observation

    def get_observation_space(self):
        obs_obj_max_len = self.env._obs_obj_max_len if self.isV2 else 6

        obj_low = np.full(obs_obj_max_len, -np.inf)
        obj_high = np.full(obs_obj_max_len, +np.inf)
        goal_low = np.zeros(3) if self.env._partially_observable \
            else self.env.goal_space.low
        goal_high = np.zeros(3) if self.env._partially_observable \
            else self.env.goal_space.high
        gripper_low = -1.
        gripper_high = +1.
        task_high = np.ones(4)
        task_low = np.zeros(4)
        return Box(
            np.hstack((self.env._HAND_SPACE.low, obj_low, goal_low, task_low)),
            np.hstack((self.env._HAND_SPACE.high, obj_high, goal_high, task_high))
        )


def make_metaworld_env(env_id, sparse=False):
    env = SawyerDoorDrawerEnv2(task_id=env_id)
    env._freeze_rand_vec = False
    if env_id // 2:  # drawer env
        rand_vec = [0.55 - 0.25644326, 1.0, 0.1524281, -0.275, 0.9, 0.05]
    elif env_id == 1:
        rand_vec = [0.7, 1.0, 0.1524281, 0.275, 0.9, 0.05]
    else:
        rand_vec = [-0.25644326, 1.0, 0.1524281, 0.275, 0.9, 0.05]
    task = {"env_cls": SawyerDoorDrawerEnv2, "partially_observable": True, "rand_vec": np.array(rand_vec)}
    env.set_task(task)
    env.reset()  # Reset environment
    env = MetaWorldWrapper(env, env_id)
    env = DoneWrapper(env, 200)
    env = EpisodeMonitor(env)
    if sparse:
        env = SparseRewardWrapper(env)
    env.reset()  # Reset environment
    return env
