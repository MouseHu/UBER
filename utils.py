import gym
import numpy as np
import torch
import d4rl
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GAME_MEAN = {
    "walker2d-medium-v0": np.array([1.1666039, 0.18356155, -0.25781313, -0.55927265, 0.33395785, -0.15875997,
                                    -0.3358816, 0.32632783, 1.6481092, -0.23856151, -0.17951322, -0.6965072,
                                    -1.0953195, -0.40002576, -0.50293297, -0.33538282, -0.49636695]),
    "halfcheetah-expert-v0": np.array([-0.11002816, 0.15680763, 0.10378725, 0.14687687, 0.07839588, -0.20106335,
                                       -0.08224171, -0.2802395, 4.463403, -0.07580097, -0.09260748, 0.41871706,
                                       -0.41171676, 0.11628567, -0.06000552, -0.09738238, -0.14540626]),
    "halfcheetah-medium-v0": np.array([-0.11002816, 0.15680763, 0.10378725, 0.14687687, 0.07839588, -0.20106335,
                                       -0.08224171, -0.2802395, 4.463403, -0.07580097, -0.09260748, 0.41871706,
                                       -0.41171676, 0.11628567, -0.06000552, -0.09738238, -0.14540626]),
    "ant-medium-v0": np.array([6.00594223e-01, 8.94445777e-01, -2.13360582e-02, -4.27316688e-02,
                               3.46369892e-01, 7.92342052e-02, 6.36665761e-01, -9.32592899e-02,
                               -6.00977957e-01, 3.85377444e-02, -7.77772844e-01, -4.80947763e-01,
                               7.21115053e-01, 3.36730123e+00, 3.30466390e-01, -6.17537764e-04,
                               1.34417582e-02, -4.43380103e-02, 1.75254643e-02, -8.21914058e-03,
                               3.05980071e-02, 4.43179486e-04, -2.63530221e-02, 3.18840817e-02,
                               -2.03623753e-02, 1.55917425e-02, 4.53362092e-02, 0.00000000e+00,
                               0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                               0.00000000e+00, 4.33813286e-04, 5.90213982e-04, 1.26110208e-07,
                               6.98995369e-04, -8.90552066e-04, 1.64154340e-02, 0.00000000e+00,
                               0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                               0.00000000e+00, 3.02396529e-05, 8.85565896e-05, 8.03554576e-05,
                               2.35492607e-05, -4.45882652e-05, 5.74514677e-04, 1.00169115e-01,
                               -5.16522638e-02, -5.35872877e-02, 4.87640351e-02, 9.51740984e-03,
                               1.01396225e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                               0.00000000e+00, 0.00000000e+00, 0.00000000e+00, -8.63465830e-05,
                               -2.88384239e-04, -4.99418602e-05, -1.19079355e-04, 7.72463100e-05,
                               1.11228763e-04, 1.63670573e-02, 6.70311153e-02, 6.94384286e-03,
                               -4.70876917e-02, 6.21639541e-04, 6.91894516e-02, 0.00000000e+00,
                               0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                               0.00000000e+00, -1.22248064e-04, -3.94487397e-05, 3.65079541e-05,
                               5.47036034e-05, 1.20963763e-04, 1.04662427e-04, -9.36306864e-02,
                               -3.09684947e-02, 4.19418551e-02, 3.51736210e-02, 6.30414486e-03,
                               9.62535143e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                               0.00000000e+00, 0.00000000e+00, 0.00000000e+00, -7.94019579e-05,
                               2.42262686e-05, -5.71605415e-05, -2.26515112e-06, 2.63253169e-05,
                               3.68543726e-04, -3.82022001e-02, -5.73370643e-02, -4.39664721e-03,
                               4.74210223e-04, -2.64345063e-04, 5.93179800e-02, ]),
    "hopper-medium-v0": np.array([1.3001906, 0.01509409, -0.26983166, -0.33195248, 0.04392744,
                                  2.1276922, -0.20837338, 0.00489949, -0.447387, -0.15586728,
                                  -0.3525342])
}

GAME_STD = {
    "walker2d-medium-v0": np.array([0.1116948, 0.45875406, 0.4934436, 0.794562, 0.6605883, 0.3066052,
                                    0.4211303, 0.6848164, 1.5526043, 1.0877421, 4.3109226, 4.6052437,
                                    5.2735662, 6.443102, 4.532768, 4.861329, 5.9127865]),
    "halfcheetah-medium-v0": np.array([0.10956703, 0.6119863, 0.49235544, 0.44962165, 0.39817896, 0.4823394,
                                       0.30695462, 0.26474255, 1.9024047, 0.939795, 1.625154, 14.427593,
                                       11.996738, 11.985555, 12.159913, 8.127248, 6.419199]),
    "halfcheetah-expert-v0": np.array([0.10956703, 0.6119863, 0.49235544, 0.44962165, 0.39817896, 0.4823394,
                                       0.30695462, 0.26474255, 1.9024047, 0.939795, 1.625154, 14.427593,
                                       11.996738, 11.985555, 12.159913, 8.127248, 6.419199]),
    "ant-medium-v0": np.array([1.1062606e-01, 1.5092355e-01, 1.1295999e-01, 1.3286906e-01, 1.6192152e-01,
                               3.7021142e-01, 1.8876345e-01, 3.4074292e-01, 1.6482490e-01, 4.4274414e-01,
                               2.4532853e-01, 1.5791972e-01, 2.7594328e-01, 1.0854841e+00, 7.5272202e-01,
                               7.7509218e-01, 1.7234231e+00, 1.5591981e+00, 1.6718178e+00, 4.8824692e+00,
                               2.9466484e+00, 4.2435117e+00, 2.1523983e+00, 7.0599337e+00, 2.6832943e+00,
                               1.1342825e+00, 3.5638821e+00, 1.0000000e-03, 1.0000000e-03, 1.0000000e-03,
                               1.0000000e-03, 1.0000000e-03, 1.0000000e-03, 5.8611676e-02, 5.4650154e-02,
                               1.3510488e-02, 9.1327704e-02, 9.7650580e-02, 1.2759840e-01, 1.0000000e-03,
                               1.0000000e-03, 1.0000000e-03, 1.0000000e-03, 1.0000000e-03, 1.0000000e-03,
                               2.0261114e-02, 2.0604799e-02, 1.8808948e-02, 2.2553181e-02, 2.2583496e-02,
                               2.4737922e-02, 3.0089292e-01, 2.9297131e-01, 2.9690889e-01, 3.0506033e-01,
                               2.9957080e-01, 3.0125484e-01, 1.0000000e-03, 1.0000000e-03, 1.0000000e-03,
                               1.0000000e-03, 1.0000000e-03, 1.0000000e-03, 2.5591981e-02, 2.5628431e-02,
                               2.2992205e-02, 2.9596901e-02, 3.0150129e-02, 3.3779152e-02, 2.5125551e-01,
                               2.5216347e-01, 2.4753653e-01, 2.5503504e-01, 2.5635090e-01, 2.5308794e-01,
                               1.0000000e-03, 1.0000000e-03, 1.0000000e-03, 1.0000000e-03, 1.0000000e-03,
                               1.0000000e-03, 2.6262373e-02, 2.3021376e-02, 2.2086490e-02, 2.8953567e-02,
                               2.8960073e-02, 3.2830965e-02, 2.9230532e-01, 2.9499784e-01, 2.9522324e-01,
                               3.0129063e-01, 2.9617333e-01, 2.9410028e-01, 1.0000000e-03, 1.0000000e-03,
                               1.0000000e-03, 1.0000000e-03, 1.0000000e-03, 1.0000000e-03, 1.6725296e-02,
                               1.6274113e-02, 1.5182885e-02, 1.8606881e-02, 1.8298745e-02, 1.9997722e-02,
                               2.2653739e-01, 2.3545633e-01, 2.1892478e-01, 2.3881425e-01, 2.3209566e-01,
                               2.3720753e-01]),
    "hopper-medium-v0": np.array([0.1686824, 0.08239981, 0.28678918, 0.2971138, 0.6330729,
                                  0.8770976, 1.4237294, 0.9800177, 1.8511007, 3.185214,
                                  5.6044817])
}


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, reward_dim=1, max_size=int(1e6), online_buffer=False):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, reward_dim))
        self.raw_reward = np.zeros((max_size, 1))
        self.ctrl_reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.tensorify = False
        self.online_buffer = online_buffer

        self.state_tensor = None
        self.action_tensor = None
        self.next_state_tensor = None
        self.reward_tensor = None
        self.raw_reward_tensor = None
        self.ctrl_reward_tensor = None
        self.not_done_tensor = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        if self.online_buffer:
            return (
                torch.tensor(self.state[ind], dtype=torch.float, device=self.device),
                torch.tensor(self.action[ind], dtype=torch.float, device=self.device),
                torch.tensor(self.next_state[ind], dtype=torch.float, device=self.device),
                torch.tensor(self.reward[ind], dtype=torch.float, device=self.device),
                torch.tensor(self.not_done[ind], dtype=torch.float, device=self.device)
            )
        if not self.tensorify:
            self.state_tensor = torch.tensor(self.state, dtype= torch.float, device=self.device)
            self.action_tensor = torch.tensor(self.action, dtype= torch.float, device=self.device)
            self.next_state_tensor = torch.tensor(self.next_state, dtype=torch.float, device=self.device)
            self.not_done_tensor = torch.tensor(self.not_done, dtype=torch.float, device=self.device)
            self.reward_tensor = torch.tensor(self.reward, dtype= torch.float, device=self.device)
            self.tensorify = True

        return (
            self.state_tensor[ind],
            self.action_tensor[ind],
            self.next_state_tensor[ind],
            self.reward_tensor[ind],
            self.not_done_tensor[ind]
        )


    def sample_contrastive(self, batch_size, neg_dim=100):
        assert self.size >= neg_dim + 2
        ind = np.random.randint(0, self.size, size=batch_size)
        next_ind = (ind + 1) % self.size
        neg_ind = np.random.randint(0, self.size, size=(batch_size, neg_dim))
        # remove current sample
        for i in range(batch_size):
            while ind[i] in neg_ind[i, :] or (ind[i] + 1) % self.size in neg_ind[i]:
                neg_ind[i, :] = np.random.randint(0, self.size, size=neg_dim)
        neg_ind = neg_ind.reshape(-1)

        tar_state = torch.FloatTensor(self.state[ind]).to(self.device)
        tar_action = torch.FloatTensor(self.action[ind]).to(self.device)
        pos_state = torch.FloatTensor(self.next_state[ind]).to(self.device)
        pos_action = torch.FloatTensor(self.action[next_ind]).to(self.device)
        neg_state = torch.FloatTensor(self.state[neg_ind]).to(self.device)
        neg_action = torch.FloatTensor(self.action[neg_ind]).to(self.device)

        return tar_state, tar_action, pos_state, pos_action, neg_state, neg_action

    def convert_D4RL(self, dataset, load_reward=False):
        self.state = dataset['observations']
        self.action = dataset['actions']
        self.next_state = dataset['next_observations']
        if load_reward:
            self.reward = dataset['rewards'].reshape(-1, 1)
            self.raw_reward = self.reward
        else:
            self.raw_reward = dataset['rewards'].reshape(-1, 1)
        self.ctrl_reward = np.sum(np.square(self.action), axis=1)

        self.not_done = 1. - dataset['terminals'].reshape(-1, 1)
        self.size = self.state.shape[0]

    def convert_D4RL_noisy(self, dataset, env_name, datanumber, discount):
        env = gym.make(env_name)
        dataset = d4rl.sequence_dataset(env)
        j = 0
        for seq in dataset:
            j += 1
            observations, actions, dones, rewards, truly_dones = seq['observations'], seq['actions'], seq[
                'timeouts'], seq['rewards'], seq['terminals']
            next_observations = seq['next_observations']
            length = len(observations)
            for i in range(length):
                self.add(observations[i], actions[i], next_observations[i], rewards[i], truly_dones[i], discount)
            if j == datanumber:
                break
        print(env_name, ' load: ', j)

        env_name_2 = 'walker2d-random-v2'
        env = gym.make(env_name_2)
        dataset = d4rl.sequence_dataset(env)
        k = 0
        for seq in dataset:
            k += 1
            observations, actions, dones, rewards, truly_dones = seq['observations'], seq['actions'], seq[
                'timeouts'], seq['rewards'], seq['terminals']
            next_observations = seq['next_observations']
            length = len(observations)
            for i in range(length - 1):
                if i >= 10:  # hopper 20
                    break
                self.add(observations[i], actions[i], next_observations[i], rewards[i], truly_dones[i], discount)
            if k == 20:  # hopper 50
                break
        print(env_name_2, ' load: ', k)

    def convert_DQN(self, dataset):
        self.state = dataset['observations']
        self.action = dataset['actions']
        self.next_state = dataset['next_observations']
        self.reward = dataset['rewards'].reshape(-1, 1)
        self.not_done = 1. - dataset['terminals'].reshape(-1, 1)
        self.size = self.state.shape[0]

    def normalize_states(self, eps=1e-3):
        mean = self.state.mean(0, keepdims=True)
        std = self.state.std(0, keepdims=True) + eps
        self.state = (self.state - mean) / std
        self.next_state = (self.next_state - mean) / std
        print("Normalizing state:", mean, std)
        return mean, std


class MaxStepWrapper(gym.Wrapper):
    def __init__(self, env):
        super(MaxStepWrapper, self).__init__(env)

    @property
    def max_episode_steps(self):
        return self.env._max_episode_steps


class RandomRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_func, std_ratio=1., mean_diff=0.):
        super(RandomRewardWrapper, self).__init__(env)
        self.std_ratio = std_ratio
        self.mean_diff = mean_diff
        self.reward_func = reward_func

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        tensor_state, tensor_action = torch.from_numpy(state).cuda(), torch.from_numpy(action).cuda()
        tensor_state, tensor_action = tensor_state.reshape(1,-1).float(), tensor_action.reshape(1,-1).float()
        new_reward = self.reward_func(tensor_state, tensor_action).item()
        new_reward = new_reward * self.std_ratio + self.mean_diff
        return state, new_reward, done, info


class RandomDimWrapper(gym.Wrapper):
    def __init__(self, env, random_dim=33, random_sample=1e6):
        super(RandomDimWrapper, self).__init__(env)
        self.noise = np.random.randn(int(random_sample) + 50000, random_dim)
        self.random_sample = random_sample
        self.t = 0
        ori_obs_dim = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(ori_obs_dim + random_dim,))

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        new_state = np.concatenate([state, self.noise[self.t, :]])
        self.t += 1
        return new_state, reward, done, info

    def reset(self):
        state = self.env.reset()
        new_state = np.concatenate([state, self.noise[self.t, :]])
        self.t += 1
        return new_state

    @property
    def max_episode_steps(self):
        return self.env._max_episode_steps
