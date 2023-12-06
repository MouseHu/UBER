import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pcgrad import PCGrad, CAGrad

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class ActorMany(Actor):
    def __init__(self, state_dim, action_dim, max_action, reward_dim):
        super(Actor, self).__init__()
        self.reward_dim = reward_dim
        self.action_dim = action_dim
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        # print(type(action_dim),type(reward_dim))
        self.l3 = nn.Linear(256, action_dim * reward_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = self.l3(a).reshape(-1, self.reward_dim, self.action_dim)
        return self.max_action * torch.tanh(a)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class CriticMany(Critic):
    def __init__(self, state_dim, action_dim, reward_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, reward_dim)

        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, reward_dim)


class TD3_BC(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            alpha=2.5,
            reward_dim=1,
            pcgrad=True
    ):
        self.pcgrad = pcgrad
        self.weight_strategy = CAGrad
        # self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor = ActorMany(state_dim, action_dim, max_action, reward_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        if pcgrad:
            self.actor_optimizer = self.weight_strategy(torch.optim.Adam(self.actor.parameters(), lr=3e-4))
        else:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = CriticMany(state_dim, action_dim, reward_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        if pcgrad:
            self.critic_optimizer = self.weight_strategy(torch.optim.Adam(self.critic.parameters(), lr=3e-4))
        else:
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha
        self.reward_dim = reward_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.total_it = 0

    def sample_actions(self, state):
        self.actor.eval()
        # if type(state) == np.ndarray:
        #     state = torch.tensor(state.reshape(1, -1),dtype=torch.float, device=device)
        actions = self.actor(state).reshape(1, self.reward_dim, self.action_dim)
        return actions

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def set_index(self, t):
        return

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        info = {}
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        noise_shape = action.shape[0], self.reward_dim, action.shape[1]
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn(noise_shape, device=device) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # next_state = next_state.repeat([self.reward_dim, 1]).reshape(-1, self.state_dim)
            next_state = next_state.reshape(-1, 1, self.state_dim).repeat([1, self.reward_dim, 1])
            next_state = next_state.reshape(-1, self.state_dim)
            next_action = next_action.reshape(-1, self.action_dim)
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q1 = target_Q1.reshape(-1, self.reward_dim, self.reward_dim)
            target_Q2 = target_Q2.reshape(-1, self.reward_dim, self.reward_dim)
            target_Q1, target_Q2 = torch.diagonal(target_Q1, 0, 1, 2), torch.diagonal(target_Q2, 0, 1, 2)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        # critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        critic_loss = F.mse_loss(current_Q1, target_Q, reduction='none') + F.mse_loss(current_Q2, target_Q,
                                                                                      reduction='none')
        critic_loss = critic_loss.mean(dim=0)
        # critic_loss[1] = 0
        info['train/critic_loss_0'] = critic_loss[0]
        # info['train/critic_loss_1'] = critic_loss[1]
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        if self.pcgrad:
            self.critic_optimizer.backward(critic_loss)
            critic_loss = critic_loss.mean()
        else:
            critic_loss = critic_loss.mean()
            critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss 
            pi = self.actor(state)
            state = state.reshape(-1, 1, self.state_dim)
            state = state.repeat([1, self.reward_dim, 1]).reshape(-1, self.state_dim)
            Q = self.critic.Q1(state, pi.reshape(-1, self.action_dim))
            Q = Q.reshape(-1, self.reward_dim, self.reward_dim)
            Q = Q.diagonal(0, 1, 2)
            lmbda = self.alpha / Q.abs().mean(dim=0).detach()

            action = action.reshape(-1, 1, self.action_dim).repeat([1, self.reward_dim, 1])
            # action = action.repeat([self.reward_dim, 1]).reshape(-1, self.reward_dim, self.action_dim)
            # actor_loss = -torch.dot(lmbda, Q.mean(dim=0)) / self.reward_dim + F.mse_loss(pi, action)

            bc_loss = F.mse_loss(pi, action, reduction='none').mean(-1)
            bc_loss = bc_loss.mean(dim=0)
            actor_loss = -lmbda * Q.mean(dim=0) + bc_loss
            # actor_loss[1] = 0
            info['train/actor_loss_0'] = actor_loss[0]
            info['train/lmbda'] = lmbda.mean()
            info['train/q_value_0'] = Q.mean(dim=0)[0]
            info['train/bc_loss_0'] = bc_loss[0]
            # info['train/bc_loss'] = bc_loss[0]
            # info['train/actor_loss_1'] = actor_loss[1]
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            if self.pcgrad:
                self.actor_optimizer.backward(actor_loss)
                actor_loss = actor_loss.mean()
            else:
                actor_loss = actor_loss.mean()
                actor_loss.backward()

            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:
            actor_loss = None

        # return critic_loss.mean(), actor_loss.mean() if actor_loss is not None else None
        return critic_loss, actor_loss, info

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        # torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        # torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def partial_load(self, network, state_dict, non_load_names=[]):

        own_state = network.state_dict()
        for name, param in state_dict.items():
            if name not in own_state or name in non_load_names:
                continue
            our_param = own_state[name]
            if our_param.shape != param.shape:
                print(f"{name} shape Mismatch, did not load, our shape:{our_param.shape}, file shape: {param.shape}")
                continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)
            print(f"Successful Load:{name} ")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)


class Contrastive(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            alpha=2.5,
            reward_dim=1,
            pcgrad=True
    ):
        self.pcgrad = pcgrad
        self.weight_strategy = PCGrad
        # self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor = ActorMany(state_dim, action_dim, max_action, reward_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        if pcgrad:
            self.actor_optimizer = self.weight_strategy(torch.optim.Adam(self.actor.parameters(), lr=3e-4))
        else:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = CriticMany(state_dim, action_dim, reward_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        if pcgrad:
            self.critic_optimizer = self.weight_strategy(torch.optim.Adam(self.critic.parameters(), lr=3e-4))
        else:
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha
        self.reward_dim = reward_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.total_it = 0

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        info = {}
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        noise_shape = action.shape[0], self.reward_dim, action.shape[1]
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn(noise_shape) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip).cuda()

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # next_state = next_state.repeat([self.reward_dim, 1]).reshape(-1, self.state_dim)
            next_state = next_state.reshape(-1, 1, self.state_dim).repeat([1, self.reward_dim, 1])
            next_state = next_state.reshape(-1, self.state_dim)
            next_action = next_action.reshape(-1, self.action_dim)
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q1 = target_Q1.reshape(-1, self.reward_dim, self.reward_dim)
            target_Q2 = target_Q2.reshape(-1, self.reward_dim, self.reward_dim)
            target_Q1, target_Q2 = torch.diagonal(target_Q1, 0, 1, 2), torch.diagonal(target_Q2, 0, 1, 2)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        # critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        critic_loss = F.mse_loss(current_Q1, target_Q, reduction='none') + F.mse_loss(current_Q2, target_Q,
                                                                                      reduction='none')
        critic_loss = critic_loss.mean(dim=0)
        # critic_loss[1] = 0
        info['train/critic_loss_0'] = critic_loss[0]
        # info['train/critic_loss_1'] = critic_loss[1]
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        if self.pcgrad:
            self.critic_optimizer.backward(critic_loss)
            critic_loss = critic_loss.mean()
        else:
            critic_loss = critic_loss.mean()
            critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            pi = self.actor(state)
            state = state.reshape(-1, 1, self.state_dim)
            state = state.repeat([1, self.reward_dim, 1]).reshape(-1, self.state_dim)
            Q = self.critic.Q1(state, pi.reshape(-1, self.action_dim))
            Q = Q.reshape(-1, self.reward_dim, self.reward_dim)
            Q = Q.diagonal(0, 1, 2)
            lmbda = self.alpha / Q.abs().mean(dim=0).detach()

            action = action.reshape(-1, 1, self.action_dim).repeat([1, self.reward_dim, 1])
            # action = action.repeat([self.reward_dim, 1]).reshape(-1, self.reward_dim, self.action_dim)
            # actor_loss = -torch.dot(lmbda, Q.mean(dim=0)) / self.reward_dim + F.mse_loss(pi, action)

            bc_loss = F.mse_loss(pi, action, reduction='none')
            bc_loss = bc_loss.mean(dim=0).mean(dim=1)
            actor_loss = -lmbda * Q.mean(dim=0) + bc_loss
            # actor_loss[1] = 0
            info['train/actor_loss_0'] = actor_loss[0]

            # info['train/actor_loss_1'] = actor_loss[1]
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            if self.pcgrad:
                self.actor_optimizer.backward(actor_loss)
                actor_loss = actor_loss.mean()
            else:
                actor_loss = actor_loss.mean()
                actor_loss.backward()

            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:
            actor_loss = None

        # return critic_loss.mean(), actor_loss.mean() if actor_loss is not None else None
        return critic_loss, actor_loss, info

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        # torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        # torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
