import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class GradientSurgery(nn.Module):
    def __init__(self, gradient_scale):
        super().__init__()
        self.gradient_scale = gradient_scale

    def forward(self, input):
        stopped_input = input.detach()
        new_input = self.gradient_scale * input + (1 - self.gradient_scale) * stopped_input
        return new_input


# HIDDEN_DIM1, HIDDEN_DIM2, HIDDEN_DIM3 = 50, 50, 50


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, gradient_scale=1.):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        # self.bn1 = nn.InstanceNorm1d(256)
        # self.bn2 = nn.InstanceNorm1d(256)
        self.bn1 = nn.Identity()
        self.bn2 = nn.Identity()
        self.grad_clip = GradientSurgery(gradient_scale)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.bn1(self.l1(state)))
        a = F.relu(self.bn2(self.l2(a)))
        a = self.grad_clip(a)
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, gradient_scale=1):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)

        self.l2 = nn.Linear(256, 256)

        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

        # self.bn1 = nn.InstanceNorm1d(256)
        # self.bn2 = nn.InstanceNorm1d(256)
        # self.bn4 = nn.InstanceNorm1d(256)
        # self.bn5 = nn.InstanceNorm1d(256)

        self.bn1 = nn.Identity()
        self.bn2 = nn.Identity()
        self.bn4 = nn.Identity()
        self.bn5 = nn.Identity()

        self.grad_clip = GradientSurgery(gradient_scale)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.bn1(self.l1(sa)))
        q1 = F.relu(self.bn2(self.l2(q1)))
        q1 = self.grad_clip(q1)
        q1 = self.l3(q1)

        q2 = F.relu(self.bn4(self.l4(sa)))
        q2 = F.relu(self.bn5(self.l5(q2)))
        q2 = self.grad_clip(q2)
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3_PEX(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            critic_tau=0.01,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            grad_clip=1,
            beta=2.5,
            omega=1.,
            guidance_policy=None,
            reward_dim=0,
            alpha=10,
    ):

        self.actor = Actor(state_dim, action_dim, max_action, grad_clip).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim, grad_clip).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.critic_tau = critic_tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.omega = omega

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.beta = beta
        self.total_it = 0

        self.guidance_policy = guidance_policy
        self.alpha = alpha
        # if self.guidance_policy is not None:
        #     self.guidance_policy.actor.eval()

    def select_action(self, state):
        state = torch.tensor(state.reshape(1, -1), dtype=torch.float, device=device)
        self.actor.eval()
        self.critic.eval()

        if self.guidance_policy is not None:
            if type(self.guidance_policy) == list:
                guidance_action = []
                for policy in self.guidance_policy:
                    guidance_action.append(policy.sample_actions(state).reshape(1, -1, self.action_dim))
                guidance_action = torch.cat(guidance_action, dim=1)
            else:
                guidance_action = self.guidance_policy.sample_actions(state)
            # guidance_action = self.guidance_policy.actor(state).reshape(1, self.reward_dim, self.action_dim)
            if type(guidance_action) == np.ndarray:
                guidance_action = torch.tensor(guidance_action, dtype=torch.float, device=device).reshape(1, self.reward_dim,
                                                                                            self.action_dim)
            # print(guidance_action.shape)
            self_action = self.actor(state).reshape(1, 1, self.action_dim)
            total_action = torch.concat([guidance_action, self_action], dim=1)
            total_action = total_action.reshape(-1, self.action_dim)
            # (state* num_actions) * action_dim
            total_state = state.repeat(self.reward_dim + 1, 1)
            guidance_Q1, guidance_Q2 = self.critic_target(total_state, total_action)
            guidance_Q = torch.min(guidance_Q1, guidance_Q2)
            guidance_Q = self.alpha * guidance_Q.reshape(-1)
            # this may be improved
            action_probs = F.softmax(guidance_Q, dim=0).detach().cpu().numpy()
            action_ind = np.random.choice(np.arange(self.reward_dim + 1), p=action_probs)
            action = total_action[action_ind, :].detach().cpu().numpy()
        else:
            action = self.actor(state).cpu().data.numpy().flatten()

        self.actor.train()
        self.critic.train()

        return action

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        info = {}
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1_ori, target_Q2_ori = self.critic_target(state, action)
            target_Q_ori = torch.min(target_Q1_ori, target_Q2_ori)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q
            target_Q = self.omega * target_Q + (1 - self.omega) * target_Q_ori

        current_Q1, current_Q2 = self.critic(state, action)
        info['train/q_value_0'] = current_Q1.mean()
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            Q = self.critic.Q1(state, self.actor(state))
            lmbda = self.beta / Q.abs().mean(dim=0).detach()
            actor_loss = -lmbda * self.critic.Q1(state, self.actor(state)).mean()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.critic_tau * param.data + (1 - self.critic_tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:
            actor_loss = None

        return critic_loss, actor_loss, info

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

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")

        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load_actor(self, filename):
        self.partial_load(self.actor, torch.load(filename + "_actor"),
                          non_load_names=["l6.weight", "l6.bias", "l3.weight", "l3.bias"])
        # self.actor.load_state_dict(torch.load(filename + "_actor"))
        # self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

    def load_critic(self, filename):
        self.partial_load(self.critic, torch.load(filename + "_critic"),
                          non_load_names=["l6.weight", "l6.bias", "l3.weight", "l3.bias"])
        # self.critic.load_state_dict(torch.load(filename + "_critic"))
        # self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

    def load(self, filename):
        self.load_critic(filename)
        self.load_actor(filename)

    def set_guidance_policy(self, guidance_policy):
        self.guidance_policy = guidance_policy
        if type(self.guidance_policy) == list:
            self.reward_dim = 0
            for policy in self.guidance_policy:
                self.reward_dim += policy.reward_dim
        else:
            self.reward_dim = guidance_policy.reward_dim
