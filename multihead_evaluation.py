# evaluate all headers from a policy network


import numpy as np
import torch
import gym
import argparse
import os
import utils
import TD3_BC
from torch.utils.tensorboard import SummaryWriter
from metaworld_env.metaworld_env import make_metaworld_env
import imageio
import jax_learner
mean = 0
std = 1

log_dir = './new_logs/multihead'

# mean = np.array([-0.11002816, 0.15680763, 0.10378725, 0.14687687, 0.07839588, -0.20106335,
#                  -0.08224171, -0.2802395, 4.463403, -0.07580097, -0.09260748, 0.41871706,
#                  -0.41171676, 0.11628567, -0.06000552, -0.09738238, -0.14540626])
# std = np.array([0.10956703, 0.6119863, 0.49235544, 0.44962165, 0.39817896, 0.4823394,
#                 0.30695462, 0.26474255, 1.9024047, 0.939795, 1.625154, 14.427593,
#                 11.996738, 11.985555, 12.159913, 8.127248, 6.419199])


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, step=0, eval_episodes=10):
    global mean, std
    # eval_env = gym.make(env_name)
    eval_env = create_env(env_name)
    eval_env = utils.MaxStepWrapper(eval_env)
    # eval_env = utils.RandomDimWrapper(eval_env, noisy_dim)
    eval_env.seed(seed + 100)
    # policy.actor.eval()
    # policy.critic.eval()
    avg_reward = 0.
    for i in range(eval_episodes):

        state, done = eval_env.reset(), False
        # images = []
        while not done:
            state = (np.array(state).reshape(1, -1) - mean) / std
            state = np.array(state).reshape(1, -1)
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)

            # if i == 0:
            #     images.append(eval_env.render(camera_name='corner'))
            # ('camera_name should be one of ', 'corner3, corner, corner2, topview, gripperPOV, behindGripper')
            avg_reward += reward
        # if i == 0:
        #     imageio.mimsave(f"./gif/{env_name}_{seed}_{step + 1}.gif", images)
    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f} Step:{step}")
    print("---------------------------------------")
    return avg_reward


def create_env(env_name):
    if "metaworld" in env_name:
        env_id = int(env_name[-1])
        env = make_metaworld_env(env_id)
    else:
        env = gym.make(env_name)
    return env


class ExtractedPolicy:
    def __init__(self, policy, reward_dim=100, action_dim=3, index=0):
        self.policy = policy
        self.reward_dim = reward_dim
        self.action_dim = action_dim
        self.index = index

    def set_index(self,index):
        assert 0<=index<self.reward_dim
        self.index= index

    @property
    def actor(self):
        return self.policy.actor

    @property
    def critic(self):
        return self.policy.critic

    def select_action(self, state):
        guidance_action = self.policy.sample_actions(state)
        # print(guidance_action.shape)
        guidance_action = guidance_action.reshape(self.reward_dim,
                                                           self.action_dim)
        return guidance_action[self.index]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--comment", default="")  # Policy name (TD3, DDPG or OurDDPG)
    # parser.add_argument("--env", default="HalfCheetah-v2")  # OpenAI gym environment name
    # parser.add_argument("--env", default="halfcheetah-medium-v0")  # OpenAI gym environment name
    parser.add_argument("--env", default="halfcheetah-medium-v0")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--critic_tau", default=0.01)  # Target network update rate
    parser.add_argument("--critic_target_freq", default=0.01)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--grad_clip", default=1, type=float)  # Range to clip gradient in representation layer
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--load_actor", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--load_critic", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--guidance_type", default="TD3_BC")

    parser.add_argument("--reward_dim", default=100, type=int)
    parser.add_argument("--omega", default=1., type=float)  # omega for relaxation methods
    args = parser.parse_args()

    file_name = f"finetune_{args.policy}_{args.env}_{args.comment}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")
    if not os.path.exists(f"{log_dir}"):
        os.makedirs(f"{log_dir}")
    # env = gym.make(args.env)
    env = create_env(args.env)
    # env = utils.RandomDimWrapper(env, noisy_dim)
    env = utils.MaxStepWrapper(env)
    # mean, std = utils.GAME_MEAN[args.env], utils.GAME_STD[args.env]
    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "critic_tau": args.critic_tau,
    }

    # Initialize policy
    # Target policy smoothing is scaled wrt the action scale
    kwargs["policy_noise"] = args.policy_noise * max_action
    kwargs["noise_clip"] = args.noise_clip * max_action
    kwargs["grad_clip"] = args.grad_clip
    kwargs["policy_freq"] = args.policy_freq
    kwargs["reward_dim"] = args.reward_dim
    del kwargs['critic_tau']
    del kwargs['grad_clip']
    # Initialize writer
    # writer = SummaryWriter(f"{log_dir}/{file_name}/")
    if args.guidance_type == "TD3_BC":
        policy = TD3_BC.TD3_BC(**kwargs)
        policy.partial_load(policy.actor, torch.load(f"./models/{args.load_actor}_actor"))
    elif args.guidance_type == "JAX" or args.guidance_type == "IQL":
        policy = jax_learner.Learner(args.seed, args.reward_dim,
                                              env.observation_space.sample()[np.newaxis],
                                              env.action_space.sample()[np.newaxis], )
        policy.load(args.load_actor)
    else:
        raise NotImplementedError
    # if args.load_actor != "" or args.load_critic != "":
    #     # assert args.load_critic != "", "Actor and Critic must be load simutanously"
    #     # assert args.load_actor != "", "Actor and Critic must be load simutanously"
    #     if args.load_actor != "":
    #         policy.partial_load(policy.actor, torch.load(f"./models/{args.load_actor}_actor"))
    #     if args.load_critic:
    #         policy.partial_load(policy.critic, torch.load(f"./models/{args.load_critic}_critic"))
    # elif args.load_model != "":
    #     policy_file = args.load_model
    #     policy.load(f"./models/{policy_file}")
    # else:
    #     print("Warning: evaluation must specify pretrained model!")
    # load model

    # Evaluate untrained policy
    evaluations = []
    state, done = env.reset(), False
    # state = (np.array(state).reshape(1, -1) - mean) / std
    state = np.array(state).reshape(1, -1)

    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    # extracted_policy = ExtractedPolicy(policy, args.reward_dim, action_dim)
    extracted_policy = policy

    for t in range(args.reward_dim):
        extracted_policy.set_index(t)
        avg_reward = eval_policy(extracted_policy, args.env, args.seed, t)
        evaluations.append(avg_reward)
        # writer.add_scalar("eval/episode_reward", avg_reward, t)
        # np.save(f"./results/{file_name}", evaluations)
        # if args.save_model: policy.save(f"./models/{file_name}")

    evaluations.sort()
    print(evaluations)
