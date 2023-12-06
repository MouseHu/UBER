import numpy as np
import torch
import gym
import argparse
import os
import utils
import TD3
from torch.utils.tensorboard import SummaryWriter
from metaworld_env.metaworld_env import make_metaworld_env
import imageio

mean = 0
std = 1
noisy_dim = 83

log_dir = './new_logs/cup/'


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
    policy.actor.eval()
    policy.critic.eval()
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
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    policy.actor.train()
    policy.critic.train()
    return avg_reward


def create_env(env_name):
    if "metaworld" in env_name:
        env_id = int(env_name[-1])
        env = make_metaworld_env(env_id)
    else:
        env = gym.make(env_name)
    return env


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
    parser.add_argument("--max_timesteps", default=3e6, type=int)  # Max time steps to run environment
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
    policy = TD3.TD3(**kwargs)
    # Initialize writer
    writer = SummaryWriter(f"{log_dir}/{file_name}/")

    if args.load_actor != "" or args.load_critic != "":
        # assert args.load_critic != "", "Actor and Critic must be load simutanously"
        # assert args.load_actor != "", "Actor and Critic must be load simutanously"
        if args.load_actor != "":
            policy.load_actor(f"./models/{args.load_actor}")
        if args.load_critic:
            policy.load_critic(f"./models/{args.load_critic}")
    elif args.load_model != "":
        policy_file = args.load_model
        policy.load(f"./models/{policy_file}")
    else:
        print("Warning: finetune must specify pretrained model!")
    # load model
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, online_buffer=True)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, args.seed)]
    writer.add_scalar("eval/episode_reward", evaluations[-1], 0)

    state, done = env.reset(), False
    # state = (np.array(state).reshape(1, -1) - mean) / std
    state = np.array(state).reshape(1, -1)

    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1
        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        next_state = (np.array(next_state).reshape(1, -1) - mean) / std
        done_bool = float(done) if episode_timesteps < env.max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            critic_loss, actor_loss, info = policy.train(replay_buffer, args.batch_size)
            if t % 100 == 0:
                writer.add_scalar("train/critic_loss", critic_loss, t)
                if actor_loss:
                    writer.add_scalar("train/actor_loss", actor_loss, t)
                for k, v in info.items():
                    writer.add_scalar(k, v, t)
        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            writer.add_scalar("train/episode_timesteps", episode_timesteps, t)
            writer.add_scalar("train/episode_reward", episode_reward, t)

            state, done = env.reset(), False
            state = (np.array(state).reshape(1, -1) - mean) / std

            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            avg_reward = eval_policy(policy, args.env, args.seed, t)
            evaluations.append(avg_reward)
            writer.add_scalar("eval/episode_reward", avg_reward, t)
            np.save(f"./results/{file_name}", evaluations)
            if args.save_model: policy.save(f"./models/{file_name}")
