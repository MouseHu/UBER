import numpy as np
import torch
import gym
import argparse
import os
import d4rl

import utils
import TD3_BC
import TD3_Ensemble
import CONTRAST
from reward_randomization import reward_randomization, reward_randomization_nn, reward_randomization_informed, \
    reward_randomization_nn_best

from torch.utils.tensorboard import SummaryWriter
from analysis import analysis

log_dir = './new_logs/pretrain/'


# Runs policy for X episodes and returns D4RL score
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            state = (np.array(state).reshape(1, -1) - mean) / std
            action = policy.select_action(state)
            # print(action.shape)
            action = action[:action_dim]
            if len(action.shape) > 1:
                action = action.squeeze()[0]  # evaluate the first actor
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {d4rl_score:.3f}")
    print("---------------------------------------")
    return avg_reward, d4rl_score


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--policy", default="TD3_BC")  # Policy name
    parser.add_argument("--comment", default="")  # Policy name
    parser.add_argument("--env", default="hopper-medium-v0")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=5e5, type=int)  # Max time steps to run environment
    parser.add_argument("--save_model", action="store_true", default=True)  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    # TD3
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    # TD3 + BC
    parser.add_argument("--alpha", default=2.5, type=float)
    parser.add_argument("--normalize", default=False)
    parser.add_argument("--pcgrad", default=False)
    parser.add_argument("--train_type", default="oracle",
                        choices=["oracle", "r4", "r4_best", "contrast", "ensemble", "bc", "avg"])

    # reward randomization
    parser.add_argument("--reward_dim", default=256, type=int)
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.train_type}_{args.comment}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    if not os.path.exists(f"{log_dir}"):
        os.makedirs(f"{log_dir}")

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    reward_dim = args.reward_dim

    batch_size = args.batch_size
    max_action = float(env.action_space.high[0])
    train_type = args.train_type
    if train_type == "oracle" or train_type == "r4_best":
        num_head = 1
    else:
        num_head = reward_dim
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "reward_dim": num_head,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        # TD3
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_freq,
        # TD3 + BC
        "alpha": args.alpha,
        "pcgrad": args.pcgrad
    }

    # Initialize writer
    writer = SummaryWriter(f"{log_dir}/{file_name}/")

    # Initialize policy
    # policy = TD3_BC_origin.TD3_BC(**kwargs)

    if train_type == "bc":
        kwargs["alpha"] = 0  # no gradient from Q
    if train_type == "contrast":
        policy = CONTRAST.Contrastive(**kwargs)
    elif train_type == "ensemble":
        policy = TD3_Ensemble.TD3_Ensemble(**kwargs)
    else:
        policy = TD3_BC.TD3_BC(**kwargs)
    # if args.load_model != "":
    #     policy_file = file_name if args.load_model == "default" else args.load_model
    #     policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, reward_dim)
    # replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env), load_reward=True)
    load_reward = train_type != "r4"
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env), load_reward)

    # reward_randomization_informed(replay_buffer, state_dim, action_dim, reward_dim, batch_size * 16)
    if args.normalize:
        mean, std = replay_buffer.normalize_states()
    else:
        mean, std = 0, 1
    if "r4" in train_type:
        if args.load_model != "":
            policy_file = args.load_model
        else:
            policy_file = None
        if "best" in train_type:
            reward_randomization_nn_best(replay_buffer, state_dim, action_dim, reward_dim, batch_size * 16,
                                         load_model=policy_file)
        else:
            # reward_randomization_vision(replay_buffer, state_dim, action_dim, reward_dim, batch_size * 16,
            #                         load_model=policy_file)
            reward_randomization_nn(replay_buffer, state_dim, action_dim, reward_dim, batch_size * 16,
                                    load_model=policy_file)
            analysis(replay_buffer)
    if train_type == "avg":
        replay_buffer.reward = np.ones_like(replay_buffer.reward) * np.mean(replay_buffer.reward)
    evaluations = []
    if train_type != "contrast":
        avg_reward, score = eval_policy(policy, args.env, args.seed, mean, std)
        evaluations.append(score)
        writer.add_scalar("eval/d4rl_score", score, 0)
        writer.add_scalar("eval/avg_reward", avg_reward, 0)
    print("Begin Training ...")
    for t in range(int(args.max_timesteps)):
        critic_loss, actor_loss, info = policy.train(replay_buffer, args.batch_size)
        # print(info)
        for k, v in info.items():
            writer.add_scalar(k, v, t)
        writer.add_scalar("train/critic_loss", critic_loss, t)
        if actor_loss:
            writer.add_scalar("train/actor_loss", actor_loss, t)
        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            if train_type != "contrast":
                print(f"Time steps: {t + 1}")
                avg_reward, score = eval_policy(policy, args.env, args.seed, mean, std)
                evaluations.append(score)
                writer.add_scalar("eval/d4rl_score", score, t)
                writer.add_scalar("eval/avg_reward", avg_reward, t)
                np.save(f"./results/{file_name}", evaluations)
            if args.save_model:
                policy.save(f"./models/{file_name}")
                print("Model Saved")
    print("Finish Training")
