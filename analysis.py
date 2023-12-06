import numpy as np


def correlation(random_reward, true_reward):
    correlations = []
    for i in range(random_reward.shape[1]):
        true_reward_mean = (true_reward - np.mean(true_reward)).squeeze()
        random_reward_mean = random_reward[:, i] - np.mean(random_reward[:, i])
        covariance = np.mean(true_reward_mean * random_reward_mean)
        true_reward_var = np.mean(true_reward_mean * true_reward_mean)
        random_reward_var = np.mean(random_reward_mean * random_reward_mean)
        correlation = covariance / np.sqrt(true_reward_var * random_reward_var)
        correlations.append(correlation)

    # print(correlations)
    print("Variance, Mean, Max, Min")
    print(np.var(correlations), np.mean(correlations), np.max(correlations), np.min(correlations))
    return correlations


def analysis(replay_buffer):
    random_reward = replay_buffer.reward
    true_reward = replay_buffer.raw_reward
    ctrl_reward = replay_buffer.ctrl_reward.reshape(-1,1)
    print(ctrl_reward.shape)
    # correlation analysis
    print("Correlation total")
    correlation(random_reward, true_reward)
    # print("Correlation control")
    #
    # correlation(random_reward, ctrl_reward)
    # print("Correlation others")
    # correlation(random_reward, true_reward + ctrl_reward)
    # projection analysis
    projection = np.linalg.inv(np.matmul(random_reward.T, random_reward))
    projection = np.matmul(random_reward, projection)
    projected_reward = np.matmul(random_reward.T, true_reward)
    projected_reward = np.matmul(projection, projected_reward)
    relative_error = np.linalg.norm(true_reward - projected_reward) / np.linalg.norm(true_reward)
    print("projection error", relative_error)
