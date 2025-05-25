import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def train_or_run(is_training=True, render=False):
    env = gym.make('CartPole-v1', render_mode='human' if render else None)

    position_bins = np.linspace(-2.4, 2.4, 10)
    velocity_bins = np.linspace(-4, 4, 10)
    angle_bins = np.linspace(-0.2095, 0.2095, 10)
    angular_velocity_bins = np.linspace(-4, 4, 10)

    if is_training:
        q_table = np.zeros((len(position_bins) + 1, len(velocity_bins) + 1,
                            len(angle_bins) + 1, len(angular_velocity_bins) + 1, env.action_space.n))
    else:
        with open('cartpole.pkl', 'rb') as file:
            q_table = pickle.load(file)

    alpha = 0.1  
    gamma = 0.99 
    epsilon = 1.0 
    epsilon_decay = 0.00001  
    rng = np.random.default_rng() 
    rewards_per_episode = []
    episode = 0

    while True:
        state = env.reset()[0] 
        state_pos = np.digitize(state[0], position_bins)
        state_vel = np.digitize(state[1], velocity_bins)
        state_ang = np.digitize(state[2], angle_bins)
        state_ang_vel = np.digitize(state[3], angular_velocity_bins)

        terminated = False
        episode_reward = 0

        while not terminated and episode_reward < 10000:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state_pos, state_vel, state_ang, state_ang_vel, :])

            next_state, reward, terminated, _, _ = env.step(action)
            
            next_state_pos = np.digitize(next_state[0], position_bins)
            next_state_vel = np.digitize(next_state[1], velocity_bins)
            next_state_ang = np.digitize(next_state[2], angle_bins)
            next_state_ang_vel = np.digitize(next_state[3], angular_velocity_bins)

            if is_training:
                best_future_q = np.max(q_table[next_state_pos, next_state_vel, next_state_ang, next_state_ang_vel, :])
                current_q = q_table[state_pos, state_vel, state_ang, state_ang_vel, action]
                q_table[state_pos, state_vel, state_ang, state_ang_vel, action] = current_q + alpha * (
                    reward + gamma * best_future_q - current_q
                )

            state_pos, state_vel, state_ang, state_ang_vel = next_state_pos, next_state_vel, next_state_ang, next_state_ang_vel
            episode_reward += reward

        rewards_per_episode.append(episode_reward)
        average_reward = np.mean(rewards_per_episode[-100:])

        if average_reward > 1000:
            break

        epsilon = max(epsilon - epsilon_decay, 0)
        episode += 1

    env.close()

    if is_training:
        with open('cartpole.pkl', 'wb') as file:
            pickle.dump(q_table, file)

    smoothed_rewards = [np.mean(rewards_per_episode[max(0, t - 100):(t + 1)]) for t in range(episode)]
    plt.plot(smoothed_rewards)
    plt.title("Rewards Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Mean Reward (Last 100 Episodes)")
    plt.savefig('cartpole.png')

if __name__ == '__main__':
    # train_or_run(is_training=True, render=False)
    train_or_run(is_training=False, render=True)
