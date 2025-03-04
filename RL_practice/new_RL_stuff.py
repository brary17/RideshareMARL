import gymnasium as gym
from tqdm import tqdm
import random
import wandb
import os
from simple_agent import Agent

os.environ["WANDB_LIVE_MODE"] = "true"
os.environ["WANDB_SYNC_INTERVAL"] = "1"
os.environ["WANDB_DISABLE_LOG_COMPRESSION"] = "true"

def main():
    env = gym.make("LunarLander-v3")

    hyper_parameters = {
        "state_space_size": len(env.observation_space.high),
        "action_space_size": env.action_space.n, 
        "batch_size": 2, 
        "train_every_n_iters": 2,
        "total_episodes": 10000,
    }

    wandb.init(project="RL Practice", name="Lunar Landar Tests")

    agent = Agent(**hyper_parameters)
    reward_full = []
    min_epsilon = 0.01
    epsilon_decay = 0.995
    epsilon = 1

    for ep in tqdm(range(hyper_parameters['total_episodes'])):
        state, _ = env.reset()
        terminated, truncated = False, False
        rewards_ep = []
        while not (terminated or truncated):
            action = agent.get_action(state)
            if random.random() < epsilon:
                action = random.randint(0, 3)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.step(state, action, reward, next_state, (terminated or truncated))
            rewards_ep.append(reward)
            state = next_state

        agent.train_models()
        reward_full.append(rewards_ep)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        wandb.log({
            "epoch": ep, 
            "Ave Reward": sum(rewards_ep)/len(rewards_ep),
            "Total Reward": sum(rewards_ep),
        }, commit=True)

    env.close()
    wandb.finish()

    want_visual = True
    if want_visual:
        env = gym.make('LunarLander-v3', render_mode='human')
    else:
        env = gym.make('LunarLander-v3')

    test_episodes = 10
    test_rewards = []
    for ep in range(test_episodes):
        state, _ = env.reset()
        terminated, truncated = False, False
        ep_rewards = []
        while (not terminated) and (not truncated):
            action = agent.get_action(state)
            nxt_stp, rwd, terminated, truncated, info = env.step(action)
            ep_rewards.append(rwd)
            if truncated or terminated:
                break
            state = nxt_stp
        test_rewards.append(ep_rewards[-1])

    tot_rewards = [sum(rewa) for rewa in test_rewards]
    ave_tot_rewards = sum(tot_rewards) / len(tot_rewards)

    agent.save_models(ave_tot_rewards)

    env.close()

if __name__ == "__main__":
    main()