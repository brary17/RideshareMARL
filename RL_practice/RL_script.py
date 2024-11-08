import gymnasium as gym
import time
from RL_practice_stuff import Agent

env = gym.make("LunarLander-v3")

state_space_size = len(env.observation_space.high)
action_space_size = env.action_space.n
agent = Agent(state_space_size, action_space_size, batch_size=64)

reward_full = []
total_episodes = 25
actor_losses = []
critic_losses = []

start_time_sec = time.time()
for ep in range(total_episodes):
    state, _ = env.reset()
    terminated, truncated = False, False
    rewards = []
    while (not terminated) and (not truncated):
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        agent.step(state, action, reward, next_state, (terminated or truncated))
        rewards.append(reward)
        if truncated or terminated:
            break
        else:
            state = next_state

    reward_full.append(rewards)

    # if ep+1 % 10 == 0:
    #     print(f"Episode: {ep+1}")

env.close()
train_time_sec = time.time() - start_time_sec
print(f"Training time (ms): {(train_time_sec * 1000): .2f}")
print(f"Training time (min): {(train_time_sec / 60): .2f}")


want_visual = False
if want_visual:
    env = gym.make('LunarLander-v3', render_mode='human')
else:
    env = gym.make('LunarLander-v3')

test_episodes = 10
start_time_sec = time.time()
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
    ep_rewards.append(rwd)

env.close()
test_time_sec = time.time() - start_time_sec
print(f"Testing time: {(test_time_sec * 1000): .2f}")
# agent.save_models(
#     train_time=train_time_sec, 
#     test_time=test_time_sec, 
#     test_scores=[ep[-1] for ep in test_rewards], 
#     reward_full=reward_full
# )