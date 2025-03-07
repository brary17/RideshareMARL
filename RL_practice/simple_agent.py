import torch
import os
from datetime import datetime
import gymnasium as gym
from tqdm import tqdm
import random
import wandb
import os


os.environ["WANDB_LIVE_MODE"] = "true"
os.environ["WANDB_SYNC_INTERVAL"] = "1"
os.environ["WANDB_DISABLE_LOG_COMPRESSION"] = "true"


class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        layers = [
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, output_size),
        ]
        self.layers = torch.nn.Sequential(*layers)

        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.layers(x)
        return x

class Actor(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()
        self.model = NeuralNet(input_size, output_size)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.model(x))

class Critic(torch.nn.Module):
    def __init__(self, input_size):
        super(Critic, self).__init__()
        self.model = NeuralNet(input_size, 1)

    def forward(self, x):
        return self.model(x)


class ReplayBuffer:
    def __init__(self, state_dim, device, buffer_size=10_000):
        self.device = device
        self.buffer_size = buffer_size
        self.position = 0
        self.size = 0
        self.states = torch.zeros((buffer_size, state_dim), dtype=torch.float16, device=self.device)
        self.actions = torch.zeros(buffer_size, dtype=torch.float16, device=self.device)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float16, device=self.device)
        self.next_states = torch.zeros((buffer_size, state_dim), dtype=torch.float16, device=self.device)
        self.dones = torch.zeros(buffer_size, dtype=torch.float16, device=self.device)

    def add(self, transition):
        state, action, reward, next_state, done = transition
        self.states[self.position].copy_(torch.tensor(state, dtype=torch.float16, device=self.device))
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position].copy_(torch.tensor(next_state, dtype=torch.float16, device=self.device))
        self.dones[self.position] = done

        self.position = (self.position + 1) % self.buffer_size
        self.size = min(self.size+1, self.buffer_size)

    def sample(self, batch_size):
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.states.index_select(0, indices),
            self.actions.index_select(0, indices),
            self.rewards.index_select(0, indices),
            self.next_states.index_select(0, indices),
            self.dones.index_select(0, indices)
        )
        
    def __len__(self):
        return self.size


class Agent:
    def __init__(
            self, 
            state_space_size, 
            action_space_size, 
            wandb_stuff,
            buffer_size=10000, 
            batch_size=64, 
            gamma=0.99,
            entropy_coeff=.01,
            **kwargs
        ):
        self.wandb = wandb_stuff
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.actor = torch.compile(Actor(state_space_size, action_space_size)).to(self.device).half()
        self.critic = torch.compile(Critic(state_space_size)).to(self.device).half()
        self.memory = ReplayBuffer(state_dim=state_space_size, device=self.device, buffer_size=buffer_size)

        self.entropy_coeff = entropy_coeff
        self.batch_size = batch_size
        self.gamma = gamma
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        self.ls_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optim, step_size=100, gamma=.99)
        self.action_space_size = action_space_size
        self.step_count = 0

    def get_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float16, device=self.device).unsqueeze(0)
            action_probs = self.actor(state)
            if torch.isnan(action_probs).any():
                action_probs = torch.ones_like(action_probs) / action_probs.shape[-1]
            return torch.multinomial(action_probs, 1).item()

    def step(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    def save_models(self, scores):
        now = datetime.now()
        time_name = now.strftime("%m_%d__%H_%M")
        dir_name = f"Agents/agent_{scores: .2f}_{time_name}"
        os.makedirs(dir_name, exist_ok=True)
        torch.save(self.actor, f"{dir_name}/actor_full_model.pth")
        torch.save(self.critic, f"{dir_name}/critic_full_model.pth")

    def train_models(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Critic
        with torch.no_grad():
            next_values_y_hat = self.critic(next_states).squeeze(-1)
            target_values = rewards + (self.gamma * next_values_y_hat * (1-dones))
        values = self.critic(states).squeeze(-1)
        critic_loss = torch.nn.functional.mse_loss(values, target_values)

        # Agent
        action_probs = self.actor(states)
        action_log_probs = torch.log(torch.clamp(action_probs.gather(1, actions.long().unsqueeze(1)), min=1e-6)).squeeze()
        advantages = (target_values - values).detach() 
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        entropy = -torch.sum(action_probs * torch.log(torch.clamp(action_probs, min=1e-6)), dim=-1)
        actor_loss = -torch.mean(action_log_probs*advantages + self.entropy_coeff*entropy)

        # Critic Optimization
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Actor Optimization
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if (self.step_count + 1) % 10 == 0:
            self.wandb.log({"actor loss": actor_loss, "critic loss": critic_loss}, commit=True)
            self.step_count = 0
        else:
            self.step_count += 1


def main():
    env = gym.make("LunarLander-v3")
    hyper_parameters = {
        "state_space_size": len(env.observation_space.high),
        "action_space_size": env.action_space.n, 
        "batch_size": 2, 
        "train_every_n_iters": 2,
        # "total_episodes": 10000,
        "total_episodes": 100,
    }
    wandb.init(project="RL Practice", name="Lunar Landar Tests")

    agent = Agent(**{"wandb_stuff": wandb}, **hyper_parameters)
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
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.step(state, action, reward, next_state, (terminated or truncated))
            rewards_ep.append(reward)
            state = next_state

        agent.train_models()
        reward_full.append(rewards_ep)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        wandb.log({
            "epoch": ep, 
            "Ave Reward": sum(rewards_ep)/len(rewards_ep),
            "Max Reward": max(rewards_ep),
            "Total Reward": sum(rewards_ep),
        }, commit=True)

    env.close()
    wandb.finish()

    want_visual = False
    if want_visual:
        env = gym.make('LunarLander-v3', render_mode='human')
    else:
        env = gym.make('LunarLander-v3', render_mode="rgb_array")

    env = gym.wrappers.RecordVideo(env, "video_output")
    test_episodes = 10
    test_rewards = []
    for ep in range(test_episodes):
        state, _ = env.reset()
        terminated, truncated = False, False
        ep_rewards = []
        while (not terminated) and (not truncated):
            action = agent.get_action(state)
            nxt_stp, rwd, terminated, truncated, _ = env.step(action)
            ep_rewards.append(rwd)
            if truncated or terminated:
                break
            state = nxt_stp
        test_rewards.append(ep_rewards)

    tot_rewards = [sum(rewa) for rewa in test_rewards]
    ave_tot_rewards = sum(tot_rewards) / len(tot_rewards)
    agent.save_models(ave_tot_rewards)
    env.close()

if __name__ == "__main__":
    main()