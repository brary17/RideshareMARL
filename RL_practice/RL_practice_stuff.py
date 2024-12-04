import torch
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast
import time

class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, output_size, is_actor):
        super(NeuralNet, self).__init__()
        actor_layers = [
            torch.nn.Linear(input_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_size),
        ]

        critic_layers = [
            torch.nn.Linear(input_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_size),
        ]

        self.layers = torch.nn.Sequential(*(actor_layers if is_actor else critic_layers))

        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

        self.softmax = torch.nn.Softmax(dim=-1) if is_actor else None

    def forward(self, x):
        x = self.layers(x)
        if self.softmax:
            x = torch.clamp(x, -10, 10)
            x = self.softmax(x)
        return x

class ReplayBuffer:
    def __init__(self, state_dim, device, buffer_size=10_000):
        self.device = device
        self.buffer_size = buffer_size
        self.position = 0
        self.size = 0

        self.states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros(buffer_size, dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32, device=self.device)
        self.next_states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros(buffer_size, dtype=torch.float32, device=self.device)

    def add(self, transition):
        state, action, reward, next_state, done = transition
        
        self.states[self.position].copy_(state if isinstance(state, torch.Tensor) else torch.as_tensor(state, device=self.device))
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state if isinstance(next_state, torch.Tensor) else torch.tensor(next_state, device=self.device)
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
        # indices = np.random.choice(self.size, batch_size, replace=False)
        # return (
        #     self.states[indices],
        #     self.actions[indices],
        #     self.rewards[indices],
        #     self.next_states[indices],
        #     self.dones[indices],
        # )

    def __len__(self):
        return self.size

class Agent:
    def __init__(
            self, 
            state_space_size, 
            action_space_size, 
            buffer_size=10000, 
            batch_size=64, 
            gamma=0.99,
            train_every_n_iters=1,
            **kwargs
        ):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.actor = NeuralNet(state_space_size, action_space_size, is_actor=True).to(self.device)
        self.critic = NeuralNet(state_space_size, 1, is_actor=False).to(self.device)
        self.memory = ReplayBuffer(state_dim=state_space_size, device=self.device, buffer_size=buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        self.action_space_size = action_space_size
        self.scaler = GradScaler()
        self.training_iter = 0
        self.episode_num = 0
        self.steps_since_train = 0
        self.train_every_n_iters = train_every_n_iters
        self.kwargs = kwargs
    
    def start_clock(self, attr_name):
        setattr(self, f"start_time_{attr_name}", time.time())

    def stop_clock(self, attr_name):
        tot_time_ms = time.time() - getattr(self, f"start_time_{attr_name}")
        setattr(self, f"end_time_{attr_name}", time.time())
        setattr(self, f"tot_time_{attr_name}", tot_time_ms)

    def get_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action_probs = self.actor(state)
            return torch.multinomial(action_probs, 1).item()

    def step(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))
        self.steps_since_train += 1

    def save_models(self, test_scores, reward_full=[]):
        now = datetime.now()
        time_name = now.strftime("%m_%d__%H_%M")
        dir_name = f"Agents/agent_{time_name}"
        os.makedirs(dir_name, exist_ok=True)
        with open(f"{dir_name}/agent_info.txt", "w") as eval_f:
            eval_f.write(f"Training Time: {(self.tot_time_train * 1000) : .2f} milliseconds\n")
            eval_f.write(f"Testing Time:  {(self.tot_time_test * 1000): .2f} milliseconds\n")
            for idx, score in test_scores:
                eval_f.write(f"Testing Score Round {idx+1}: {score}\n")
            eval_f.write('\n\nHyperparameters:\n')
        

        torch.save(self.actor, f"{dir_name}/actor_full_model.pth")
        torch.save(self.critic, f"{dir_name}/critic_full_model.pth")


        if reward_full == []: return
        
        cat_data = {
            "hundred": [float(rew[-1] == 100)/(idx+1) for idx, rew in enumerate(reward_full)],
            "neither": [float(rew[-1] != 100 and rew[-1] != -100)/(idx+1) for idx, rew in enumerate(reward_full)],
            "neg_hundred": [float(rew[-1] == -100)/(idx+1) for idx, rew in enumerate(reward_full)]
        }

        categories = list(cat_data.keys())
        data = np.array(list(cat_data.values()))
        data = np.cumsum(data, axis=1)
        time_points = range(data.shape[1])

        data_percentage = data / data.sum(axis=0)

        fig, ax = plt.subplots()
        ax.stackplot(time_points, data_percentage, labels=categories, alpha=0.6)

        ax.set_title("100% Stacked Area Chart")
        ax.set_xlabel("Time")
        ax.set_ylabel("Percentage (%)")
        ax.legend(loc='upper left')

        fig.savefig(f"{dir_name}/stacked_area_training.png")

    def train_models(self):
        self.episode_num += 1
        if (self.episode_num) % self.train_every_n_iters != 0: return
        for _ in range(self.steps_since_train):
            self.learn_from_replay()
        self.steps_since_train = 0

    def learn_from_replay(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        with autocast(device_type=self.device.type):
            next_values_y_hat = self.critic(next_states).squeeze(-1)
            target_values = rewards + (self.gamma * next_values_y_hat * (1-dones))
            values = self.critic(states).squeeze(-1)
            critic_loss = torch.nn.functional.mse_loss(values, target_values)

        self.critic_optim.zero_grad()
        self.scaler.scale(critic_loss).backward()
        self.scaler.step(self.critic_optim)

        with autocast(device_type=self.device.type):
            action_probs = self.actor(states)
            action_log_probs = torch.log(action_probs.gather(1, actions.long().unsqueeze(1)).squeeze())
            # action_log_probs = torch.log(action_probs[range(self.batch_size), actions.int()])
            advantages = (target_values - values).detach() 
            actor_loss = -torch.mean(action_log_probs*advantages)

        self.actor_optim.zero_grad()
        self.scaler.scale(actor_loss).backward()
        self.scaler.step(self.actor_optim)
        
        self.scaler.update()

        self.training_iter += 1
