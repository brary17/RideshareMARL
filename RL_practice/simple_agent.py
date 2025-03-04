import torch
import os
from datetime import datetime

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
            buffer_size=10000, 
            batch_size=64, 
            gamma=0.99,
            entropy_coeff=.01,
            **kwargs
        ):
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
        action_probs, _ = self.actor(states)
        action_log_probs = torch.log(
            torch.clamp(action_probs.gather(1, actions.long().unsqueeze(1)).squeeze(), min=1e-6)
        )
        advantages = (target_values - values).detach() 
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        entropy = -torch.sum(action_probs * torch.log(torch.clamp(action_probs, min=1e-6)), dim=-1)
        actor_loss = -torch.mean(action_log_probs*advantages + self.entropy_coeff*entropy)

        if self.step_count == 0:
            self.wandb.log({"actor loss": actor_loss, "critic loss": critic_loss}, commit=True)
            self.step_count = 0
        else:
            self.step_count += 1

        # Critic Optimization
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Actor Optimization
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
