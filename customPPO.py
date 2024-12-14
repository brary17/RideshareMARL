# Based entirely on https://github.com/seolhokim/Mujoco-Pytorch/

from abc import *

import torch
import torch.nn as nn
import torch.optim as optim

from utils.utils import ReplayBuffer
from actor import Actor
from critic import Critic


class PPO(nn.Module):
    def __init__(
            self, 
            model_type,
            state_dim,
            action_dim,
            writer,
            device,
            args,
            **kwargs
        ):
        super(PPO, self).__init__()
        self.writer = writer
        self.device = device
        self.episode_num = -1

        self.actor = Actor(
            state_dim=state_dim, 
            action_dim=action_dim, 
            model_type=model_type, 
            trainable_std=True
        ).to(device)

        self.critic = Critic(
            state_dim=state_dim, 
            model_type=model_type
        ).to(device)

        self.args = args
        self.kwargs = kwargs

        self.data = ReplayBuffer(
            max_size=self.args.traj_length, 
            state_dim=state_dim,
            num_action=action_dim,
            device=self.device,
        )

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.args.critic_lr)
   
    def get_action(self, x):
        x = x.unsqueeze(0) if len(x.shape) == 1 else x
        mu, sigma = self.actor(x)

        return mu, sigma
    
    def v(self, x):
        return self.critic(x)

    def put_data(self, state, action, reward, next_state, done, log_probs):
        self.data.put_data(state, action, reward, next_state, done, log_probs)

    def reset_hidden_states(self):
        self.hidden_state_actor = None
        self.hidden_state_critic = None

    def get_gae(self, states, rewards, next_states, dones):
        ### what does our values ann say?
        values = self.v(states).detach()
        ### what does it say about the resulting state after discounting
        td_target = rewards + self.args.gamma * self.v(next_states) * (1 - dones)
        ### what is the difference?
        delta = td_target - values
        delta = delta.detach().cpu().numpy()
        advantage_lst = []
        advantage = 0.0
        for idx in reversed(range(len(delta))):
            if dones[idx] == 1:
                advantage = 0.0
            advantage = self.args.gamma * self.args.lambda_ * advantage + delta[idx][0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantages = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)
        return values, advantages

    def train_net(self):
        self.episode_num += 1

        for _ in range(self.args.train_epoch):
            for batch in self.data.make_mini_batch(
                shuffle=False,
                mini_batch_size=self.args.batch_size, 
                get_gae=self.get_gae,
            ):
                state, action, old_log_prob, advantage, return_, old_value = batch
                curr_mu, curr_sigma = self.get_action(state)
                value = self.v(state).float()
                curr_dist = torch.distributions.Normal(curr_mu, curr_sigma)
                entropy = curr_dist.entropy() * self.args.entropy_coef
                curr_log_prob = curr_dist.log_prob(action).sum(1, keepdim=True)

                # policy clipping
                ratio = torch.exp(curr_log_prob - old_log_prob.detach())
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.args.max_clip, 1 + self.args.max_clip) * advantage
                actor_loss = (-torch.min(surr1, surr2) - entropy).mean()

                # value clipping (PPO2 technic)
                old_value_clipped = old_value + (value - old_value).clamp(-self.args.max_clip, self.args.max_clip)
                value_loss = (value - return_.detach().float()).pow(2)
                value_loss_clipped = (old_value_clipped - return_.detach().float()).pow(2)
                critic_loss = 0.5 * self.args.critic_coef * torch.max(value_loss, value_loss_clipped).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.max_grad_norm)
                self.critic_optimizer.step()

                if self.writer != None:
                    self.writer.add_scalar("loss/actor_loss", actor_loss.item(), self.episode_num)
                    self.writer.add_scalar("loss/critic_loss", critic_loss.item(), self.episode_num)
