from customPPO import PPO
import torch

class CompanyPriceModel:
    def __init__(self, model_type, writer, device, state_dim, action_shape, agent_args, reward_scale=1):
        action_dim = action_shape[0] * action_shape[1]
        self.agent = PPO(
            model_type,
            writer=writer, 
            device=device, 
            state_dim=state_dim, 
            action_dim=action_dim, 
            args=agent_args,
        )
        self.device = device
        self.reward_scale = reward_scale
        self.action_shape = action_shape

    def sample_action(self, state):
        self.mu, self.sigma = self.agent.get_action(torch.from_numpy(state).float().to(self.device))
        self.dist = torch.distributions.Normal(self.mu, self.sigma[0])
        self.action = self.dist.sample()
        self.log_prob = self.dist.log_prob(self.action).sum(-1, keepdim=True)
        return self.action.cpu().numpy().reshape(self.action_shape)

    def record_transition(self, state, action, reward, next_state, done):
        self.agent.put_data(
            state=state,
            action=action,
            reward=torch.tensor([reward * self.reward_scale]),
            next_state=next_state,
            done=torch.tensor([done]),
            log_probs=self.log_prob.detach(),
        )


