from configparser import ConfigParser
from argparse import ArgumentParser

import torch
import numpy as np
import os

import wandb

from customPPO import PPO
from rideshare.env_multi import MultiRideshareEnv

from utils.utils import make_transition, Dict, RunningMeanStd

os.makedirs('./model_weights', exist_ok=True)

parser = ArgumentParser('parameters')

parser.add_argument("--env_name", type=str, default='', help="Default petting zoo env is ...")
parser.add_argument("--algo", type=str, default='ppo', help='algorithm to adjust (default : ppo)')
parser.add_argument('--train', type=bool, default=True, help="(default: True)")
parser.add_argument('--render', type=bool, default=False, help="(default: False)")
parser.add_argument('--epochs', type=int, default=100, help='number of epochs, (default: 1)')
parser.add_argument('--tensorboard', type=bool, default=False, help='use_tensorboard, (default: False)')
parser.add_argument("--load", type=str, default='no', help='load network name in ./model_weights')
parser.add_argument("--save_interval", type=int, default=100000, help='save interval(default: 100)')
parser.add_argument("--print_interval", type=int, default=10000, help='print interval(default : 20)')
parser.add_argument("--use_cuda", type=bool, default=True, help='cuda usage(default : True)')
parser.add_argument("--reward_scaling", type=float, default=1, help='reward scaling(default : 0.1)')
args = parser.parse_args()
parser = ConfigParser()
parser.read('config.ini')
agent_args = Dict(parser, args.algo)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.use_cuda == False:
    device = 'cpu'

if args.tensorboard:
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter()
else:
    writer = None


OD2 = [[0., 0.9],
       [0.2, 0.]]
C2 = [[0., 5.],
      [2., 0.]]
init_passenger_distribution2 = [2000., 3000.]
init_driver_distribution2 = [500., 1000.]

# Set vector_state to false in order to use visual observations (significantly longer training time)
ts = 8*864000
d = 0.1
env_kwargs = {
    "OD":                                       np.array(OD2),
    "C":                                        np.array(C2),
    "init_passenger_distribution":              np.array(init_passenger_distribution2),
    "init_driver_distribution":                 np.array(init_driver_distribution2),
    "max_rate":         20.,                    # maximum charged per mile is 20
    "max_timestep":     ts ,                    # seconds in a day [Deprecated]
    "lbd":              2.,                     # Base waiting cost lambda
    "rp":               10.,                    # public transportation charges
    "g":                5.,                     # gas cost
    "num_D_samples":    10,                     # random samples generated by drivers
    "a_d":              d,                      # at each second, can only change by this much
    "p_d":              1.,                     # Passengers responsive
    "alpha":            1e-2,                   # step parameter for continuous term
    "mode":             "interp",               # Always interp. "delta" deprecated
    "traj_length":      agent_args["traj_length"]
}

run = wandb.init(
    project="Rideshare Multi Episodic",
    config={**env_kwargs, **agent_args},
)

env = MultiRideshareEnv(**env_kwargs)
possible_agents = env.possible_agents

action_shape = env.action_space(None).shape
action_dim = action_shape[0] * action_shape[1]
state_dim = env.observation_space(None).shape[0]
state_rms = RunningMeanStd(state_dim)

class PPOAgent:
    def __init__(self, writer, device, state_dim, action_dim, agent_args):
        self.agent = PPO(writer, device, state_dim, action_dim, agent_args)

    def sample_action(self, state):
        self.mu, self.sigma = self.agent.get_action(torch.from_numpy(state).float().to(device))
        self.dist = torch.distributions.Normal(self.mu, self.sigma[0])
        self.action = self.dist.sample()
        self.log_prob = self.dist.log_prob(self.action).sum(-1, keepdim=True)
        return self.action.cpu().numpy().reshape(action_shape)

    def record_transition(self, next_state, reward, done):
        transition = make_transition(state, \
                                     self.action.cpu().numpy(), \
                                     np.array([reward * args.reward_scaling]), \
                                     next_state, \
                                     np.array([done]), \
                                     self.log_prob.detach().cpu().numpy() \
                                     )
        self.agent.put_data(transition)


if args.algo == 'ppo':
    agent_U = PPOAgent(writer, device, state_dim, action_dim, agent_args)
    agent_L = PPOAgent(writer, device, state_dim, action_dim, agent_args)
else:
    raise NotImplementedError("Only PPO supported for now")

if (torch.cuda.is_available()) and (args.use_cuda):
    agent_U = agent_U.cuda()
    agent_L = agent_L.cuda()

if args.load != 'no':
    agent_U.load_state_dict(torch.load("./model_weights/" + args.load + "_U"))
    agent_L.load_state_dict(torch.load("./model_weights/" + args.load + "_L"))

score_lst = []
state_lst = []

seed = 42
scoreU, score_L = 0.0, 0.0
observations, infos = (env.reset(seed=seed))
state_ = observations[possible_agents[0]]   # agents share the observation so doesn't really matter which dict we pull from
state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
for n_epi in range(args.epochs):
    for t in range(agent_args.traj_length):
        state_lst.append(state_)
        action_U = agent_U.sample_action(state)
        action_L = agent_L.sample_action(state)
        actions = {"U": action_U, "L": action_L}
        log = done = agent_args.traj_length -1 == t
        observations, rewards, terminations, truncations, infos = env.step(actions, log)

        next_state_ = observations["U"] # agents share observation, whichever's fine
        reward_U, reward_L = rewards["U"], rewards["L"]

        next_state = np.clip((next_state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
        agent_U.record_transition(next_state, reward_U, done)
        agent_L.record_transition(next_state, reward_L, done)

    agent_U.agent.train_net(n_epi)
    agent_L.agent.train_net(n_epi)
    state_rms.update(np.vstack(state_lst))    # All observations are [0, 1] so not really used.
    env.reset(seed=None)
