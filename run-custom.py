from configparser import ConfigParser
from argparse import ArgumentParser

import torch
import numpy as np
import os
import itertools
import json
import tqdm

import wandb

from rideshare.env_multi import MultiRideshareEnv
from utils.utils import RunningMeanStd
from CompanyPriceModel import CompanyPriceModel
from env_kwargs import get_env_kwargs

os.makedirs('./model_weights', exist_ok=True)

class Dict(dict):
    def __init__(self, config, section_name, location=False):
        super(Dict, self).__init__()
        self.initialize(config, section_name, location)

    def initialize(self, config, section_name, location):
        for key, value in config.items(section_name):
            if location:
                self[key] = value
            else:
                self[key] = eval(value)

    def __getattr__(self, val):
        return self[val]


def main(cli_args, agent_args, env_kwargs):
    if cli_args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter()
    else:
        writer = None


    run = wandb.init(
        project="Two-Way Markets",
        config={**env_kwargs, **agent_args},
    )

    env = MultiRideshareEnv(**env_kwargs)
    possible_agents = env.possible_agents

    action_shape = env.action_space(None).shape
    state_dim = env.observation_space(None).shape[0]
    state_rms = RunningMeanStd(state_dim)

    # model_kwargs_configs = json.load(open('default_model_config.json'))
    # model_permutations = list(itertools.permutations(model_kwargs_configs.keys(), 2))

    # tmp_agents = model_permutations[0] 

    if cli_args.algo == 'ppo':
        device = torch.device(
            'cuda' if (
                torch.cuda.is_available() and 
                (cli_args.use_cuda)
            )
            else 'cpu'
        )
        agent_U = CompanyPriceModel(
            model_type='short_skinny_fcc',
            writer=writer, 
            device=device, 
            state_dim=state_dim, 
            action_shape=action_shape, 
            agent_args=agent_args
        )
        agent_L = CompanyPriceModel(
            model_type='tall_skinny_fcc',
            writer=writer, 
            device=device, 
            state_dim=state_dim, 
            action_shape=action_shape, 
            agent_args=agent_args
        )
    else:
        raise NotImplementedError("Only PPO supported for now")

    score_lst = []
    state_lst = []

    seed = 42
    observations, _ = (env.reset(seed=seed))
    state = observations[possible_agents[0]]   # agents share the observation
    # state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
    for _ in tqdm.tqdm(range(cli_args.epochs)):
        for t in range(agent_args.traj_length):
            state_lst.append(state)
            action_U = agent_U.sample_action(state)
            action_L = agent_L.sample_action(state)
            actions = {"U": action_U, "L": action_L}
            # Do something to make the log work
            log = agent_args.traj_length -1 == t
            observations, rewards, terminations, truncations, _ = env.step(actions, log)

            next_state = observations["U"] # agents share observations
            reward_U, reward_L = rewards["U"], rewards["L"]

            # next_state = np.clip((next_state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
            agent_U.record_transition(
                state=state,
                action=action_U,
                reward=reward_U,
                next_state=next_state, 
                done=(terminations['U'] or truncations['U']),
            )
            agent_L.record_transition(
                state=state,
                action=action_L,
                reward=reward_L,
                next_state=next_state, 
                done=(terminations['L'] or truncations['L']),
            )

            state = next_state
            if terminations['U'] or terminations['L'] or truncations['U'] or truncations['L']:
                break

        agent_U.agent.train_net()
        agent_L.agent.train_net()
        # state_rms.update(np.vstack(state_lst))
        env.reset(seed=None)


if __name__ == "__main__":
    parser = ArgumentParser('parameters')

    parser.add_argument("--env_name", type=str, default='', help="Default petting zoo env is ...")
    parser.add_argument("--algo", type=str, default='ppo', help='algorithm to adjust (default : ppo)')
    parser.add_argument('--train', type=bool, default=True, help="(default: True)")
    parser.add_argument('--render', type=bool, default=False, help="(default: False)")
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs, (default: 1)')
    parser.add_argument('--tensorboard', type=bool, default=False, help='use_tensorboard, (default: False)')
    parser.add_argument("--load", type=str, default='no', help='load network name in ./model_weights')
    parser.add_argument("--save_interval", type=int, default=100000, help='save interval(default: 100)')
    parser.add_argument("--print_interval", type=int, default=10000, help='print interval(default : 20)')
    parser.add_argument("--use_cuda", type=bool, default=True, help='cuda usage(default : True)')
    parser.add_argument("--reward_scaling", type=float, default=1, help='reward scaling(default : 0.1)')
    cli_args = parser.parse_args()
    parser = ConfigParser()
    parser.read('config.ini')
    agent_args = Dict(parser, cli_args.algo)

    env_kwargs = get_env_kwargs(agent_args)

    main(cli_args, agent_args, env_kwargs)