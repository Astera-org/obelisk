# TODO(michael): Move this into models/ and fix the Python import issues

import argparse
import random

import math
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from network.genpy.env.ttypes import ETensor, Action
from network.thrift_agent_server import setup_server

# FWorld

SEED = 1


torch.manual_seed(SEED)




class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super().__init__()
        self.affine1 = nn.Linear(416, 128) # Inputs from V2Wd
        self.action_head = nn.Linear(128, 5) # 5 actions
        # action & reward buffer
        self.saved_actions = []
        self.save_full_actions = []
        self.best_action = [] #actions it should take
        self.best_action_history = [] #all best actions it should take
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))
        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)
        return action_prob

    def delete_history(self):
        del self.model.rewards[:]
        del self.model.saved_actions[:]
        #if len(model.best_action) > 0:
        #    model.best_action_history.append(model.best_action[-1].detach())
        del self.model.best_action[:]
        del self.model.save_full_actions[:]

def select_action(model:nn.Module, state:np.array):
    state = torch.from_numpy(state).float()
    probs = model(state)
    probs = torch.nan_to_num(probs, nan=0.1)
    foundNan = False
    for i in range(len(probs)):
        if math.isnan(probs[i]):
            foundNan = True
    if foundNan:
        print("Found some NaN values")
    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)
    # and sample an action using the distribution
    action = m.sample()
    model.save_full_actions.append(probs) #full action space
    # save to action buffer
    return action.item()


def supervised_finish_episode(model:nn.Module, optimizer:optim.Optimizer):
    current_actions =model.save_full_actions[0].float() #torch.tensor(model.save_full_actions)
    current_actions = current_actions.view(1,current_actions.shape[-1])
    best_actions  = torch.tensor([model.best_action[-1]]).long()
    loss = nn.CrossEntropyLoss()
    the_error = loss(current_actions,best_actions)
    the_error.backward()
    optimizer.step()
    #record performance
    model.best_action_history.append(model.best_action[-1])


class FWorldHandler:

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer):
        self.name: str = "AgentZero"
        self.model: nn.Module = model
        self.optimizer: optim.Optimizer = optimizer

        self._number_to_action= {0:"Forward",1:"Left",2:"Right",3:"Eat",4:"Drink"} #quickly hard coded

    def init(self, actionSpace, observationSpace):
        '''
        server connect init
        '''
        self.actionSpace = actionSpace
        self.observationSpace = observationSpace
        print("Action Space:", actionSpace, "\n\nObservation Space:", observationSpace)
        # reset episode reward
        self.ep_reward = 0
        self.running_reward = 10
        return {"name":self.name}

    def step(self, observations, debug):
        print(f'step called obs: {observations} debug: {debug}')

        i_episode = 0
        if ":" in debug: # TODO Get i from here!
            i_episode = int(debug.split(":")[-1])

        do_learning = True
        # Do learning for the previous timestep

        if (i_episode > 1) and do_learning:
            reward = observations["Reward"].values[0]
            self.model.rewards.append(reward)
            self.ep_reward += reward
            self.model.best_action.append(observations["Heuristic"].values[0])
            #finish_episode()
            supervised_finish_episode(self.model,self.optimizer)

        # select action from policy
        world_state = np.array(observations["V2Wd"].values)


        # TODO Handle n-dimensional shapes
        action = select_action(self.model,world_state)

        return {"move":Action(discreteOption=action)}


if __name__ == '__main__':
    model: nn.Module = Policy()
    optimizer: optim.Optimizer  = optim.Adam(model.parameters(), lr=.00001)
    handler:FWorldHandler = FWorldHandler(model,optimizer)
    server = setup_server(handler)
    server.serve()

