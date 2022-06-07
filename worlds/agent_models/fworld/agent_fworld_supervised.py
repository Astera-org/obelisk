# TODO(michael): Move this into models/ and fix the Python import issues

import argparse
import random
from typing import Any, Dict, List, AnyStr
import math
import numpy as np
from itertools import count
from collections import namedtuple
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from network.genpy.env.ttypes import ETensor, Action
from network.thrift_agent_server import setup_server
import pandas as pd

# FWorld
from worlds.agent_models.fworld.fworld_features import ConfigETensorVariable
from worlds.agent_models.fworld.fworld_features import ConfigFWorldVariables
from worlds.agent_models.fworld.fworld_features import file_to_fworldconfig

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
        self.chosen_action_history = []

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

    def get_worldstate(self,observations:Dict[AnyStr,List[float]]):
           return np.array(observations["V2Wd"])

    def delete_history(self):
        #del self.rewards[:]
        del self.saved_actions[:]
        #if len(model.best_action) > 0:
        #    model.best_action_history.append(model.best_action[-1].detach())
        del self.best_action[:]
        del self.save_full_actions[:]

class PolicyDynamicInput(Policy):
    def __init__(self, fworld_state_shapes:List[ConfigETensorVariable], hidden_size):
        super().__init__()
        self.relevant_features:List[ConfigETensorVariable] = fworld_state_shapes
        input_size = np.sum([i.flattened_shape for i in self.relevant_features])

        total_hidden_size: int = int(input_size * hidden_size) if hidden_size < 1 else hidden_size

        self.affine1 = nn.Linear(input_size, total_hidden_size) # Inputs from V2Wd
        self.action_head = nn.Linear(total_hidden_size, 5) # 5 actions

        self.store_history = []

    def get_worldstate(self,observations:Dict[AnyStr,List[float]]):
        all_observations = [observations[etensor_info.name].values for etensor_info in self.relevant_features]
        return np.concatenate(all_observations).ravel()

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


def quick_analysis(model):
    actions = pd.DataFrame(0)
    actions["ground_truth"] = model.best_action_history
    actions["predicted"] = model.chosen_action_history
    bins = pd.cut(actions.index,bins = 10)
    actions["bins"]=[bin.left for bin in bins ]

    for name, group in actions.groupby("bins"):
        print(group["ground_truth"].value_counts())
        print(group["predicted"].value_counts())

    #do Normalized performance Metric
    #corrected F1 score
    #do kl divergence
    #do reward

def supervised_finish_episode(model:nn.Module, optimizer:optim.Optimizer):
    current_actions =model.save_full_actions[0].float() #torch.tensor(model.save_full_actions)
    current_actions = current_actions.view(1,current_actions.shape[-1])
    best_actions  = torch.tensor([model.best_action[-1]]).long()
    loss = nn.CrossEntropyLoss()

    the_error = loss(current_actions,best_actions)
    the_error.backward()
    optimizer.step()
    optimizer.zero_grad()
    #record performance
    #model.best_action_history.append(model.best_action[-1])


class FWorldHandler:

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer):
        self.name: str = "AgentZero"
        self.model: PolicyDynamicInput = model
        self.optimizer: optim.Optimizer = optimizer
        self._number_to_action= {0:"Forward",1:"Left",2:"Right",3:"Eat",4:"Drink"} #quickly hard coded
        self._use_heuristic = True
        self.do_learning = True


    def set_mode_to_send(self,use_heuristic:bool):
        self._use_heuristic = use_heuristic

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
        #print(f'step called obs: {observations} debug: {debug}')

        i_episode = 0
        if ":" in debug: # TODO Get i from here!
            i_episode = int(debug.split(":")[-1])


        # Do learning for the previous timestep

        if (i_episode > 1) and self.do_learning:
            reward = observations["Reward"].values[0]
            self.model.rewards.append(reward)
            self.ep_reward += reward
            self.model.best_action.append(observations["Heuristic"].values[0])
            self.model.best_action_history.append(self.model.best_action[-1])


            #finish_episode()
            supervised_finish_episode(self.model,self.optimizer)
            self.model.delete_history()

        # select action from policy

        world_state = self.model.get_worldstate(observations)

        # TODO Handle n-dimensional shapes
        action = select_action(self.model,world_state)

        if (i_episode>1):
            self.model.chosen_action_history.append(action)
            self.model.store_history.append(world_state)

        if (len(self.model.chosen_action_history)>1000):
            self.set_mode_to_send(use_heuristic=False)
            self.do_learning = False
        else:
            self.set_mode_to_send(use_heuristic=True)

        return {"move":Action(discreteOption=action),"use_heuristic":Action(discreteOption=int(self._use_heuristic))}


if __name__ == '__main__':


    config_fworld: ConfigFWorldVariables = file_to_fworldconfig(os.path.join("config", "config_inputs.yaml"))

    model_dync = PolicyDynamicInput([config_fworld.object_seen,config_fworld.visionwide,config_fworld.visionlocal,config_fworld.internal_state], 125)


    optimizer_o: optim.Optimizer  = optim.Adam(model_dync.parameters(), lr=.00001)
    handler:FWorldHandler = FWorldHandler(model_dync,optimizer_o)
    server = setup_server(handler)
    server.serve()

    #todo
    #convert output into sparse encoding
    #Use PCT Cortex FOr X runs
    #STORE RUNNING HISTORY OF INPUTS AND OUTPUTS
    #off policy results
    #on policy results
    #Visulize results


