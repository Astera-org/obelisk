# TODO(michael): Move this into models/ and fix the Python import issues

import argparse
import random
from typing import Any, Dict, List, AnyStr
import math
import numpy as np
from itertools import count
from collections import namedtuple
import os

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.distributions import Categorical

from network.genpy.env.ttypes import ETensor, Action
from network.thrift_agent_server import setup_server
import pandas as pd

# FWorld
from worlds.agent_models.fworld.config_experiment import ConfigETensorVariable
from worlds.agent_models.fworld.config_experiment import ConfigFWorldVariables
from worlds.agent_models.fworld.config_experiment import file_to_fworldconfig
from worlds.agent_models.fworld.config_experiment import ConfigRuns
from worlds.agent_models.fworld import fworld_metrics

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

        self.affine1 = nn.Linear(input_size, total_hidden_size) # Inputs from fworld
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

    #failing to adjust to out of distirbution

class FWorldHandler:
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, train_runs:int,max_runs:int):
        self.name: str = "AgentZero"
        self.model: PolicyDynamicInput = model
        self.optimizer: optim.Optimizer = optimizer
        self._number_to_action= {0:"Forward",1:"Left",2:"Right",3:"Eat",4:"Drink"} #quickly hard coded
        self._use_heuristic = True
        self.do_learning = True
        self._train_runs = train_runs
        self._max_runs = max_runs

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
        i_episode = 0
        if ":" in debug: # TODO Get i from here!
            i_episode = int(debug.split(":")[-1])

        # Do learning for the previous timestep
        if (i_episode > 1):
            reward = observations["Reward"].values[0]
            self.model.rewards.append(reward)
            self.ep_reward += reward
            self.model.best_action.append(observations["Heuristic"].values[0])
            self.model.best_action_history.append(self.model.best_action[-1])
            #finish_episode()
            if self.do_learning:
                supervised_finish_episode(self.model,self.optimizer)
            self.model.delete_history()
        # select action from policy
        world_state = self.model.get_worldstate(observations)

        # TODO Handle n-dimensional shapes
        action = select_action(self.model,world_state)

        if (i_episode>1):
            self.model.chosen_action_history.append(action)
            self.model.store_history.append(world_state)

        if (i_episode>=self._train_runs): #quick hack so doesn't reset, should be cleaned up
            self.set_mode_to_send(use_heuristic=False)
            self.do_learning = False
        else:
            self.set_mode_to_send(use_heuristic=True)

        #log data at end of training, instead of online, and then
        if i_episode==self._train_runs:
            self.log_results("train",self.model.chosen_action_history,self.model.best_action_history,self.model.store_history,self.model.rewards)
            self.model.chosen_action_history = []
            self.model.best_action_history = []
            self.model.store_history = []
            self.model.rewards = []

        if i_episode==self._max_runs:
            self.log_results("offpolicy-inference",self.model.chosen_action_history,self.model.best_action_history,self.model.store_history,self.model.rewards)


        return {"move":Action(discreteOption=action),"use_heuristic":Action(discreteOption=int(self._use_heuristic))}


    def log_results(self, title:str, predicted_actions:List[int],heuristic_actions:List[int],input_space:List[int], rewards:List[float]):
        actions = pd.DataFrame()
        actions["ground_truth"] = heuristic_actions
        actions["predicted"] = predicted_actions
        actions["rewards"] = rewards
        bins = pd.cut(actions.index,bins = 10)
        actions["bins"]=[bin.left for bin in bins ]

        step =0
        for name, group in actions.groupby("bins"):
            ground_truth = group["ground_truth"].values
            predicted = group["predicted"].values
            reward = group["rewards"].mean()
            kl_divergence:float = fworld_metrics.calc_kl(predicted,ground_truth)
            f1_score:float = fworld_metrics.calc_precision(predicted,ground_truth)

            wandb.log({"{}_kl_divergence".format(title):kl_divergence},step=step)
            wandb.log({"{}_f1".format(title):f1_score},step=step)
            wandb.log({"{}_reward".format(title):reward},step=step)
            step+=int(len(actions)/10)

        wandb.log({"{}_conf_mat".format(title) : wandb.plot.confusion_matrix(class_names=["Forward","Left","Right","Eat","Drink"],y_true=heuristic_actions, preds= predicted_actions)})

        wandb.run.summary["{}_f1".format(title)] = fworld_metrics.calc_precision(actions.predicted,actions.ground_truth)
        wandb.run.summary["{}_kl".format(title)] = fworld_metrics.calc_kl(actions.predicted,actions.ground_truth)

        sample_amount = len(input_space) if len(input_space) < 1000 else 1000
        history = pd.DataFrame(input_space).sample(sample_amount)#so don't make this thing lag
        del actions["bins"]
        del actions["rewards"]

        wandb.log({"actions":actions,"inputs":history, "rewards":rewards}) #so can replicate results if neccesary


if __name__ == '__main__':

    config_fworld: ConfigFWorldVariables = file_to_fworldconfig(os.path.join("config", "config_inputs.yaml"))
    config_run: ConfigRuns = ConfigRuns.file_to_configrun(os.path.join("config","run_config.yaml"))

    wandb.init(project="fworld-evaluations1")
    wandb.config.update(config_run.asdict()) #given conftext information

    wandb.run.name = "{}-{}".format(config_run.name,str(wandb.run.id))

    all_input_information:List[ConfigETensorVariable] = [config_fworld.object_seen,config_fworld.visionwide,
                              config_fworld.visionlocal,config_fworld.internal_state,
                              config_fworld.sensory_local2,config_fworld.sensory_local]

    model_dync = PolicyDynamicInput([config_fworld.object_seen,config_fworld.visionwide,config_fworld.visionlocal,config_fworld.internal_state],
                                    config_run.hidden_size)

    wandb.config.update({"inputsize":model_dync.affine1.in_features})
    wandb.config.update({"inputareas":",".join([i.name for i in all_input_information])})
    optimizer_o: optim.Optimizer  = optim.Adam(model_dync.parameters(), lr=.00001)
    handler:FWorldHandler = FWorldHandler(model_dync,optimizer_o,config_run.train_runs,config_run.max_runs)
    server = setup_server(handler)
    server.serve()

    #todo
    #convert output into sparse encoding
    #Use PCT Cortex FOr X runs
    #STORE RUNNING HISTORY OF INPUTS AND OUTPUTS
    #off policy results
    #on policy results
    #Visulize results

