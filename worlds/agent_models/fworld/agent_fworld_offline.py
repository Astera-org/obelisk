#a empirical way to analyze performance over time
#this should overfit, and is more to analyze the efficiacy of different variable inputs
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from worlds.agent_models.fworld import agent_fworld_supervised as agent_fworld


#ablate each part
#run N iterations
#save actions at a given point

from typing import List

import pickle
import os
import numpy as np
from typing import Any, Dict, List, AnyStr
from worlds.agent_models.fworld.config_experiment import ConfigETensorVariable
from worlds.agent_models.fworld.config_experiment import ConfigFWorldVariables
from worlds.agent_models.fworld.config_experiment import file_to_fworldconfig
from worlds.agent_models.fworld.config_experiment import ConfigRuns

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class DatasetFWorld(Dataset):
    def __init__(self, fworld_features:List[dict], feature_names:List[str], target_name:str):
        self._fworld_features:List[dict] = fworld_features
        self._feature_names = feature_names
        self._relevant_target_name = target_name
    def set_features(self,relevant_feature_names:List[str]):
        self.feature_names = relevant_feature_names
    def set_target (self,relevant_target:str):
        self._relevant_target_name = relevant_target
    def __len__(self):
        return len(self._fworld_features)
    def __getitem__(self, item):
        elements:dict = self._fworld_features[item]["observations"]
        raw_values:dict = dict()
        target_value = dict()
        for name in self._feature_names:
            raw_values[name] = np.array(elements[name].values).T #so shapes are proper format
        target_value[self._relevant_target_name] = elements[name].values
        return raw_values,target_value

class PolicyDynamicOffline(agent_fworld.PolicyDynamicInput):
    def __init__(self, *args,**kwargs):
        super().__init__(*args,**kwargs)
    def get_worldstate(self,observations:Dict[AnyStr,List[float]]):
        assert False, "wordstate function needs to be retooled"
    def get_worldstate_tensors(self,observations:Dict[AnyStr,List[float]])->torch.Tensor:
        relevant_tensors = [observations[feature.name] for feature in self.relevant_features]
        return torch.cat(relevant_tensors,1)




if __name__ == '__main__':
    config_fworld: ConfigFWorldVariables = file_to_fworldconfig(os.path.join("config", "config_inputs.yaml"))
    config_run: ConfigRuns = ConfigRuns.file_to_configrun(os.path.join("config","run_config.yaml"))

    datapath:str = "fworld_onpolicysupervised-2ets9y8o.pkl"
    data:dict = None
    with open(datapath, 'rb') as f:
        data = pickle.load(f)

    all_input_information:List[ConfigETensorVariable] = [config_fworld.object_seen,config_fworld.visionwide,
                                                         config_fworld.visionlocal,config_fworld.internal_state,
                                                         config_fworld.sensory_local2,config_fworld.sensory_local]

    model_dync:PolicyDynamicOffline = PolicyDynamicOffline(all_input_information,
                                    config_run.hidden_size)
    optimizer_o: optim.Optimizer  = optim.Adam(model_dync.parameters(), lr=.00001)

    dataset_fworld = DatasetFWorld(data,[i.name for i in all_input_information],"Heuristic")



for j in range(config_run.max_epochs):
        dataloader_fworld = DataLoader(dataset_fworld,batch_size=20, num_workers=0, shuffle=True)
        for i, batch in enumerate(dataloader_fworld):
            formatted_tensors = model_dync.get_worldstate_tensors(batch)

            prediction = model_dync(formatted_tensors.float())

            print("OK")
