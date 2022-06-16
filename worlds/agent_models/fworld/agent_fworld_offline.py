#a empirical way to analyze performance over time
#this should overfit, and is more to analyze the efficiacy of different variable inputs


from worlds.agent_models.fworld import agent_fworld_supervised as agent_fworld


#ablate each part
#run N iterations
#save actions at a given point

from typing import List

import pickle
import os
import numpy as np
from worlds.agent_models.fworld.config_experiment import ConfigETensorVariable
from worlds.agent_models.fworld.config_experiment import ConfigFWorldVariables
from worlds.agent_models.fworld.config_experiment import file_to_fworldconfig
from worlds.agent_models.fworld.config_experiment import ConfigRuns

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class DatasetFWorld(Dataset):
    def __init__(self, fworld_features:List[dict]):
        self._fworld_features:List[dict] = fworld_features
    def set_features(self,relevant_feature_names:List[str]):
        self._relevant_names = relevant_feature_names
    def set_target (self,relevant_target:str):
        self._relevant_target_name = relevant_target
    def __len__(self):
        return len(self._fworld_features)
    def __getitem__(self, item):
        elements:dict = self._fworld_features[item]["observations"]
        raw_values:dict = dict()
        for name in elements:
            raw_values[name] = np.array(elements[name].values).T #so shapes are proper format
        return raw_values




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

    model_dync:agent_fworld.PolicyDynamicInput = agent_fworld.PolicyDynamicInput(all_input_information,
                                    config_run.hidden_size)




    temp = DatasetFWorld(data,None,None)
    dataloader_fworld = DataLoader(temp,batch_size=20, num_workers=0, shuffle=True)
    example = next(iter(dataloader_fworld))


    for j in range(config_run.max_runs):
        dataloader_fworld = DataLoader(temp,batch_size=20, num_workers=0, shuffle=True)
        for i, batch in enumerate(dataloader_fworld):
            print(batch)
