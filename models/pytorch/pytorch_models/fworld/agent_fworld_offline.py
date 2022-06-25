#a empirical way to analyze performance over time
#this should overfit, and is more to analyze the efficiacy of different variable inputs
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


import pickle
import os
import numpy as np
from copy import copy

from typing import Dict, List, AnyStr
from models.pytorch.pytorch_models.fworld.config_experiment import ConfigETensorVariable
from models.pytorch.pytorch_models.fworld.config_experiment import ConfigFWorldVariables
from models.pytorch.pytorch_models.fworld.config_experiment import file_to_fworldconfig
from models.pytorch.pytorch_models.fworld.config_experiment import ConfigRunsOffline
from models.pytorch.pytorch_models.fworld import agent_fworld_supervised as agent_fworld, fworld_metrics


#todo add clearer documentation about run process and functions
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
        target_value[self._relevant_target_name] = np.array(elements[self._relevant_target_name].values).T
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
    config_run: ConfigRunsOffline = ConfigRunsOffline.file_to_configrun(os.path.join("config", "run_config_offline.yaml"))

    data:dict = None
    with open(config_run.datapath, 'rb') as f:
        data = pickle.load(f)

    all_input_information:List[ConfigETensorVariable] = [config_fworld.object_seen,config_fworld.visionwide,
                                                         config_fworld.visionlocal,config_fworld.internal_state,
                                                         config_fworld.sensory_local2,config_fworld.sensory_local]

    for hidden_size in config_run.hidden_size:
        for feature_index in range(len(all_input_information)+1):
                wandb.init(project="fworldsupervised-clean")
                config_variables = config_run.asdict()
                config_variables["hidden_size"] = hidden_size
                ##give logging context
                wandb.config.update(config_variables) #given context of ablated information
                copied_input_information: List[ConfigETensorVariable] = copy(all_input_information)

                ablation_feature = "keepall" if feature_index == len(all_input_information) else copied_input_information[feature_index].name

                if feature_index < len(all_input_information):
                    del copied_input_information[feature_index]

                wandb.config.update({"ablation_feature": ablation_feature})



                model_dync:PolicyDynamicOffline = PolicyDynamicOffline(copied_input_information,
                                               hidden_size)
                optimizer_o: optim.Optimizer  = optim.Adam(model_dync.parameters(), lr=.0005)
                dataset_fworld = DatasetFWorld(data,[i.name for i in copied_input_information],"Heuristic")

                wandb.run.name = "ablate-{}-{}-{}".format(ablation_feature, config_run.name, str(wandb.run.id))
                for j in range(config_run.max_epochs):
                        total_error = 0
                        dataloader_fworld = DataLoader(dataset_fworld,batch_size=50, num_workers=0, shuffle=True)
                        chosen_action_history = []
                        best_action_history = []

                        for i, batch in enumerate(dataloader_fworld):
                            formatted_tensors_features = model_dync.get_worldstate_tensors(batch[0])
                            ground_truth = batch[1]["Heuristic"]
                            predictions = model_dync(formatted_tensors_features.float())
                            best_actions  = ground_truth.flatten().long()
                            loss = nn.CrossEntropyLoss()
                            the_error:torch.Tensor = loss(predictions, best_actions)
                            the_error.backward()
                            optimizer_o.step()
                            optimizer_o.zero_grad()
                            total_error += the_error.detach()

                            chosen_action_history.append(torch.argmax(predictions, 1).detach().numpy())
                            best_action_history.append(best_actions.detach().numpy())

                        np_chosen = np.concatenate(chosen_action_history)
                        np_best = np.concatenate(best_action_history)
                        kl_divergence:float = fworld_metrics.calc_kl(np_chosen, np_best)
                        f1_score:float = fworld_metrics.calc_precision(np_chosen, np_best)
                        wandb.log({"kl":kl_divergence},step=j)
                        wandb.log({"f1":f1_score},step=j)
                #final confusion matrix
                confusion_matrix = fworld_metrics.calc_confusion_matrix(np_chosen, np_best, ["f", "l", "r", "e", "d"])
                wandb.log({"action_confusion_matrix" : wandb.plot.confusion_matrix(class_names=["Forward","Left","Right","Eat","Drink"],
                                                                                   y_true=np_best, preds= np_chosen)})

                #print("kl {:.2f}\nf1 {:.2f}".format(kl_divergence,f1_score)) #make logging flag
                #print(confusion_matrix) #make logging flag
                wandb.finish()