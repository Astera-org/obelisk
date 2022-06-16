import dataclasses
import os
from dataclasses import dataclass,asdict
from dataclasses import field
from typing import Any, Dict, List, AnyStr
import yaml
import copy
import numpy as np
import warnings

# temporary for now
warnings.filterwarnings(action="once")
warnings.simplefilter("ignore")

def dict_to_configobj(parsed_yaml: Dict[Any, Any],dataclass_obj: type):
    cleaned_yaml = keep_relevant_keys(parsed_yaml,dataclass_obj)
    return dataclass_obj(**cleaned_yaml)

def file_to_configobj(path: str, dataclass_obj: type) :
    with open(path) as yaml_file:
        yaml_string = yaml_file.read()
        yaml_file = yaml.safe_load(yaml_string)
        return dict_to_configobj(yaml_file,dataclass_obj)

def file_to_dict(path: str)->Dict[str,Any]:
    with open(path) as yaml_file:
        yaml_string = yaml_file.read()
        yaml_file = yaml.safe_load(yaml_string)
        return yaml_file


@dataclass
class ConfigRuns():
    name: str
    train_runs: int
    max_runs: int
    description: str
    hidden_size: float = 125.0
    infer_offpolicy: bool = True
    max_epochs: int = field(init=False, default=10)

    @staticmethod
    def file_to_configrun(path):
        return file_to_configobj(path, ConfigRuns)

    def asdict(self)->dict:
        return asdict(self)

@dataclass
class ConfigETensorVariable():
    shape: List[int]
    name: str
    flattened_shape:int =  field(init=False)
    def flattened(self):
        return np.prod(self.shape)
    def asdict(self)->dict:
        return asdict(self)
    def __post_init__(self):
        self.flattened_shape = self.flattened()

@dataclass
class ConfigFWorldVariables():
    """
    Handles the parsing of relevant parameter names from FWorld, have specific mapping
    """
    visionwide: ConfigETensorVariable
    gui_purposes: ConfigETensorVariable
    visionlocal: ConfigETensorVariable
    reward: ConfigETensorVariable
    object_seen: ConfigETensorVariable
    reward: ConfigETensorVariable #should consider normalization, or other characteristics that need to be done
    internal_state: ConfigETensorVariable
    best_action: ConfigETensorVariable
    sensory_local: ConfigETensorVariable
    sensory_local2: ConfigETensorVariable

    def asdict(self)->dict:
        return asdict(self)

def keep_relevant_keys(parsed_yaml: Dict[Any, Any], dataclass_obj: type) -> Dict[Any, Any]:
    copy_yaml = copy.deepcopy(parsed_yaml)
    for key in parsed_yaml:
        if (key in dataclass_obj.__dataclass_fields__) == False:
            del copy_yaml[key]
            warnings.warn("at least one additional key in yamlfile does not exist in ConfigFWorldVariables, "
                          "this may or may not be intentional, missing element called: {}".format(key), UserWarning)
    return copy_yaml

def file_to_fworldconfig(path:str)->ConfigFWorldVariables:
    shapes_fworld:ConfigFWorldVariables = file_to_dict(path)
    configFWorld: Dict[str,ConfigETensorVariable] = dict()
    for name in shapes_fworld:
        data_etensor:ConfigETensorVariable = dict_to_configobj(shapes_fworld[name],ConfigETensorVariable)
        configFWorld[name] = data_etensor
    return dict_to_configobj(configFWorld, ConfigFWorldVariables)

def test_flatten():
    #quick test to ensure basic operations work
    configFWorld = file_to_fworldconfig(os.path.join("config", "config_inputs.yaml"))
    assert configFWorld.sensory_local.flattened_shape == 8, "shape mismatch, default size has changed"

def test_configrun():
    config_run = ConfigRuns.file_to_configrun(os.path.join("config","run_config.yaml"))
    assert config_run!=None, "lazy assert, just check if it can run"

if __name__ == '__main__':
    test_flatten()
    test_configrun()


