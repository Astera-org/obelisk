import collections
import sys
import time
from collections import OrderedDict
import json
import copy
import os

import numpy as np
import yaml
from decimal import *

import wandb

import optimization
import concurrent.futures
import threading

import os
import psutil
import configuration




# TODO REMOVE EXTRANEOUS PRINT STATEMENTS


# NOTE bones must be pulled and installed locally from
# https://gitlab.com/generally-intelligent/bones/
from bones import BONES
from bones import BONESParams
from bones import ObservationInParam
from bones import LinearSpace

OBSERVATIONS_FILE = "bones_obs.txt"  # todo doesn't seem to actually write to file


def print_cpu_usage():
    l1, l2, l3 = psutil.getloadavg()
    cpu_use = (l3 / os.cpu_count()) * 100
    print("CPU Usage: {} {} {}".format(l3, os.cpu_count(), cpu_use))


def get_bones_suggestion(current_suggestions: dict, parametername, guidelines):
    return current_suggestions[parametername]


# TODO Pull this into optimization.py
def create_bones_suggested_params(params, suggestions, trial_name: str):
    cparams = copy.deepcopy(params)
    parameters_to_modify = optimization.enumerate_parameters_to_modify(cparams)
    # print("PARAMETERS TO MODIFY IN BONES")
    # print(parameters_to_modify)
    for info in parameters_to_modify:
        value_to_assign = get_bones_suggestion(suggestions, info["uniquename"],
                                               info["values"]["Hypers"][info["paramname"]])
        info["values"]["Params"][info["paramname"]] = str(value_to_assign)
    # This creates a version of Params that has stripped out everything that didn't have Hypers
    updated_parameters = optimization.create_hyperonly(cparams, trial_name)
    # print(updated_parameters)
    return updated_parameters


# Bones needs the space to be defined at start.
def prepare_hyperparams_bones(the_params):
    hyperparams = optimization.enumerate_parameters_to_modify(the_params)
    initial_params = {}
    params_space_by_name = OrderedDict()
    for vals in hyperparams:
        uniquename = vals["uniquename"]
        relevantvalues = vals["values"]["Hypers"][vals["paramname"]]
        assert "Val" in relevantvalues, "a default value needs to exist in relevant values in hyperparmas"
        value = float(relevantvalues["Val"])
        # Assume that we always supply a standard deviation.
        stddev = value / 4.0  # Default
        if "StdDev" in relevantvalues:
            stddev = float(relevantvalues["StdDev"])
        if "Sigma" in relevantvalues:
            stddev = float(relevantvalues["Sigma"])
        # TODO Have a parameter for Linear/LogLinear/Etc.
        # TODO Allow integer and categorical spaces.
        is_int = "Type" in relevantvalues and relevantvalues["Type"] == "Int"
        distribution_type = LinearSpace(scale=stddev, is_integer=is_int)
        if "Min" in relevantvalues and "Max" in relevantvalues:
            distribution_type = LinearSpace(scale=stddev, min=float(relevantvalues["Min"]),
                                            max=float(relevantvalues["Max"]), is_integer=is_int)
        elif "Min" in relevantvalues:
            distribution_type = LinearSpace(scale=stddev, min=float(relevantvalues["Min"]), is_integer=is_int)
        elif "Max" in relevantvalues:
            distribution_type = LinearSpace(scale=stddev, min=float(relevantvalues["Max"]), is_integer=is_int)
        initial_params.update({uniquename: value})
        params_space_by_name.update([(uniquename, distribution_type)])

    return {"initial_params": initial_params, "paramspace_conditions": params_space_by_name}


def optimize_bones(params, suggestions: dict, trial_name: str):
    print("BEGIN OPTIMIZE " + str(trial_name))
    # This creates a version of Params that has stripped out everything that didn't have Hypers
    updated_parameters = create_bones_suggested_params(params, suggestions, trial_name)

    # Save the hyperparameters so that they can be read by the model
    hyperfile = "hyperparams{}.json".format(trial_name)
    with open(hyperfile, "w") as outfile:
        json.dump(updated_parameters, outfile)

    # Run go program with -params arg
    optimization.run_model(
        "-paramsFile={} -nogui=true -epclog=true -params={} -runs={} -epochs={} -randomize=true".format(
            hyperfile, trial_name, str(optimization.NUM_RUNS), str(optimization.NUM_EPOCHS)))

    # Get valuation from logs

    return optimization.get_score_from_logs(trial_name)


# def run_bones(bones_obj, trialnumber, params):
#     best_suggest = {}
#     best_score = sys.float_info.max
#     for i in range(trialnumber):
#         trial_name = "Searching_" + str(i)
#         current_suggestion = None
#         suggestions = bones_obj.suggest().suggestion
#         observed_value = optimize_bones(params, suggestions, trial_name)
#         # print(observed_value)
#         bones_obj.observe(ObservationInParam(input=suggestions, output=observed_value))
#         if observed_value < best_score:
#             best_score = observed_value
#             best_suggest = suggestions
#         with open(OBSERVATIONS_FILE, "a") as file_object:
#             file_object.write("\n{}\t{}\t{}\t{}".
#                               format(str(suggestions), str(observed_value), str(best_suggest), str(best_score)))
#     return best_suggest, best_score


class SimpleTimerObj():
    def __init__(self):
        self.start = time.time() * 1000
        self.end = -1

    def start_timer(self):
        self.start = time.time() * 1000

    def end_timer(self):
        self.end = float(int(time.time() * 1000))  # milliseconds are fine
        return float(int(self.end - self.start))

    def elapsed(self):
        return float(int(time.time() * 1000 - self.start))


all_observations = collections.deque()
total_timer = SimpleTimerObj()
all_times = []
observations_queue = collections.deque()


def single_bones_trial(bones_obj, params, lock, i):
    trial_timer = SimpleTimerObj()
    trial_name = "Searching_" + str(i)
    print("Starting trial: " + trial_name)
    with lock:
        suggest_timer = SimpleTimerObj()
        suggestions = bones_obj.suggest().suggestion
        print("Suggest timer: " + str(suggest_timer.end_timer()))
    print("TRYING THESE SUGGESTIONS " + str(suggestions))
    # print_cpu_usage()
    obtimize_timer = SimpleTimerObj()
    observed_value = optimize_bones(params, suggestions, trial_name)
    print("Optimization timer: " + str(obtimize_timer.end_timer()))
    print("GOT OBSERVED VALUE " + str(observed_value))
    with lock:
        observe_timer = SimpleTimerObj()
        # bones_obj.observe(ObservationInParam(input=suggestions, output=observed_value))
        observations_queue.append(ObservationInParam(input=suggestions, output=observed_value))
        print("Pure observe timer: " + str(observe_timer.elapsed()))
        all_observations.append((observed_value, suggestions, trial_name))
        print("Finished this many observations: " + str(len(all_observations)))
        # for so in all_observations:
        #     print(so[2] + " Score: " + str(so[0]) + " From Sugg: " + str(so[1]))
        best = sorted(all_observations, key=lambda a: a[0])[0]
        print("BEST RESULT: " + str(best))

        elapsed_time = trial_timer.end_timer()
        print("Trial timer: " + str(elapsed_time))
        all_times.append(elapsed_time)
        # all_times_np = np.array(all_times)
        all_obs_len = len(all_observations)
        print("Observation timer: " + str(observe_timer.end_timer()))
    # print(all_times_np)
    avg_time = total_timer.elapsed() / all_obs_len
    # print((all_times_np.mean()), (all_times_np.max()), int(all_times_np.min()))
    wandb.log({"runtime": trial_timer.elapsed(), "avgtime": avg_time, "totaltime": total_timer.elapsed()}, step=i)  # TODO Is this right? Is it incrementing badly? Use i
    print("Average elapsed timer: " + str(avg_time))
    print("Full timer: " + str(trial_timer.end_timer()))


# TODO Step isn't maintained correctly, so observations won't match up with other logged parameters
def observer(bones_obj, lock):
    global observations_queue
    num_observed = 0
    while True:
        # print("OBSERVER QUEUE " + str(observations_queue))
        while observations_queue:
            obs = observations_queue.pop()
            print("OBSERVER OBS: " + str(obs))
            with lock:
                bones_obj.observe(obs)
            num_observed += 1
        # os.sleep(1)
        # print("OBSERVER DONE SLEEPING")
        if num_observed >= optimization.NUM_TRIALS:
            # All done.
            return


def run_bones_parallel(bones_obj, params):
    locky = threading.Lock()
    global total_timer
    total_timer.start_timer()
    with concurrent.futures.ThreadPoolExecutor(max_workers=optimization.NUM_PARALLEL) as executor:
        # TODO This will fail if NUM_PARALLEL is 1
        executor.submit(observer, bones_obj, locky)
        for i in range(optimization.NUM_TRIALS):
            print("Starting to execute: " + str(i))
            future = executor.submit(single_bones_trial, bones_obj, params, locky, i)

    best = sorted(all_observations, key=lambda a: a[0])[0]
    return best[1], best[0]


def loadyaml(name):
    with open(name, 'r') as file:
        return yaml.safe_load(file)


def main():
    os.chdir('../')  # Move into the models/ directory
    params = optimization.get_hypers()

    prep_params_dict = prepare_hyperparams_bones(params)
    initial_params = prep_params_dict["initial_params"]
    params_space_by_name = prep_params_dict["paramspace_conditions"]

    better_direction_sign = -1 if optimization.MINIMIZE else 1
    #print("BETTER DIRECTION: " + str(better_direction_sign))
    bone_params = BONESParams(
        better_direction_sign=better_direction_sign, is_wandb_logging_enabled=optimization.WANDLOGGING, initial_search_radius=0.5, resample_frequency=-1
    )

    assert len(params_space_by_name) > 0

    bones = BONES(bone_params, params_space_by_name)
    bones.set_search_center(initial_params)
    if optimization.WANDLOGGING:
        wandb.log({"numtrials": optimization.NUM_TRIALS, "numparallel": optimization.NUM_PARALLEL,
                   "numepochs": optimization.NUM_EPOCHS})
    best, best_score = run_bones_parallel(bones, params)
    print("Best parameters at: " + str(best) + " with score: " + str(best_score))
    print("FINAL TIME", str(total_timer.end_timer()))


def load_key(config_path="bone_config.yaml"):
    config_file = loadyaml(config_path)
    wandb_key = config_file["wandb_key"]
    return wandb_key


if __name__ == '__main__':
    configObj: configuration.ConfigOptimizer = configuration.file_to_configobj("../configs/bone_config.yaml")
    configuration.assign_to_optimizer_constants(configObj)

    if configObj.use_onlinelogging:
        wandb.login(key=configObj.wandb_key)

    print("Starting optimization main func")
    main()
    print("FINAL TIME", str(total_timer.elapsed()))