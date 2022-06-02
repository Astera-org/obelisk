
import math
import time
from collections import OrderedDict

# NOTE bones must be pulled and installed locally from
# https://gitlab.com/generally-intelligent/bones/
from bones import BONES
from bones import BONESParams
from bones import ObservationInParam
from bones import LinearSpace


def func_to_optimize(vec):
    sum = 0
    i = 0
    for val in vec:
        i += 1
        # print(val)
        sugg = vec[val]
        if isinstance(sugg, str):
            continue
        sum += math.pow(i - sugg, i/3.0)
    return sum

def profile_bones():
    num_params = 5

    bone_params = BONESParams(
        better_direction_sign=True, initial_search_radius=0.5,
        is_wandb_logging_enabled=False, resample_frequency=-1
    )
    print("BONES PARAMS:", bone_params)

    centers = {}
    space = OrderedDict()
    for i in range(num_params):
        centers.update({"val_" + str(i): 0.0})
        space.update([("val_" + str(i), LinearSpace(scale=1, is_integer=False))])
    bones = BONES(bone_params, space)
    bones.set_search_center(centers)
    print("CENTERS:", centers)
    print("SPACE:", space)

    num_trials = 1000

    sugg_durations = []
    obs_durations = []

    for i in range(num_trials):
        sugg_start_time = time.time()
        print("STARTING TRIAL "+str(i))
        sugg = bones.suggest().suggestion
        print("GOT SUGG:", sugg)
        sugg_duration = time.time() - sugg_start_time
        val = func_to_optimize(sugg)
        print("GOT VAL:", val)
        observe_start_time = time.time()
        bones.observe(ObservationInParam(input=sugg, output=val))
        obs_duration = time.time() - observe_start_time
        print("SUGGEST TOOK TIME", sugg_duration)
        print("OBSERVATION TOOK TIME", obs_duration)
        sugg_durations.append(sugg_duration)
        obs_durations.append(obs_duration)
        if i % 100 == 0:
            print("SUGGESTION DURATIONS:", sugg_durations)
            print("OBSERVATION DURATIONS:", obs_durations)
    # In seconds
    print("SUGGESTION DURATIONS:", sugg_durations)
    print("OBSERVATION DURATIONS:", obs_durations)

profile_bones()