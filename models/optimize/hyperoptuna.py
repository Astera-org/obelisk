import optuna
import json
import copy
import os
from optuna import Trial
import optimization


# This is some optuna specific logic, because it creates the min and the max if they don't exist in the Hypers.
def get_opt_value(trial: Trial, parametername, guidelines):
    val = float(guidelines["Val"])
    if guidelines.get("Min") is not None and guidelines.get("Max") is not None:
        print("GUIDELINES")
        print(guidelines)
        mino = float(guidelines["Min"])
        maxo = float(guidelines["Max"])
    # This is a bit of a hack because optuna wants a min and a max, but we might only specify a StdDev.
    elif guidelines.get("StdDev") is not None:
        mino = val - 2 * float(guidelines["StdDev"])
        maxo = val + 2 * float(guidelines["StdDev"])
    else:
        # TODO Make this better
        mino = val * .5
        maxo = val * 1.5
    return trial.suggest_float(parametername, mino, maxo)


# TODO Pull this into optimization.py
def create_suggested_params(params, trial):
    cparams = copy.deepcopy(params)
    parameters_to_modify = optimization.enumerate_parameters_to_modify(cparams)
    print("PARAMETERS TO MODIFY")
    print(parameters_to_modify)
    for info in parameters_to_modify:
        value_to_assign = get_opt_value(trial, info["uniquename"], info["values"]["Hypers"][info["paramname"]])
        info["values"]["Params"][info["paramname"]] = str(value_to_assign)
    # This creates a version of Params that has stripped out everything that didn't have Hypers
    updated_parameters = (optimization.create_hyperonly(cparams, "Searching"))
    print("UPDATED PARAMS")
    print(updated_parameters)
    return updated_parameters


def main():
    os.chdir('../')  # Move into the models/ directory
    params = optimization.get_hypers()

    # Study definition
    study = optuna.create_study(direction='minimize', study_name="Find_Some_Hypers")

    def optimize_optuna(trial: Trial):
        # TODO Pull this out into optimization.py
        print("BEGIN OPTIMIZE")
        # Create a new version of params that only has parameters with a Hypers annotation, with values chosen by trial.
        updated_parameters = create_suggested_params(params, trial)

        # Save the hyperparameters so that they can be read by the model
        with open("hyperparams.json", "w") as outfile:
            json.dump(updated_parameters, outfile)

        # Run go program with -params arg
        optimization.run_model(
            "-paramsFile=hyperparams.json -nogui=true -epclog=true -params=Searching -runs={0} -epochs={1}".format(
                str(optimization.NUM_RUNS), str(optimization.NUM_EPOCHS)))

        # Get valuation from logs
        return optimization.get_score_from_logs("Searching")

    # Starts optimization
    study.optimize(optimize_optuna, n_trials=optimization.NUM_TRIALS)
    print("BEST PARAMS")
    print(study.best_params)

    # TODO Create a full parameters set that uses the best params


if __name__ == '__main__':
    print("Starting optimization main func")
    main()
