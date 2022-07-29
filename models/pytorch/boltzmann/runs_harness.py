import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from scipy import spatial
import statistics
import scipy.stats as st
import boltzmann_machine


def calc_nearest_example_index(predicted:torch.FloatTensor,possible_targets:torch.FloatTensor):
    all_targets = torch.concat(possible_targets)
    min_index = torch.argmin((all_targets - predicted).abs().sum(dim=1))
    return min_index


def create_and_run_network(params=None):
    if params is None:
        params = {}

    # Network
    input_size = 3
    output_size = 2
    hidden_size = params["hidden_size"]

    # XOR (input and output sizes different)
    xs = [torch.tensor([[0, 0, 1]]), torch.tensor([[0, 1, 1]]), torch.tensor([[1, 0, 1]]), torch.tensor([[1, 1, 1]])] # Third input is bias
    ys = []
    y_xor = [torch.tensor([[0, 0]]), torch.tensor([[1, 1]]), torch.tensor([[1, 1]]), torch.tensor([[0, 0]])] # This weird binumeral format helps with cosine similarity
    y_and = [torch.tensor([[0, 1]]), torch.tensor([[0, 1]]), torch.tensor([[0, 1]]), torch.tensor([[1, 0]])]
    y_or = [torch.tensor([[0, 1]]), torch.tensor([[1, 0]]), torch.tensor([[1, 0]]), torch.tensor([[1, 0]])]
    if "io" not in params or params["io"] == "random":
        num_data = 4 if "num_data" not in params else params["num_data"]
        input_size = params["input_size"] if "input_size" in params else input_size
        output_size = params["output_size"] if "output_size" in params else output_size
        xs = [torch.rand(size=(1, input_size)) for _ in range(num_data)]
        ys = [torch.rand(size=(1, output_size)) for _ in range(num_data)]
    elif params["io"] == "xor":
        ys = y_xor
    elif params["io"] == "and":
        ys = y_and
    elif params["io"] == "or":
        ys = y_or
    elif params["io"] == "ra25":
        num_data = 25
        choose_n = 6
        input_size = 25
        output_size = 25
        xs = [torch.tensor([random.sample([1] * choose_n + [0] * (input_size - choose_n), input_size)]) for _ in range(num_data)]
        ys = [torch.tensor([random.sample([1] * choose_n + [0] * (output_size - choose_n), output_size)]) for _ in range(num_data)]
    num_data = len(xs)

    boltzy = boltzmann_machine.BoltzmannMachine(input_size, hidden_size, output_size, params)

    all_distances = []
    all_h_distances = []
    all_classification = []
    first_correct = max(params["epochs"]+1, 999999)
    first_correct_start_of_run = first_correct
    for epoch in range(params["epochs"]):
        distances = []
        h_distances = []
        classification = []

        for data_row in range((num_data)):
            index = data_row % num_data
            x = xs[index]
            y = ys[index]

            # TODO First 4 should have learning off. Michael: What does this mean?
            acts_clamp_x, acts_clamp_y = boltzy.run_minus_and_plus(x, y)

            # Analytics
            dist = boltzy.y_distance(acts_clamp_x, acts_clamp_y) # For reporting, not used for training
            h_dist = boltzy.h_distance(acts_clamp_x, acts_clamp_y)
            predicted_y = acts_clamp_x[0][boltzy.input_size:boltzy.input_size+boltzy.output_size]
            predicted_index = calc_nearest_example_index(predicted_y, ys)
            correct = ((ys[predicted_index] - y).abs().sum() == 0) # same class predicted
            classification.append(correct)
            if "verbose" in params and params["verbose"] >= 5:
                print("Clamp X:    ", acts_clamp_x.detach())
                print("Clamp X, Y: ", acts_clamp_y.detach())
                print("Correct? ", correct.item(), "\n")
            # print("Y Distance: ", dist)
            # print("H Distance: ", h_dist)
            # print("Weights: ", boltzy.layer.weight)
            distances.append(dist)
            h_distances.append(h_dist)

            if params["epochs"] == 1:
                assert False, "first epoch has no training to get baseline, needs more to be meaningful"
            if epoch >= 1:
                boltzy.delta_rule_update_weights_matrix(acts_clamp_x, acts_clamp_y)

            # if epoch >= (params["epochs"]-1):
            #     print("end",predicted_y.detach().numpy(),y)
            # elif epoch == 0:
            #     print("start",predicted_y.detach().numpy(), y)
        all_distances.append(distances)
        all_classification.append(classification)
        all_h_distances.append(h_distances)
        percent_correct = (torch.Tensor(classification).sum()/len(classification)).detach().numpy()
        if percent_correct < 1.0:
            first_correct = max(params["epochs"]+1, 999999)
        else:
            first_correct = min(first_correct, epoch)
            if epoch >= first_correct + 5:
                first_correct_start_of_run = first_correct
                if "verbose" in params and params["verbose"] >= 5:
                    print("Hooray! Got 5 successes in a row starting at time: ", first_correct)
                break

    # print("X: ", xs, " Y: ", ys)
    # print("Weights: ", boltzy.layer.weight)

    final_correct = torch.Tensor(all_classification[-1])
    initial_correct = torch.Tensor(all_classification[0])

    final_percent_correct = (final_correct.sum()/len(final_correct)).detach().numpy()
    initial_percent_correct = (initial_correct.sum()/len(initial_correct)).detach().numpy()
    initial_score = torch.Tensor(all_distances[0]).mean().detach().numpy()

    final_score = torch.Tensor(all_distances[-1]).mean().detach().numpy()
    if params["num_runs"] == 1:
        print("End correct ", final_percent_correct, "Start correct",initial_percent_correct ,"End Dist: ", final_score, " compared to initial score: ", initial_score)
        if first_correct_start_of_run < params["epochs"]:
            print("Got first correct score in a run of at least 5 at timestep: ", first_correct_start_of_run)
        else:
            print("It never converged to 100% correct :(")
        print("End distance: ", final_score, " compared to initial score: ", initial_score)
        print("End H distance: ", torch.Tensor(all_h_distances[0]).mean().numpy(), " compared to initial score: ",torch.Tensor(all_h_distances[-1]).mean().numpy())
        # print("End weights: ", boltzy.layer.weight)
    return first_correct_start_of_run


def run_many_times(params):
    print("\nWith params: ", params)
    scores = []
    number_runs = params["num_runs"]
    for ii in range(number_runs):
        final_score = create_and_run_network(params)
        scores.append(final_score)
    total_score = sum(scores) / len(scores)
    print("All scores: ", ["%.2f" % x.item() if hasattr(x, "item") else x for x in scores])
    print("Got score: ", "%.2f" % total_score, " Confidence Bars: ", st.norm.interval(alpha=0.95, loc=np.mean(scores), scale=st.sem(scores)) if len(scores) > 1 else "(nan, only one run)")
    return total_score
