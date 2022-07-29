import torch
import numpy as np
import scipy.stats as st
import boltzmann_machine
from hyperparams import HParams
import datasets


def calc_nearest_example_index(predicted: torch.FloatTensor, possible_targets: torch.FloatTensor):
    #make this into a batch operaiton later, but blobby
    nearest_indices = []
    for example in predicted:
        min_index = torch.argmin((possible_targets - example).abs().sum(dim=1))
        nearest_indices.append(min_index)
    return torch.Tensor(nearest_indices)


# Get data, create a network, and then run it, collecting a lot of performance data on the way.
def create_and_run_network(params: HParams = HParams()):
    xs, ys, input_size, hidden_size, output_size, num_data = datasets.get_data(params)

    boltzy = boltzmann_machine.BoltzmannMachine(input_size, hidden_size, output_size, params)

    all_distances = []
    all_h_distances = []
    all_classification = []
    first_correct = max(params.epochs+1, 999999)
    first_correct_start_of_run = first_correct
    for epoch in range(params.epochs):
        distances = []
        h_distances = []
        classification = []

        if params.batch_data == True:
            num_data = 1 #all data is run at once

        for data_row in range(num_data):
            index = data_row % num_data
            if params.batch_data == False:
                x = torch.unsqueeze(xs[index],dim=0)
                y = torch.unsqueeze(ys[index],dim=0)
            else:
                x = xs
                y = ys
            # TODO First 4 should have learning off. Michael: What does this comment mean?
            acts_clamp_x, acts_clamp_y = boltzy.run_minus_and_plus(x, y)

            # Analytics
            dist = boltzy.y_distance(acts_clamp_x, acts_clamp_y) # For reporting, not used for training
            h_dist = boltzy.h_distance(acts_clamp_x, acts_clamp_y)
            predicted_ys = acts_clamp_x[:,boltzy.input_size:boltzy.input_size+boltzy.output_size]

            predicted_indices = calc_nearest_example_index(predicted_ys, ys)

            correct = (((ys[(predicted_indices).long()] - y).abs().sum(dim=1)) == 0).sum()/len(predicted_indices) #so can handle batches
            #correct = ((ys[predicted_index] - y).abs().sum() == 0) # same class predicted
            #print(ys[(predicted_indices).long()])
            classification.append(correct)
            if params.verbose >= 5:
                print("Clamp X:    ", acts_clamp_x.detach())
                print("Clamp X, Y: ", acts_clamp_y.detach())
                print("Correct? ", correct.item(), "\n")
            # print("Y Distance: ", dist)
            # print("H Distance: ", h_dist)
            # print("Weights: ", boltzy.layer.weight)
            distances.append(dist)
            h_distances.append(h_dist)

            if params.epochs == 1:
                assert False, "first epoch has no training to get baseline, needs more to be meaningful"
            if epoch >= 1:
                boltzy.delta_rule_update_weights_matrix(acts_clamp_x, acts_clamp_y)

            # if epoch >= (params.epochs-1):
            #     print("end",predicted_y.detach().numpy(),y)
            # elif epoch == 0:
            #     print("start",predicted_y.detach().numpy(), y)
        all_distances.append(distances)
        all_classification.append(classification)
        all_h_distances.append(h_distances)
        percent_correct = (torch.Tensor(classification).sum()/len(classification)).detach().numpy()
        if percent_correct < 1.0:
            first_correct = max(params.epochs+1, 999999)
        else:
            first_correct = min(first_correct, epoch)
            if epoch >= first_correct + params.stopping_success:
                first_correct_start_of_run = first_correct
                if params.verbose >= params.stopping_success:
                    print("Hooray! Got", params.stopping_success, "successes in a row starting at time: ", first_correct)
                break

    # print("X: ", xs, " Y: ", ys)
    # print("Weights: ", boltzy.layer.weight)

    final_correct = torch.Tensor(all_classification[-1])
    initial_correct = torch.Tensor(all_classification[0])

    final_percent_correct = (final_correct.sum()/len(final_correct)).detach().numpy()
    initial_percent_correct = (initial_correct.sum()/len(initial_correct)).detach().numpy()
    initial_score = torch.Tensor(all_distances[0]).mean().detach().numpy()

    final_score = torch.Tensor(all_distances[-1]).mean().detach().numpy()
    if params.num_runs == 1 and params.verbose > 0:
        print("End correct ", final_percent_correct, "Start correct", initial_percent_correct, "End Dist: ", final_score, " compared to initial score: ", initial_score)
        if first_correct_start_of_run < params.epochs:
            print("Got first correct score in a run of at least", params.stopping_success, "at timestep: ", first_correct_start_of_run)
        else:
            print("It never converged to 100% correct :(")
        print("End distance: ", final_score, " compared to initial score: ", initial_score)
        print("End H distance: ", torch.Tensor(all_h_distances[0]).mean().numpy(), " compared to initial score: ", torch.Tensor(all_h_distances[-1]).mean().numpy())
        # print("End weights: ", boltzy.layer.weight)
    if params.score == "distance":
        return final_score
    if params.score == "distance_improvement":
        return final_score - initial_score
    if params.score == "h_distance":
        return torch.Tensor(all_h_distances[0]).mean().numpy()
    elif params.score == "perc_correct":
        return final_percent_correct
    elif params.score == "convergence":
        return first_correct_start_of_run
    else:
        assert False, "Invalid params.score: " + params.score


def run_many_times(params: HParams):
    if params.verbose >= 0:
        print("\nWith params: ", params)
    scores = []
    number_runs = params.num_runs
    for ii in range(number_runs):
        final_score = create_and_run_network(params)
        scores.append(final_score)
    total_score = sum(scores) / len(scores)
    confidence_bars = st.norm.interval(alpha=0.95, loc=np.mean(scores), scale=st.sem(scores)) if len(scores) > 1 else (float("nan"), float("nan")) # It prints an annoying warning if you give it a single element list
    if params.verbose >= 0:
        if params.verbose >= 1:
            print("All scores: ", ["%.2f" % x.item() if hasattr(x, "item") else x for x in scores])
        if len(scores) > 1:
            print("Got", params.score, ": ", "%.2f" % total_score, " Confidence Bars: ", confidence_bars)
        else:
            print("Got", params.score, ": ", "%.2f" % total_score)
    return total_score, confidence_bars
