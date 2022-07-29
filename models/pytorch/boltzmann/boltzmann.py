import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from scipy import spatial
import statistics
import scipy.stats as st


class BoltzmannMachine(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, params):
        super().__init__()
        self.params = params
        self.layer_size = input_size + output_size + hidden_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = 0.1 if "learning_rate" not in self.params else self.params["learning_rate"]
        self.layer = nn.Linear(self.layer_size, self.layer_size, bias=False)
        with torch.no_grad():
            self.layer.weight = self.layer.weight.fill_diagonal_(0) # No self connections
        # for ii in range(self.layer_size):
        #     for jj in range(self.layer_size):
        #         with torch.no_grad():
        #             if ii < jj:
        #                 self.layer.weight[ii, jj] = self.layer.weight[jj, ii] # Start symmetric # This seems to make it worse

    def forward(self, x, y, n, clamp_x=False, clamp_y=False):
        h: torch.Tensor = torch.zeros(size=(1, self.hidden_size))
        if clamp_y:
            full_act = torch.concat([x, y, h], dim=1)
        else:
            full_act = torch.concat([x, torch.zeros_like(y), h], dim=1) # Y shouldn't be present here. Use zeros
        record = None
        if "average_window" in self.params:
            record = torch.zeros(size=(self.params["average_window"], self.layer_size))
        for ii in range(n):
            full_act = F.relu(self.layer(full_act))
            if clamp_x:
                full_act[0, 0:self.input_size] = x
            if clamp_y:
                full_act[0, self.input_size:self.input_size+self.output_size] = y
            # print("Act: ", full_act.detach())
            if "norm_act" in self.params and self.params["norm_act"] is True:
                full_act = F.normalize(full_act, p=2.0, dim=1)
            if "norm_hidden" not in self.params or self.params["norm_hidden"] is True:
                hidden = full_act[0, self.input_size + self.output_size:]
                hidden = F.normalize(hidden, p=2.0, dim=0)
                with torch.no_grad():
                    full_act[0, self.input_size + self.output_size:] = hidden
            if record is not None:
                record[ii % record.size(0), :] = full_act
            if "verbose" in self.params and self.params["verbose"] >= 5:
                print("Normed vec: ", self.print_activity(full_act.detach()))
        # TODO Maybe take a running average here, because full_act seems like it might alternate with period>1
        # print(torch.mean(record, 0))
        if "average_window" not in self.params:
            return full_act
        else:
            # TODO Evaluate the benefit of this
            return torch.mean(record, 0, True)

    def print_activity(self, activity):
        s = "In: "
        for i in range(self.input_size):
            s += str("%.2f" % activity[0, i].item()) + ", "
        s += "\tOut: "
        for i in range(self.input_size, self.input_size + self.output_size):
            s += str("%.2f" % activity[0, i]) + ", "
        s += "\tHid: "
        for i in range(self.input_size + self.output_size, self.layer_size):
            s += str("%.2f" % activity[0, i]) + ", "
        return s

    def norm_weights(self): # Normalize inputs to neurons
        if "norm_weights" not in self.params or self.params["norm_weights"] is True:
            F.normalize(self.layer.weight, p=2.0, dim=0, out=self.layer.weight)  # Normalize inputs to neurons

    def delta_rule_update_weights_matrix(self, minus_phase:torch.Tensor, plus_phase:torch.Tensor):
        # Rule: delta_w = x'y' - xy # Primes taken from acts_x_y, normals taken from acts_x. x and y here don't correspond to the other x and y, they're the pre- and post-neurons, so it's for all pairs. Sorry for that notation.
        # print("Weights before adjustment: ", self.layer.weight)
        assert ((len(minus_phase.shape)) == 2) & (minus_phase.shape[0] == 1)
        assert ((len(plus_phase.shape)) == 2) & (plus_phase.shape[0] == 1)
        # Equivalent to this:
        # for ii in range(acts_x.size(dim=1)):  # Pre, X
        #     for jj in range(acts_x_y.size(dim=1)):  # Post, Y # Should be the same length, square matrix
        #         if ii == jj:
        #             continue # 0 on diagonal
        #         sender_plus = acts_x_y[0, ii]
        #         receiver_plus = acts_x_y[0, jj]
        #         sender_minus = acts_x[0, ii]
        #         receiver_minus = acts_x[0, jj]
        #         # Contrastive Hebbian Learning
        #         delta = (sender_plus * receiver_plus) - (sender_minus * receiver_minus)
        #         self.layer.weight[ii, jj] += self.learning_rate * delta
        with torch.no_grad():
            minus_mult = torch.mm(minus_phase.T, minus_phase).fill_diagonal_(0) # no self correlation, multiply incoming times outgoing
            plus_mult = torch.mm(plus_phase.T, plus_phase).fill_diagonal_(0)
            self.layer.weight[:] = self.layer.weight[:] + self.learning_rate * (plus_mult - minus_mult)
            self.norm_weights()

    def y_distance(self, act1, act2):
        y1 = act1[0, self.input_size:self.input_size+self.output_size]
        y2 = act2[0, self.input_size:self.input_size+self.output_size]
        # print("Y clamp x:   ", y1)
        # print("Y clamp x,y: ", y2)
        # return 1-spatial.distance.cosine(y1.detach().numpy(), y2.detach().numpy())
        return (y1 - y2).abs().sum().detach()

    def h_distance(self, act1, act2):
        h1 = act1[0, self.input_size+self.output_size:]
        h2 = act2[0, self.input_size+self.output_size:]
        # return 1-spatial.distance.cosine(h1.detach().numpy(), h2.detach().numpy())
        return (h1 - h2).abs().sum().detach()

    def run_minus_and_plus(self, x, y):
        if "verbose" in self.params and self.params["verbose"] >= 5:
            print("Starting Minus Phase")
        acts_clamp_x = self(x, y, self.params["num_rnn_steps"], clamp_x=True) # Minus phase
        if "verbose" in self.params and self.params["verbose"] >= 5:
            print("Starting Plus Phase")
        acts_clamp_y = self(x, y, self.params["num_rnn_steps"], clamp_x=True, clamp_y=True) # Plus phase
        return acts_clamp_x, acts_clamp_y


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

    boltzy = BoltzmannMachine(input_size, hidden_size, output_size, params)

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


# TODO params should be an object so it can have defaults
if __name__ == '__main__':
    # torch.manual_seed(2) # For debugging purposes.
    torch.set_printoptions(precision=3, sci_mode=False)
    num_runs = 10
    epochs = 100

    # Only one run
    run_many_times({"epochs": epochs, "hidden_size": 2, "num_rnn_steps": 5, "num_runs": 1, "io": "xor", "verbose": 0, "norm_weights": True, "learning_rate": 0.1})

    # # Look at learning rate for large num_data
    # run_many_times({"epochs": epochs, "hidden_size": 10, "num_rnn_steps": 5, "num_runs": 5, "io": "random", "verbose": 0, "norm_weights": True, "input_size": 10, "output_size": 10, "learning_rate": 0.1, "num_data": 10})
    # run_many_times({"epochs": epochs, "hidden_size": 10, "num_rnn_steps": 5, "num_runs": 5, "io": "random", "verbose": 0, "norm_weights": False, "input_size": 10, "output_size": 10, "learning_rate": 0.01, "num_data": 10})

    # # Try to solve random
    # run_many_times({"epochs": epochs, "hidden_size": 10, "num_rnn_steps": 5, "num_runs": num_runs, "io": "random"})
    # run_many_times({"epochs": epochs, "hidden_size": 10, "num_rnn_steps": 10, "num_runs": num_runs, "io": "random"})
    # run_many_times({"epochs": epochs, "hidden_size": 20, "num_rnn_steps": 10, "num_runs": num_runs, "io": "random"})
    # run_many_times({"epochs": epochs, "hidden_size": 20, "num_rnn_steps": 20, "num_runs": num_runs, "io": "random"})
    # run_many_times({"epochs": epochs, "hidden_size": 50, "num_rnn_steps": 20, "num_runs": num_runs, "io": "random"})

    # Test XOR
    # run_many_times({"epochs":20,"epochs": epochs, "hidden_size": 4, "num_rnn_steps": 5, "num_runs": 1, "io": "random"})
    # run_many_times({"epochs": 50, "hidden_size": 5, "num_rnn_steps": 5, "num_runs": 1, "io": "xor"})
    # run_many_times({"epochs": 50, "hidden_size": 5, "num_rnn_steps": 5, "num_runs": 1, "io": "and"})
    # run_many_times({"epochs": 50, "hidden_size": 5, "num_rnn_steps": 5, "num_runs": 1, "io": "or"})
    # #run_many_times({"epochs": 50, "hidden_size": 5, "num_rnn_steps": 5, "num_runs": 1, "io": "random"})
    # # run_many_times({"epochs": epochs, "hidden_size": 4, "num_rnn_steps": 5, "num_runs": 1, "io": "random"})

    # # Test XOR
    # run_many_times({"epochs": epochs, "hidden_size": 5, "num_rnn_steps": 5, "num_runs": num_runs, "io": "xor"})
    # run_many_times({"epochs": epochs, "hidden_size": 5, "num_rnn_steps": 5, "num_runs": num_runs, "io": "and"})
    # run_many_times({"epochs": epochs, "hidden_size": 5, "num_rnn_steps": 5, "num_runs": num_runs, "io": "or"})
    # run_many_times({"epochs": epochs, "hidden_size": 5, "num_rnn_steps": 5, "num_runs": num_runs, "io": "random"})

    # # Test normalization in XOR
    # # First XOR has both
    # run_many_times({"epochs": epochs, "hidden_size": 5, "num_rnn_steps": 5, "num_runs": num_runs, "io": "xor", "norm_weights": False})
    # run_many_times({"epochs": epochs, "hidden_size": 5, "num_rnn_steps": 5, "num_runs": num_runs, "io": "xor", "norm_act": False})
    # run_many_times({"epochs": epochs, "hidden_size": 5, "num_rnn_steps": 5, "num_runs": num_runs, "io": "xor", "norm_weights": False, "norm_act": False})

    # Test RNN averaging # This seems to not make a difference
    # run_many_times({"epochs": epochs, "hidden_size": 5, "num_rnn_steps": 5, "num_runs": num_runs, "io": "random"})
    # run_many_times({"epochs": epochs, "hidden_size": 5, "num_rnn_steps": 5, "num_runs": num_runs, "io": "random", "average_window": 3})

    # run_many_times({"epochs": epochs, "hidden_size": 0, "num_rnn_steps": 1, "num_runs": num_runs)
    # run_many_times({"epochs": epochs, "hidden_size": 2, "num_rnn_steps": 1, "num_runs": num_runs)
    # run_many_times({"epochs": epochs, "hidden_size": 5, "num_rnn_steps": 1, "num_runs": num_runs)
    # run_many_times({"epochs": epochs, "hidden_size": 10, "num_rnn_steps": 1, "num_runs": num_runs)
    # run_many_times({"epochs": epochs, "hidden_size": 0, "num_rnn_steps": 5, "num_runs": num_runs)
    # run_many_times({"epochs": epochs, "hidden_size": 2, "num_rnn_steps": 5, "num_runs": num_runs)
    # run_many_times({"epochs": epochs, "hidden_size": 5, "num_rnn_steps": 5, "num_runs": num_runs)
    # run_many_times({"epochs": epochs, "hidden_size": 10, "num_rnn_steps": 5, "num_runs": num_runs)
    # run_many_times({"epochs": epochs, "hidden_size": 0, "num_rnn_steps": 10, "num_runs": num_runs)
    # run_many_times({"epochs": epochs, "hidden_size": 2, "num_rnn_steps": 10, "num_runs": num_runs)
    # run_many_times({"epochs": epochs, "hidden_size": 5, "num_rnn_steps": 10, "num_runs": num_runs)
    # run_many_times({"epochs": epochs, "hidden_size": 10, "num_rnn_steps": 10, "num_runs": num_runs)

# NOTES
# ☑️ Evaluate Y similarity in two cases
# ☑️ Many different X,Y pairs
# Experimental setup to try hyperparameters, like hidden layer size
# Hypothesis: After learning enough, the Y component of acts_clamp_x will approach the true Y
