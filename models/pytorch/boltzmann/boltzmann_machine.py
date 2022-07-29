
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
