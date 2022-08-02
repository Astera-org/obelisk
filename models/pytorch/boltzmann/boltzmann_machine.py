import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import linear
from hyperparams import HParams


def create_symmetric_weights(weights: torch.Tensor):
    """
    make the weights symmetric, if we want to enforce this
    """

    # for ii in range(self.layer_size):
    #     for jj in range(self.layer_size):
    #         with torch.no_grad():
    #             if ii < jj:
    #                 self.layer.weight[ii, jj] = self.layer.weight[jj, ii] # Start symmetric # This seems to make it worse

    assert weights.shape[0] == weights.shape[1], "expected square matrix"

    n = weights.shape[0]
    with torch.no_grad():
        vals = torch.triu(weights).flatten()[:int(n * (n + 1) / 2)]
        new_mat = torch.zeros_like(weights)
        i, j = torch.triu_indices(n, n)
        new_mat[i, j] = vals
        new_mat.T[i, j] = vals

    assert (new_mat.T == new_mat).sum() == len(weights.flatten()), "weights are not symmetric"
    return new_mat


class BoltzmannMachine(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, params: HParams):
        super().__init__()
        self.params: HParams = params
        self.layer_size = input_size + output_size + hidden_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = params.learning_rate
        self.self_connection_strength = params.self_connection_strength
        self.layer = nn.Linear(self.layer_size, self.layer_size, bias=False)
        with torch.no_grad():
            self.layer.weight = self.layer.weight.fill_diagonal_(0)  # No self connections

            if params.weights_start_symmetric:
                self.layer.weight[:] = create_symmetric_weights(self.layer.weight)

        self._activation_strength_matrix = torch.zeros_like(self.layer.weight)
        self.set_activation_strength(forward_strength=params.forward_connection_strength,
                                     backward_strength=params.backward_connection_strength,
                                     lateral_strength=params.lateral_connection_strength,
                                     self_connect_strength=params.self_connection_strength)

    def set_activation_strength(self, forward_strength: float, backward_strength: float, lateral_strength: float,
                                self_connect_strength: float):
        """
        used to vary how strong activation is dampened or strengthened based on the direction it is coming from
        """
        activation_strength_mat = self._activation_strength_matrix
        # note this assumes order is input, output, hidden connections
        input_length, output_length, hidden_length = self.input_size, self.output_size, self.hidden_size
        upper = torch.triu_indices(activation_strength_mat.shape[0], activation_strength_mat.shape[1])
        lower = torch.tril_indices(activation_strength_mat.shape[0], activation_strength_mat.shape[1])

        activation_strength_mat[upper[0], upper[1]] = forward_strength
        activation_strength_mat[lower[0], lower[1]] = backward_strength

        end_input = input_length
        end_output = input_length + output_length
        end_hidden = input_length + output_length + hidden_length  # this is same as length as full layer size

        activation_strength_mat[0:end_input, 0:end_input] = lateral_strength
        activation_strength_mat[end_input:end_output, end_input:end_output] = lateral_strength
        activation_strength_mat[end_output:end_hidden, end_output:end_hidden] = lateral_strength
        activation_strength_mat.fill_diagonal_(self_connect_strength)

        assert (activation_strength_mat >= 0).sum() == len(
            activation_strength_mat.flatten()), "all strengths are expected to be greater than 0"

    def forward(self, x, y, n, clamp_x=False, clamp_y=False):
        h: torch.Tensor = torch.zeros(size=(x.shape[0], self.hidden_size))
        if clamp_y:
            full_act = torch.concat([x, y, h], dim=1)
        else:
            full_act = torch.concat([x, torch.zeros_like(y), h], dim=1)  # Y shouldn't be present here. Use zeros
        record = None
        if self.params.average_window > 0:
            record = torch.zeros(size=(self.params.average_window, self.layer_size))
        for ii in range(n):
            with torch.no_grad():
                modified_weights = self.layer.weight * self._activation_strength_matrix.T  # since F linear is doing A.T

                full_act = F.relu(linear(full_act, modified_weights))
                if clamp_x:
                    full_act[:, 0:self.input_size] = x
                if clamp_y:
                    full_act[:, self.input_size:self.input_size + self.output_size] = y
                # print("Act: ", full_act.detach())
                if self.params.norm_hidden is True:
                    hidden = full_act[:, self.input_size + self.output_size:]
                    hidden = F.normalize(hidden, p=2.0, dim=1)
                    with torch.no_grad():
                        full_act[:, self.input_size + self.output_size:] = hidden
                if record is not None:
                    record[ii % record.size(0), :] = full_act
                # full_act = torch.where(full_act > 0.2, 1.0, 0.0) # Binarizing the vector. # TODO Run a test or something
                if self.params.verbose >= 5:
                    print("Normed vec: ", self.print_activity(full_act.detach()))
            # TODO Maybe take a running average here, because full_act seems like it might alternate with period>1
        # print(torch.mean(record, 0))
        if self.params.average_window <= 0:
            return full_act
        else:
            # TODO(andrew) Evaluate the benefit of this
            return torch.mean(record, 0, True)  # TODO(michael): verify this is correct

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

    def norm_weights(self):  # Normalize inputs to neurons
        if self.params.norm_weights is True:
            F.normalize(self.layer.weight, p=2.0, dim=0, out=self.layer.weight)  # Normalize inputs to neurons

    def delta_rule_update_weights_matrix(self, minus_phase: torch.Tensor, plus_phase: torch.Tensor):
        # Rule: delta_w = x'y' - xy # Primes taken from acts_x_y, normals taken from acts_x. x and y here don't correspond to the other x and y, they're the pre- and post-neurons, so it's for all pairs. Sorry for that notation.
        # print("Weights before adjustment: ", self.layer.weight)
        if not self.params.batch_data:
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
            minus_mult = torch.mm(minus_phase.T, minus_phase).fill_diagonal_(self.self_connection_strength) / \
                         minus_phase.shape[0]  # no self correlation, multiply incoming times outgoing
            plus_mult = torch.mm(plus_phase.T, plus_phase).fill_diagonal_(self.self_connection_strength) / \
                        plus_phase.shape[0]
            self.layer.weight[:] = self.layer.weight[:] + (self.learning_rate * (plus_mult - minus_mult))
            self.norm_weights()

    def y_distance(self, act1, act2):
        y1 = act1[:, self.input_size:self.input_size + self.output_size]
        y2 = act2[:, self.input_size:self.input_size + self.output_size]
        # print("Y clamp x:   ", y1)
        # print("Y clamp x,y: ", y2)
        # return 1-spatial.distance.cosine(y1.detach().numpy(), y2.detach().numpy())
        return (y1 - y2).abs().sum().detach()

    def h_distance(self, act1, act2):
        h1 = act1[:, self.input_size + self.output_size:]
        h2 = act2[:, self.input_size + self.output_size:]
        # return 1-spatial.distance.cosine(h1.detach().numpy(), h2.detach().numpy())
        return (h1 - h2).abs().sum().detach()

    def run_minus_and_plus(self, x, y):
        if self.params.verbose >= 5:
            print("Starting Minus Phase")
        acts_clamp_x = self(x, y, self.params.num_rnn_steps, clamp_x=True)  # Minus phase
        if self.params.verbose >= 5:
            print("Starting Plus Phase")
        acts_clamp_y = self(x, y, self.params.num_rnn_steps, clamp_x=True, clamp_y=True)  # Plus phase
        return acts_clamp_x, acts_clamp_y
