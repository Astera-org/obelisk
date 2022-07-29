import torch
from parameters import Parameters
import random


def get_data(params: Parameters):

    # Network
    input_size = 3
    output_size = 2
    hidden_size = params.hidden_size

    # XOR (input and output sizes different)
    xs = [torch.tensor([[0, 0, 1]]), torch.tensor([[0, 1, 1]]), torch.tensor([[1, 0, 1]]), torch.tensor([[1, 1, 1]])] # Third input is bias
    ys = []
    y_xor = [torch.tensor([[0, 0]]), torch.tensor([[1, 1]]), torch.tensor([[1, 1]]), torch.tensor([[0, 0]])] # This weird binumeral format helps with cosine similarity
    y_and = [torch.tensor([[0, 1]]), torch.tensor([[0, 1]]), torch.tensor([[0, 1]]), torch.tensor([[1, 0]])]
    y_or = [torch.tensor([[0, 1]]), torch.tensor([[1, 0]]), torch.tensor([[1, 0]]), torch.tensor([[1, 0]])]
    if params.io == "random":
        num_data = 4 if "num_data" not in params else params.num_data
        input_size = params.input_size if "input_size" in params else input_size
        output_size = params.output_size if "output_size" in params else output_size
        xs = [torch.rand(size=(1, input_size)) for _ in range(num_data)]
        ys = [torch.rand(size=(1, output_size)) for _ in range(num_data)]
    elif params.io == "xor":
        ys = y_xor
    elif params.io == "and":
        ys = y_and
    elif params.io == "or":
        ys = y_or
    elif params.io == "ra25":
        num_data = 25
        choose_n = 6
        input_size = 25
        output_size = 25
        xs = [torch.tensor([random.sample([1] * choose_n + [0] * (input_size - choose_n), input_size)]) for _ in range(num_data)]
        ys = [torch.tensor([random.sample([1] * choose_n + [0] * (output_size - choose_n), output_size)]) for _ in range(num_data)]
    num_data = len(xs)

    return xs, ys, input_size, hidden_size, output_size, num_data
