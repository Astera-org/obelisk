import math

import torch
from hyperparams import HParams
import random
from torchvision import datasets as dts
from torchvision.transforms import ToTensor


def get_mnist(params: HParams):
    train_X=dts.MNIST(
        root='data',
        train=True,
        transform=ToTensor(),
        download=True,
    )
    test_dt=dts.MNIST(
        root='data',
        train=False,
        transform=ToTensor()
    )
    x = train_X.data[15]
    items = []
    labels = []
    size = int(math.sqrt(params.input_size))
    for _ in range(params.num_data):
        sample = random.randint(0, len(train_X)-1)
        number = train_X[sample][0]
        samp = [[0 if number.data[0, int(j*28/size), int(i*28/size)] < .2 else 1 for i in range(size)] for j in range(size)] # Downsample
        items.append(torch.tensor([[x for xs in samp for x in xs]]).float()) # Flatten and tensorize
        label = train_X[sample][1]
        labels.append(torch.Tensor([[1 if i == label else 0 for i in range(10)]]))
    return items, labels


def get_data(params: HParams):
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
    if params.dataset == "random":
        num_data = params.num_data
        input_size = params.input_size
        output_size = params.output_size
        xs = [torch.rand(size=(1, input_size)) for _ in range(num_data)]
        ys = [torch.rand(size=(1, output_size)) for _ in range(num_data)]
    elif params.dataset == "xor":
        ys = y_xor
    elif params.dataset == "and":
        ys = y_and
    elif params.dataset == "or":
        ys = y_or
    elif params.dataset == "mnist":
        input_size = params.input_size
        output_size = 10 # 10 digits
        xs, ys = get_mnist(params)
    elif params.dataset == "ra25":
        num_data = 25
        choose_n = 6
        input_size = 25
        output_size = 25
        xs = [torch.tensor([random.sample([1] * choose_n + [0] * (input_size - choose_n), input_size)]) for _ in range(num_data)]
        ys = [torch.tensor([random.sample([1] * choose_n + [0] * (output_size - choose_n), output_size)]) for _ in range(num_data)]
    num_data = len(xs)

    return torch.concat(xs), torch.concat(ys), input_size, hidden_size, output_size, num_data
