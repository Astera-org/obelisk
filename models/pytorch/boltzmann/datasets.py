import math

import torch
import torchvision.transforms

from hyperparams import HParams
import random
from torchvision import datasets as dts
from torchvision.transforms import ToTensor
from torchvision import transforms
from fworld_loader import get_fworld_data

from torch.utils.data import random_split, DataLoader, Dataset


class DatasetDefault(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class MnistDataset(Dataset):
    def __init__(self, train=True, max_sample=-1, size=10):
        self.train = train  # either train or test, no validate
        self.max_sample = max_sample
        self.setup()  # this typically shouldn't go here for runtime reasons, see pytorc hlightning

    def setup(self):
        train_x = dts.MNIST(  # You could also use FashionMNIST, which is harder
            root='data',
            train=self.train,
            transform=None,
            download=True
        )
        apply_transforms = torch.nn.Sequential(transforms.Resize(size=10))
        self.data = apply_transforms(train_x.data).float()
        self.labels = train_x.targets.float()

    def __len__(self):
        if self.max_sample < 0:
            return len(self.data)
        else:
            return self.max_sample

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def default_dataloader(x, y, batch_size=10, shuffle=False) -> DataLoader:
    dataset = DatasetDefault(x, y)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return train_dataloader


def mnist_dataloader(size=10, max_train=-1, max_test=-1, shuffle=False, batch_size=10) -> (DataLoader, DataLoader):
    """
    returns data in terms of a dataloader
    """
    train_dataset = MnistDataset(True, max_train, size)
    test_dataset = MnistDataset(False, max_test, size)
    if batch_size == -1:
        train_batch_size = len(train_dataset)
        test_batch_size = len(test_dataset)
    else:
        train_batch_size = batch_size
        test_batch_size = batch_size
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=shuffle)
    return train_dataloader, test_dataloader


def get_mnist(params: HParams):
    train_x = dts.MNIST(  # You could also use FashionMNIST, which is harder
        root='data',
        train=True,
        transform=ToTensor(),
        download=True,
    )
    test_x = dts.MNIST(
        root='data',
        train=False,
        transform=ToTensor(),
        download=True,
    )
    mnist_digits = train_x
    if params.testing:
        mnist_digits = test_x
    items = []
    labels = []
    size = int(math.sqrt(params.input_size))
    for _ in range(params.num_data):
        sample = random.randint(0, len(mnist_digits) - 1)
        number = mnist_digits[sample][0]
        # TODO Torch has a one-liner for this
        # TODO Don't binarize
        samp = [[number.data[0, int(j * 28 / size), int(i * 28 / size)] for i in range(size)] for j in
                range(size)]  # Downsample
        items.append(torch.tensor([[x for xs in samp for x in xs]]).float())  # Flatten and tensorize
        label = mnist_digits[sample][1]
        labels.append(torch.Tensor([[1 if i == label else 0 for i in range(10)]]))
    return items, labels


def get_data(params: HParams):
    # Network
    input_size = 3
    output_size = 2
    hidden_size = params.hidden_size

    # XOR (input and output sizes different)
    xs = [torch.tensor([[0, 0, 1]]), torch.tensor([[0, 1, 1]]), torch.tensor([[1, 0, 1]]),
          torch.tensor([[1, 1, 1]])]  # Third input is bias
    ys = []
    y_xor = [torch.tensor([[0, 0]]), torch.tensor([[1, 1]]), torch.tensor([[1, 1]]),
             torch.tensor([[0, 0]])]  # This weird binumeral format helps with cosine similarity
    y_and = [torch.tensor([[0, 1]]), torch.tensor([[0, 1]]), torch.tensor([[0, 1]]), torch.tensor([[1, 0]])]
    y_or = [torch.tensor([[0, 1]]), torch.tensor([[1, 0]]), torch.tensor([[1, 0]]), torch.tensor([[1, 0]])]
    if params.dataset == "random":
        num_data = params.num_data
        input_size = params.input_size
        output_size = params.output_size
        xs = [torch.rand(size=(1, input_size)) for _ in range(num_data)]
        ys = [torch.rand(size=(1, output_size)) for _ in range(num_data)]
        xs, ys = torch.concat(xs), torch.concat(ys)
        if params.batch_size == -1:
            params.batch_size = len(xs)
        d1 = default_dataloader(xs, ys, batch_size=params.batch_size)
    elif params.dataset == "xor":
        ys = y_xor
        if params.batch_size == -1:
            params.batch_size = len(xs)
        d1 = default_dataloader(xs, ys, batch_size=params.batch_size)
    elif params.dataset == "and":
        ys = y_and
        if params.batch_size == -1:
            params.batch_size = len(xs)
        d1 = default_dataloader(xs, ys, batch_size=params.batch_size)
    elif params.dataset == "or":
        ys = y_or
        if params.batch_size == -1:
            params.batch_size = len(xs)
        d1 = default_dataloader(xs, ys, batch_size=params.batch_size)
    elif params.dataset == "mnist":
        input_size = params.input_size
        if params.batch_size == -1:
            params.batch_size = len(xs)
        output_size = 10  # 10 digits
        xs, ys = get_mnist(params)
        d1, t1 = mnist_dataloader(size=input_size, batch_size=params.batch_size)
    elif params.dataset == "ra25":
        num_data = 25
        choose_n = 6
        input_size = 25
        output_size = 25
        xs = [torch.tensor([random.sample([1] * choose_n + [0] * (input_size - choose_n), input_size)]) for _ in
              range(num_data)]
        ys = [torch.tensor([random.sample([1] * choose_n + [0] * (output_size - choose_n), output_size)]) for _ in
              range(num_data)]
        if params.batch_size == -1:
            params.batch_size = len(xs)
        d1 = default_dataloader(xs, ys, batch_size=params.batch_size)
    elif params.dataset == "fworld":
        xs, ys, input_size, output_size = get_fworld_data(params)
        if params.batch_size == -1:
            params.batch_size = len(xs)
        d1 = default_dataloader(xs, ys, batch_size=params.batch_size)
    num_data = len(xs)

    return torch.concat(xs), torch.concat(ys), input_size, hidden_size, output_size, num_data, d1
