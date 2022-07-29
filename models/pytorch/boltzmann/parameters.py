from dataclasses import dataclass


@dataclass
class Parameters:
    # What data set to use. See datasets.py for legal values.
    io: str = "xor"

    # What metric to report as the score.
    score: str = "perc_correct"

    # How long to run it.
    epochs: int = 100
    num_runs: int = 1

    # num_rnn_steps governs how much recurrence occurs in the settling phase of the Boltzmann machine.
    num_rnn_steps: int = 5

    # What sort of normalization should we do? By default, only normalize the hidden component of the activity vector, setting its L2 length to 1.
    norm_weights: bool = True
    norm_act: bool = False
    norm_hidden: bool = True

    # Network characteristics. Input and output size will be overwritten by some values of io.
    hidden_size: int = 10
    input_size: int = 10
    output_size: int = 10

    # The learning rate used by the network.
    learning_rate: float = 0.1

    # Number of data points. This will be overwritten by some values of io.
    num_data: int = 10

    # average_window lets you take a running average here, because full_act seems like it might alternate with period>1. It does not seem to make a difference.
    average_window: int = 0

    # verbose is purely a logging option. Higher numbers mean more logging.
    verbose: int = 0
