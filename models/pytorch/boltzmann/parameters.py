from dataclasses import dataclass


@dataclass
class Parameters:
    # What data set to use. See datasets.py for legal values.
    dataset: str = "xor"

    # What metric to report as the score.
    score: str = "perc_correct"

    # How long to run it.
    epochs: int = 100
    num_runs: int = 1

    # num_rnn_steps governs how much recurrence occurs in the settling phase of the Boltzmann machine.
    num_rnn_steps: int = 5

    # The learning rate used by the network.
    learning_rate: float = 0.1

    # What sort of normalization should we do? By default, only normalize the hidden component of the activity vector, setting its L2 length to 1.
    norm_weights: bool = True
    norm_act: bool = False
    norm_hidden: bool = True

    # Network characteristics. Input and output size will be overwritten by some values of dataset.
    hidden_size: int = 10
    input_size: int = 10
    output_size: int = 10

    # Number of data points. This will be overwritten by some values of dataset.
    num_data: int = 10

    # If the network gets 100% correct performance this many times in a row, it will halt.
    stopping_success: int = 5

    # average_window lets you take a running average over the last n steps of the rnn, rather than just use the final activity value, because full_act seems like it might alternate with period>1. If 0, no averaging occurs, and the final value is used. It does not seem to make a difference.
    average_window: int = 0

    # verbose is purely a logging option. Higher numbers mean more logging.
    verbose: int = 0

    # Put all the data from a single epoch into a single batch. In the future, for larger datasets, we will need to use dataloaders to accommodate this.
    batch_data: bool = False
