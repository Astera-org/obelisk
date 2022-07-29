import torch
from runs_harness import run_many_times
from parameters import Parameters


# Verify that the network learns some basic tasks.


if __name__ == '__main__':
    # torch.manual_seed(2) # For debugging purposes.
    torch.set_printoptions(precision=3, sci_mode=False)
    num_runs = 10
    epochs = 100

    # Only one run
    run_many_times(Parameters(epochs=epochs, hidden_size=2, num_rnn_steps=5, num_runs=1, io="xor", verbose=0, norm_weights=True, learning_rate=0.1))