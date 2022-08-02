import torch
from runs_harness import run_many_times
from runs_harness import train_and_test
from hyperparams import HParams

if __name__ == '__main__':
    # torch.manual_seed(2) # For debugging purposes.
    torch.set_printoptions(precision=3, sci_mode=False)
    num_runs = 10
    epochs = 20

    # Only one run
    # run_many_times(HParams(epochs=epochs, hidden_size=100, num_rnn_steps=5, num_runs=3, dataset="mnist", input_size=100, verbose=3, score="perc_correct", num_data=1000, batch_data=False, learning_rate=0.01))

    # FWorld
    # run_many_times(HParams(epochs=epochs, hidden_size=300, num_rnn_steps=5, num_runs=3, dataset="fworld", input_size=100, verbose=3, score="perc_correct", num_data=1000, batch_data=False, learning_rate=0.001))

    # # Try Train and Test
    train_and_test(
        HParams(num_runs=1, epochs=2, hidden_size=100, num_rnn_steps=5, dataset="mnist", input_size=100, verbose=1,
                norm_weights=True, score="perc_correct", num_data=1000, batch_data=True, learning_rate=0.1,
                self_connection_strength=0),
        HParams(verbose=1, score="perc_correct", num_data=1000))

    # # Hypothesis: norm_weights=True improves performance only for high values of num_rnn_steps.
    # run_many_times(HParams(epochs=epochs, hidden_size=10, num_rnn_steps=5, num_runs=20, dataset="xor", verbose=1, norm_weights=False, score="convergence"))
    # run_many_times(HParams(epochs=epochs, hidden_size=10, num_rnn_steps=5, num_runs=20, dataset="xor", verbose=1, norm_weights=True, score="convergence"))
    # run_many_times(HParams(epochs=epochs, hidden_size=10, num_rnn_steps=50, num_runs=20, dataset="xor", verbose=1, norm_weights=False, score="convergence"))
    # run_many_times(HParams(epochs=epochs, hidden_size=10, num_rnn_steps=50, num_runs=20, dataset="xor", verbose=1, norm_weights=True, score="convergence"))

    # # Look at learning rate for large num_data
    # run_many_times(HParams(epochs=epochs, hidden_size=10, num_rnn_steps=5, num_runs=5, dataset="random", verbose=0, norm_weights=True, input_size=10, output_size=10, learning_rate=0.1, num_data=10))
    # run_many_times(HParams(epochs=epochs, hidden_size=10, num_rnn_steps=5, num_runs=5, dataset="random", verbose=0, norm_weights=False, input_size=10, output_size=10, learning_rate=0.01, num_data=10))

    # # Try to solve random
    # run_many_times(HParams(epochs=epochs, hidden_size=10, num_rnn_steps=5, num_runs=num_runs, dataset="random"))
    # run_many_times(HParams(epochs=epochs, hidden_size=10, num_rnn_steps=10, num_runs=num_runs, dataset="random"))
    # run_many_times(HParams(epochs=epochs, hidden_size=20, num_rnn_steps=10, num_runs=num_runs, dataset="random"))
    # run_many_times(HParams(epochs=epochs, hidden_size=20, num_rnn_steps=20, num_runs=num_runs, dataset="random"))
    # run_many_times(HParams(epochs=epochs, hidden_size=50, num_rnn_steps=20, num_runs=num_runs, dataset="random"))

    # Test XOR
    # run_many_times(HParams("epochs":20,epochs=epochs, hidden_size=4, num_rnn_steps=5, num_runs=1, dataset="random"))
    # run_many_times(HParams(epochs=50, hidden_size=5, num_rnn_steps=5, num_runs=1, dataset="xor"))
    # run_many_times(HParams(epochs=50, hidden_size=5, num_rnn_steps=5, num_runs=1, dataset="and"))
    # run_many_times(HParams(epochs=50, hidden_size=5, num_rnn_steps=5, num_runs=1, dataset="or"))
    # #run_many_times(HParams(epochs=50, hidden_size=5, num_rnn_steps=5, num_runs=1, dataset="random"))
    # # run_many_times(HParams(epochs=epochs, hidden_size=4, num_rnn_steps=5, num_runs=1, dataset="random"))

    # # Test XOR
    # run_many_times(HParams(epochs=epochs, hidden_size=5, num_rnn_steps=5, num_runs=num_runs, dataset="xor"))
    # run_many_times(HParams(epochs=epochs, hidden_size=5, num_rnn_steps=5, num_runs=num_runs, dataset="and"))
    # run_many_times(HParams(epochs=epochs, hidden_size=5, num_rnn_steps=5, num_runs=num_runs, dataset="or"))
    # run_many_times(HParams(epochs=epochs, hidden_size=5, num_rnn_steps=5, num_runs=num_runs, dataset="random"))

    # # Test normalization in XOR
    # # First XOR has both
    # run_many_times(HParams(epochs=epochs, hidden_size=5, num_rnn_steps=5, num_runs=num_runs, dataset="xor", norm_weights=False))
    # run_many_times(HParams(epochs=epochs, hidden_size=5, num_rnn_steps=5, num_runs=num_runs, dataset="xor", norm_act=False))
    # run_many_times(HParams(epochs=epochs, hidden_size=5, num_rnn_steps=5, num_runs=num_runs, dataset="xor", norm_weights=False, norm_act=False))

    # Test RNN averaging # This seems to not make a difference
    # run_many_times(HParams(epochs=epochs, hidden_size=5, num_rnn_steps=5, num_runs=num_runs, dataset="random"))
    # run_many_times(HParams(epochs=epochs, hidden_size=5, num_rnn_steps=5, num_runs=num_runs, dataset="random", average_window=3))

    # run_many_times({epochs=epochs, hidden_size=0, num_rnn_steps=1, num_runs=num_runs)
    # run_many_times({epochs=epochs, hidden_size=2, num_rnn_steps=1, num_runs=num_runs)
    # run_many_times({epochs=epochs, hidden_size=5, num_rnn_steps=1, num_runs=num_runs)
    # run_many_times({epochs=epochs, hidden_size=10, num_rnn_steps=1, num_runs=num_runs)
    # run_many_times({epochs=epochs, hidden_size=0, num_rnn_steps=5, num_runs=num_runs)
    # run_many_times({epochs=epochs, hidden_size=2, num_rnn_steps=5, num_runs=num_runs)
    # run_many_times({epochs=epochs, hidden_size=5, num_rnn_steps=5, num_runs=num_runs)
    # run_many_times({epochs=epochs, hidden_size=10, num_rnn_steps=5, num_runs=num_runs)
    # run_many_times({epochs=epochs, hidden_size=0, num_rnn_steps=10, num_runs=num_runs)
    # run_many_times({epochs=epochs, hidden_size=2, num_rnn_steps=10, num_runs=num_runs)
    # run_many_times({epochs=epochs, hidden_size=5, num_rnn_steps=10, num_runs=num_runs)
    # run_many_times({epochs=epochs, hidden_size=10, num_rnn_steps=10, num_runs=num_runs)
