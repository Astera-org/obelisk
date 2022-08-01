import torch
from runs_harness import run_many_times
from hyperparams import HParams
from boltzmann_machine import BoltzmannMachine
import unittest

# Verify that the network learns some basic tasks.


class BoltzmannTest(unittest.TestCase):
    def test_xor(self):
        torch.manual_seed(0)
        print("\nTesting XOR")
        first_success, _ = run_many_times(HParams(epochs=100, hidden_size=10, num_rnn_steps=5, num_runs=1, dataset="xor", verbose=0, norm_weights=True, score="convergence", batch_data=False))
        self.assertLess(first_success, 50)

        score, _ = run_many_times(HParams(epochs=50, hidden_size=10, num_rnn_steps=5, num_runs=1, dataset="xor", verbose=0, norm_weights=True, score="perc_correct", batch_data=False))
        self.assertEqual(score, 1.0)

    def test_random(self):
        torch.manual_seed(0)
        print("\nTesting Random")
        score, _ = run_many_times(HParams(epochs=50, hidden_size=10, num_rnn_steps=5, num_runs=1, dataset="random", verbose=0, norm_weights=True, score="perc_correct", batch_data=False))
        self.assertGreaterEqual(score, 0.65) # Pretty lenient

    def test_random_batch(self):
        torch.manual_seed(0)
        print("\nTesting Random Batch")
        score, _ = run_many_times(HParams(epochs=100, hidden_size=10, num_rnn_steps=5, num_runs=1, dataset="random", verbose=0, norm_weights=True, score="perc_correct", batch_data=True, learning_rate=.1))
        self.assertGreaterEqual(score, 0.65) # Pretty lenient

    def test_ra25(self):
        print("\nTesting RA25")
        first_success, _ = run_many_times(HParams(epochs=100, hidden_size=10, num_rnn_steps=5, num_runs=1, dataset="xor", verbose=0, norm_weights=True, score="convergence"))
        self.assertLess(first_success, 40)

        score, _ = run_many_times(HParams(epochs=50, hidden_size=10, num_rnn_steps=5, num_runs=1, dataset="xor", verbose=0, norm_weights=True, score="perc_correct"))
        self.assertEqual(score, 1.0)

    def test_confidence(self):
        print("\nTesting Confidence Bounds")
        first_success, (confidence_low, confidence_high) = run_many_times(HParams(epochs=30, hidden_size=3, num_rnn_steps=5, num_runs=10, dataset="and", verbose=1, norm_weights=True, score="convergence"))
        self.assertLess(confidence_low, confidence_high)

    def test_xor_batch(self):
        torch.manual_seed(0)
        print("\nTesting XOR Batch")
        first_success, _ = run_many_times(HParams(epochs=100, hidden_size=10, num_rnn_steps=5, num_runs=1, dataset="xor", verbose=0, norm_weights=True, score="convergence", batch_data=True))
        self.assertLess(first_success, 60)

        score, _ = run_many_times(HParams(epochs=50, hidden_size=10, num_rnn_steps=5, num_runs=1, dataset="xor", verbose=0, norm_weights=True, score="perc_correct", batch_data=True))

    def test_weight_symmetry(self):

        params = HParams(weights_start_symmetric=True)
        b1 = BoltzmannMachine(2,2,2,params)
        assert ((b1.layer.weight.T == b1.layer.weight).sum()) == len(b1.layer.weight.flatten()), "weights should be symmetric"

        params = HParams(weights_start_symmetric=False)

        b2 = BoltzmannMachine(2,2,2,params)
        assert ((b2.layer.weight.T == b2.layer.weight).sum()) != len(b2.layer.weight.flatten()), "weights should not be symmetric"


    def activation_modulation(self):
        params:HParams = HParams(weights_start_symmetric=False,backward_connection_strength=1,
                                 forward_connection_strength=1, self_connection_strength = 1)


if __name__ == '__main__':
    torch.manual_seed(0) # To make tests consistent
    torch.set_printoptions(precision=3, sci_mode=False)

    t = unittest.TestLoader().loadTestsFromTestCase(BoltzmannTest)
    unittest.TextTestRunner(verbosity=3).run(t)
