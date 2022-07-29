import torch
from runs_harness import run_many_times
from parameters import Parameters
import unittest

# Verify that the network learns some basic tasks.


class BoltzmannTest(unittest.TestCase):
    def test_xor(self):
        torch.manual_seed(0)
        print("\nTesting XOR")
        first_success, _ = run_many_times(Parameters(epochs=100, hidden_size=10, num_rnn_steps=5, num_runs=1, io="xor", verbose=0, norm_weights=True, score="convergence", batch_data=False))
        self.assertLess(first_success, 50)

        score, _ = run_many_times(Parameters(epochs=50, hidden_size=10, num_rnn_steps=5, num_runs=1, io="xor", verbose=0, norm_weights=True, score="perc_correct", batch_data=False))
        self.assertEqual(score, 1.0)

    def test_random(self):
        torch.manual_seed(0)
        print("\nTesting Random")
        score, _ = run_many_times(Parameters(epochs=50, hidden_size=10, num_rnn_steps=5, num_runs=1, io="random", verbose=0, norm_weights=True, score="perc_correct", batch_data=False))
        self.assertGreaterEqual(score, 0.65) # Pretty lenient

    def test_random_batch(self):
        torch.manual_seed(0)
        print("\nTesting Random Batch")
        score, _ = run_many_times(Parameters(epochs=100, hidden_size=10, num_rnn_steps=5, num_runs=1, io="random", verbose=0, norm_weights=True, score="perc_correct", batch_data=True, learning_rate=.1))
        self.assertGreaterEqual(score, 0.65) # Pretty lenient

    def test_ra25(self):
        print("\nTesting RA25")
        first_success, _ = run_many_times(Parameters(epochs=100, hidden_size=10, num_rnn_steps=5, num_runs=1, io="xor", verbose=0, norm_weights=True, score="convergence"))
        self.assertLess(first_success, 40)

        score, _ = run_many_times(Parameters(epochs=50, hidden_size=10, num_rnn_steps=5, num_runs=1, io="xor", verbose=0, norm_weights=True, score="perc_correct"))
        self.assertEqual(score, 1.0)

    def test_confidence(self):
        print("\nTesting Confidence Bounds")
        first_success, (confidence_low, confidence_high) = run_many_times(Parameters(epochs=30, hidden_size=3, num_rnn_steps=5, num_runs=10, io="and", verbose=1, norm_weights=True, score="convergence"))
        self.assertLess(confidence_low, confidence_high)

    def test_xor_batch(self):
        torch.manual_seed(0)
        print("\nTesting XOR Batch")
        first_success, _ = run_many_times(Parameters(epochs=100, hidden_size=10, num_rnn_steps=5, num_runs=1, io="xor", verbose=0, norm_weights=True, score="convergence", batch_data=True))
        self.assertLess(first_success, 60)

        score, _ = run_many_times(Parameters(epochs=50, hidden_size=10, num_rnn_steps=5, num_runs=1, io="xor", verbose=0, norm_weights=True, score="perc_correct", batch_data=True))
        self.assertEqual(score, 1.0)


if __name__ == '__main__':
    torch.manual_seed(0) # To make tests consistent
    torch.set_printoptions(precision=3, sci_mode=False)

    t = unittest.TestLoader().loadTestsFromTestCase(BoltzmannTest)
    unittest.TextTestRunner(verbosity=3).run(t)
