import unittest
from src.utils import load_probs


class MyTestCase(unittest.TestCase):
    def test_np_load(self):
        filename = '../../data/link_prediction/ggi_0.8_unweighted_probs.csv'
        probs = load_probs(filename)
        assert len(probs) == 12331


if __name__ == '__main__':
    unittest.main()
