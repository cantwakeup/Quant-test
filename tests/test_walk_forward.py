import unittest

from src.walk_forward import purged_walk_forward_splits


class WalkForwardTests(unittest.TestCase):
    def test_purge_preserves_time_order(self):
        splits = list(purged_walk_forward_splits(100, min_train_size=50, test_size=10, purge=5, embargo=2))
        train, test = splits[0]
        self.assertLess(train.max(), test.min())
        self.assertEqual(test.min() - train.max(), 6)


if __name__ == "__main__":
    unittest.main()
