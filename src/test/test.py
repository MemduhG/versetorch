import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data_utils.batch import rebatch
from src.data_utils.data import get_training_iterators, get_dataset


def test_iterators():
    """Make sure you can reproduce the original order of a shuffled dataset."""
    pad_idx = 3
    print(pad_idx)
    train, val, test, train_idx, val_idx, test_idx = get_training_iterators("antoloji")
    mt_train, mt_dev, mt_test = get_dataset("antoloji")
    for r in (rebatch(pad_idx, b) for b in val):
        print(r)
        input()


if __name__ == "__main__":
    test_iterators()