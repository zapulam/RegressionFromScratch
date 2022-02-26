"""
Written by Zachary Pulliam

main.py simply calls run() to fit regression models for the WineDataset and two SyntheticDataset's

Within the run() function...
- tranform can be set to normalize, standardize, or None depending on the desired data transormation
- poly can be set to None or 2+ for the desired polynomial regression (x values after x1 in the regression equation will reflect x^n
- alpha is the learning rate
- epochs is the number of training epochs
- lam is the L2 regularization term to prevent overfitting
"""

import os

from regression import Regression
from datasets import WineDataset, SyntheticDataset


""" Variables for the user to change"""
ROOT = ''  # path to data folder


def run():
    # datasets
    wd = WineDataset(os.path.join(ROOT, 'winequality-red.csv'), transform=None)
    sd1 = SyntheticDataset(os.path.join(ROOT, 'synthetic-1.csv'), transform=None, poly=3)
    sd2 = SyntheticDataset(os.path.join(ROOT, 'synthetic-2.csv'), transform=None, poly=5)

    # regression for the WineDataset
    reg = Regression(wd.x, wd.y, alpha=0.00001, lam=None, epochs=10000)
    reg.mse(wd.x, wd.y)

    # regression for the SyntheticDataset #1
    reg = Regression(sd1.x, sd1.y, alpha=0.01, lam=None, epochs=10000)
    reg.mse(sd1.x, sd1.y)

    # regression for the SyntheticDataset #2
    reg = Regression(sd2.x, sd2.y, alpha=0.01, lam=None, epochs=10000)
    reg.mse(sd2.x, sd2.y)

if __name__ == '__main__':
    run()