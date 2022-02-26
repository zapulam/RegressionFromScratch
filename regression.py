"""
Written by Zachary Pulliam

regression.py contains the Regression class which performs linear regression
    - x is a matrix of observations or inputs
    - y is a matrix of outputs
    - alpha is the learning rate
    - epochs is the number of epochs
    - (data can be correctly manipulated to fit the input format using the functions in datasets.py)
"""

import numpy as np

class Regression:
    def __init__(self, x, y, alpha, lam, epochs):
        self.alpha = alpha  # learning rate
        self.epochs = epochs  # number of training epochs
        self.lam = lam  # L2 regularization hyperparameter
        self.theta = np.random.rand(x.shape[1], 1)  # nx1 array of theta values
        self.fit(x, y)  # fits regression model according to x and y
        self.print_h()  # prints regression equation


    """Performs Gradient Descent to create a regression model"""
    def fit(self, x, y):
        m = x.shape[0]  # number of training examples

        for i in range(self.epochs):
            h = np.matmul(x, self.theta)  # m x 1 matrix containing predictions for each observation based on theta

            if i > self.epochs - self.epochs*0.9:
                    self.alpha = self.alpha * 0.99999  # adjusts learning rate towards the end of the regression
            
            if self.lam == None:
                self.theta = self.theta - self.alpha*(1/m)*(x.T@(h - y))  # updates all theta values according to loss function
            else:
                self.theta = self.theta - self.alpha*((1/m)*(x.T@(h - y)) + self.lam*self.theta)  # updates all theta values according to loss function


    """Prints the regression equation"""
    def print_h(self):
        print('Regression equation:   y = ', end='')
        for i, val in enumerate(self.theta):
            if i == 0:
                print(round(val[0], 3), end=' ')
            else:
                print('+ ', end='')
                print('{y}*x{z}'.format(y=round(val[0], 3), z=i), end=' ')
        print('')


    """Computes Mean Squared Error for a test set"""
    def mse(self, x, y):
        h = np.matmul(x, self.theta)  # compute hypothesis for all inputs
        mse = ((h-y).T@(h-y))/(2*y.shape[0])  # compute MSE
        
        print('MSE = {z}'.format(z=round(mse.item(), 3)), end='\n')
        print('')