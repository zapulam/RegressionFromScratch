# Python - Regression From Scratch - Written by Zachary Pulliam

This code was written in order to implement a regression model that will be used to predict output 
for two 2D synthetic datasets and an 11D dataset of red wine quality.

The Regression class can be used on any other dataset as long as the output label 
is placed in the last column. This can be done in four steps...

1. Simply create a dataset class similar to those in datasets.py, by loading it into a pandas df
2. Decide if you wish to normalize or standardize the data using the built in functions in datasets.py
3. Convert the df to x and y arrays using the built in function conv_to_array()
4. Create an instance of the Regression class, passing in x, y, alpha, lambda, and epochs

In order to visualize the regression equation created for the 2D synthetic datasets, use the ipython notebook visualize.ipynb 
with the same variables.

The synthetic dataset contains two features, x and y.
The wine dataset contains values for acidity, sulphates, alcohol, etc.
