"""
Written by Zachary Pulliam

datasets.py contains 2 dataset classes for the two example dataset types which are tested on...
WineDataset and SyntheticDataset, other dataset classes can be added here as well so that the functiosn can be used

    - the functions standardize and normalize can be used on pandas df's
    - the conv_to_array function should be used on pandas df's which have outputs in the last column
"""

import numpy as np
import pandas as pd


"""Normalizes a dataframe"""
def normalize(df):
    df = (df-df.min())/(df.max()-df.min())
    return df


"""Standardizes a dataframe"""
def standardize(df):
    for col in df:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df


"""Converts a dataframe to input and output arrays to be used by the Regresion class"""
def conv_to_array(df):
    data = np.array(df, dtype=float)
    x = data[:,:-1]
    x = np.hstack((np.ones((x.shape[0],1)), x))
    y = data[:, -1].reshape(1,x.shape[0]).T
    return x, y



"""Synthetic Datasets class for a single synthetic.csv file"""
class WineDataset:
    def __init__(self, path, transform):
        self.path = path
        self.transform = transform
        self.name = 'Wine Dataset'
        self.df = None  # original dataframe
        self.scaled_df = None  # dataframe after transformation
        self.x = None
        self.y = None
        self.load_data()

    def load_data(self):
        self.df = pd.read_csv(self.path, names=[
            'fixed_acididty',
            'volatile_acidity',
            'citric_acid',
            'residual_sugar',
            'chlorides',
            'free_SO2',
            'total_SO2',
            'density',
            'pH',
            'sulphates',
            'alcohol',
            'quality'], header=0)

        # transform, if desired
        if self.transform == 'normalize':
            self.scaled_df = normalize(self.df)
        elif self.transform == 'standardize':
            self.scaled_df = standardize(self.df)
        elif self.transform == None:
            self.scaled_df = self.df
        else:
            raise ValueError('Use either n, s, or None for normalize, standarize, or None')

        # x and y, input and output arrays
        self.x, self.y = conv_to_array(self.scaled_df)



"""Synthetic Datasets class for a single synthetic.csv file"""
class SyntheticDataset:
    def __init__(self, path, transform, poly):
        self.path = path
        self.transform = transform
        self.poly = poly
        self.name = 'Wine Dataset'
        self.df = None  # original dataframe
        self.scaled_df = None  # dataframe after transformation
        self.x = None
        self.y = None
        self.load_data()

    def load_data(self):
        self.df = pd.read_csv(self.path, names=['x','y'])

        if self.poly != None:
            for i in range(self.poly-1):
                self.df.insert(i+1, "x^{x}".format(x=i+2), self.df.iloc[:,0]**(i+2))

        # transform, if desired
        if self.transform == 'normalize':
            self.scaled_df = normalize(self.df)
        elif self.transform == 'standardize':
            self.scaled_df = standardize(self.df)
        elif self.transform == None:
            self.scaled_df = self.df
        else:
            raise ValueError('Use either n, s, or None for normalize, standarize, or None')

        # x and y, input and output arrays
        self.x, self.y = conv_to_array(self.scaled_df)