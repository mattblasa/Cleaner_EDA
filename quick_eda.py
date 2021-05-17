import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


def correlation_graph(df):
    '''
    Parameters 
    Takes in dataframe 

    Returns 
    Returns correlation heatmap 

    '''
    corr = ansur_df.corr()
    sns.heatmap(corr,  cmap=cmap, center=0, linewidths=1, annot=True, fmt=".2f")
    plt.show()

def PCA(args, kwargs, **kwargs):
    '''
    Parameters 

    Returns 
    PCA Chart
    '''

def PCA_dataframe(args, kwargs):
    '''
    '''

def multicol(args, kwargs):
    '''
    Parameters 

    Returns 
    This method returns a chart that checks multicolinearity when called. 
    '''