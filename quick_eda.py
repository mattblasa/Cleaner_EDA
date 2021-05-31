import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt





########## Graphing Methods ######################
def correlation_graph(df):
    '''
    Parameters 
    Takes in dataframe 

    Returns 
    Returns correlation heatmap 

    '''
    corr = df.corr()
    sns.heatmap(corr,  cmap=cmap, center=0, linewidths=1, annot=True, fmt=".2f")
    plt.show()

def pairplot(df, features):
    '''
    Parameters 
    df - original dataframe being used to create pairplot
    features - features from the df. These must be in an array 

    Returns 
    pair plot 
    '''
    try:
        df2 = df[features].copy()
        sns.pairplot(df2)
        plt.show()
    
    except:
        print('Check parameters if correct. Features must be in an array format for function to run.')

############Exploratory Data Analysis###########################################

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