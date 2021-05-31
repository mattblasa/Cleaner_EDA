import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt



#############Cleaning Methods#####################

def zero_to_NaN(df, column): #This is only a single column. Will add multiple columns in future update
    '''
    Parameters 
    df- takes in dataframe 
    column - 

    Returns 
    Returns correlation heatmap 

    '''
    if df.loc[df[column] == 0.0]:
        df.loc[df[column] == 0.0] = np.nan
    else:
        pass 
    return df 



########## Graphing Methods ######################
def no_kwargs_plot(x, y, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.plot(x, y) ## example plot here
    return(ax)plt.figure(figsize=(10, 5))


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

def norm_feat(series): #feature normalization
    return (series - series.mean())/series.std()



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