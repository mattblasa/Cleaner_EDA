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


def stepwise_selection(X, y,
                       initial_list=[],
                       threshold_in=0.01,
                       threshold_out=0.05,
                       verbose=True):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed = False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(
                pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print(
                    'Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()  # null if pvalues is empty
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print(
                    'Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included