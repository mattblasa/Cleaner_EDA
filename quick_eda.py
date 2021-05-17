import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


def correlation_graph(df):
    corr = ansur_df.corr()
    sns.heatmap(corr,  cmap=cmap, center=0, linewidths=1, annot=True, fmt=".2f")
    plt.show()