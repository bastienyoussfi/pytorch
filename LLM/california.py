from sklearn.datasets import fetch_california_housing
import numpy as np
from matplotlib import pyplot as plt

# Loads the Dataframe object using as_frame=True
housing = fetch_california_housing(as_frame=True)

# Creates the Dataframe and generates a random sample of size 100, random_state ensure the reproducibility of it
housing = housing['data'].join(housing['target']).sample(1000, random_state=52)
head = housing.head()

# Fix parameters to printing values
np.set_printoptions(precision=2, suppress=True)

# Computes examples or correlation coefficients
corrcoef = np.corrcoef(housing, rowvar=False)
corrcoef = np.corrcoef(housing.MedInc, housing.AveOccup)

# Computes the correlation coefficients
def corr(x, y):
    return np.corrcoef(x, y)[0][1]

# Computes correlation coefficient between the a and b columns of the Dataframe
def show_corr(df, a, b):
    x,y = df[a],df[b]
    plt.scatter(x,y, alpha=0.5, s=4)
    plt.title(f'{a} vs {b}; r: {corr(x, y):.2f}')
    plt.show()

# Filters the Dataframe to remove the outliers and fix the correlation coefficient
subset = housing[housing.AveRooms<15]
corrcoef = show_corr(subset, 'HouseAge', 'AveRooms')

# Computes Pearson coefficient
def corr_d(eval_pred): 
    return {'pearson': corr(*eval_pred)}