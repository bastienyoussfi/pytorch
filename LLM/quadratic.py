import numpy as np, matplotlib.pyplot as plt
from numpy.random import normal,seed,uniform

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Fix the randomization
np.random.seed(42)

# Parameters
min=-2
max=2

# Creates a function that we'll use to try and fit 
def f(x): 
    return -3*x**2 + 2*x + 20

# Adds a noise using a normal distribution
def noise(x, scale): 
    return normal(scale=scale, size=x.shape)

def add_noise(x, mult, add): 
    return x * (1+noise(x, mult)) + noise(x, add)

# Defines the variables
x = np.linspace(min, max, num=20)[:,None]
y = add_noise(f(x), 0.1, 1.3)

# Plots the desired function
def plot_function(f, min=-2, max=2, color='r'):
    plt.plot(x, f(x), color)
    
# Plots a fitting polynomial function of the desired degree
def plot_poly(degree):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(x, y)
    plt.scatter(x, y)
    plot_function(model.predict)

plot_poly(2)
plot_function(f, color='b')
plt.show()