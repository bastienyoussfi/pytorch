import numpy as np, matplotlib.pyplot as plt


def f(x): return -3*x**2 + 2*x + 20


def plot_function(f, min=-2.1, max=2.1, color='r'):
    x = np.linspace(min,max, 100)[:,None]
    plt.plot(x, f(x), color)
    plt.show()

plot_function(f)