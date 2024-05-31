from ipywidgets import interact
from fastai.basics import *
from IPython.display import display
from ipywidgets import interactive
import ipywidgets as widgets
import torch.nn.functional as F

np.random.seed(42)

plt.rc('figure', dpi=90)

def plot_function(f, title=None, min=-2.1, max=2.1, color='r', ylim=None):
    x = torch.linspace(min,max, 100)[:,None]
    if ylim: plt.ylim(ylim)
    plt.plot(x, f(x), color)
    if title is not None: plt.title(title)

def f(x): 
    return 3*x**2 + 2*x + 1

def quad(a, b, c, x): 
    return a*x**2 + b*x + c

def mk_quad(a,b,c): 
    return partial(quad, a,b,c)

def noise(x, scale): 
    return np.random.normal(scale=scale, size=x.shape)

def add_noise(x, mult, add): 
    return x * (1+noise(x,mult)) + noise(x,add)

x = torch.linspace(-2, 2, steps=20)[:,None]
y = add_noise(f(x), 0.15, 1.5)

# @interact(a=1.1, b=1.1, c=1.1)
def plot_quad(a, b, c):
    f = mk_quad(a,b,c)
    plt.scatter(x,y)
    loss = mae(f(x), y)
    plot_function(f, ylim=(-3,12), title=f"MAE: {loss:.2f}")

# Mean absolute error
def mae(preds, acts): 
    return (torch.abs(preds-acts)).mean()

def quad_mae(params):
    f = mk_quad(*params)
    return mae(f(x), y)

# Computes the mean absolute error on our parameters
quad_mae([1.1, 1.1, 1.1])

# Creates a tensor with the parameters
abc = torch.tensor([1.1,1.1,1.1])

# Specifies that we want to do a gradient descent, in order to minimize the loss
abc.requires_grad_()

# Computes the loss
loss = quad_mae(abc)

# Computes the gradient, stored in abc.grad
loss.backward()
abc.grad

# Computes the loss after modifying our parameters according to the gradient, the 0.01 is called "learning rate"
# torch.no_grad() disables the calculation of gradients for any operations inside that context manager
with torch.no_grad():
    abc -= abc.grad*0.01
    loss = quad_mae(abc)

# More of the above
for i in range(10):
    loss = quad_mae(abc)
    loss.backward()
    with torch.no_grad(): 
        abc -= abc.grad*0.01
    print(f'step={i}; loss={loss:.2f}')

def rectified_linear(m,b,x):
    y = m*x+b
    return torch.clip(y, 0.)

def rectified_linear2(m,b,x): 
    return F.relu(m*x+b)
plot_function(partial(rectified_linear2, 1,1))

# @interact(m=1.5, b=1.5)
def plot_relu(m, b):
    plot_function(partial(rectified_linear, m,b), ylim=(-1,4))

def double_relu(m1,b1,m2,b2,x):
    return rectified_linear(m1,b1,x) + rectified_linear(m2,b2,x)

# @interact(m1=-1.5, b1=-1.5, m2=1.5, b2=1.5)
def plot_double_relu(m1, b1, m2, b2):
    plot_function(partial(double_relu, m1,b1,m2,b2), ylim=(-1,6))