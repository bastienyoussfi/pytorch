import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Change the backend to 'Agg' or another compatible backend
import pandas as pd
import matplotlib.pyplot as plt
import torch, numpy as np
from torch import tensor
from fastai.data.transforms import RandomSplitter
import sympy as sp
import math
import torch.nn.functional as F

# Environment variable
iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')

# Path to the competition, downloads the data
if iskaggle: 
    path = Path('../input/titanic')
else:
    path = Path('titanic')
    if not path.exists():
        import zipfile,kaggle
        kaggle.api.competition_download_cli(str(path))
        zipfile.ZipFile(f'{path}.zip').extractall(path)

# ----------------- Data Preprocessing -----------------

# Display options
np.set_printoptions(linewidth=140)
torch.set_printoptions(linewidth=140, sci_mode=False, edgeitems=7)
pd.set_option('display.width', 140)

# Load the data
df = pd.read_csv(path/'train.csv')

# Gets the amount of NaN values in the dataset
naNum = df.isna().sum()

# mode calculates the most frequent value in the dataset for each column, iloc[0] gets the first row
modes = df.mode().iloc[0]

# Fills the NaN values with the most frequent value
df.fillna(modes, inplace=True)

# Generates summary only for the numerical columns
describe = df.describe(include=(np.number))

# Plots the histogram of the 'Fare' column
# df['Fare'].hist()
# plt.savefig('fare_histogram.png')

# Plots the histogram of the 'Fare' column with a logarithmic scale (added 1 to avoid log(0))
df['LogFare'] = np.log(df['Fare']+1)
# df['LogFare'].hist()
# plt.savefig('log_fare_histogram.png')

# Sorts the pclass values
pclasses = sorted(df.Pclass.unique())

# Generates summary only for the object columns, usually used for strings
df.describe(include=[object])

# Add dummy columns for non numerical columns
df_dummies = df[["Sex","Pclass","Embarked"]]
dummies = pd.get_dummies(df_dummies, columns=["Sex","Pclass","Embarked"]).astype(int)

# Shows the added columns
added_cols = ['Sex_male', 'Sex_female', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
for col in added_cols:
    df[col] = dummies[col]

added_cols_head = df[added_cols].head()

# Creating our dependent variable and converting it to a tensor
t_dep = tensor(df.Survived)

# Creating our independent variables and converting it to a tensor
indep_cols = ['Age', 'SibSp', 'Parch', 'LogFare'] + added_cols
numeric_cols = df[indep_cols].select_dtypes(include=[np.number]).columns.tolist()
t_indep = torch.tensor(df[numeric_cols].values, dtype=torch.float)

#----------------- Setting up a linear model -----------------

# Setting up the model with a random seed
torch.manual_seed(442)

# Gets the random coefficients, coefficients are the weights of the model and are between -0.5 and 0.5
n_coeff = t_indep.shape[1]
coeffs = torch.rand(n_coeff)-0.5

# Divides the independent variables by the maximum value of each column so that no column has more impact than another
vals,indices = t_indep.max(dim=0)
t_indep = t_indep / vals

# Calculates the predictions for one step of the model, sums each element on the same row (column sum), return a tensor of 891 elements
# def calc_preds(coeffs, indeps): 
#     return (indeps*coeffs).sum(axis=1)

# Uses the sigmoid function to calculate the predictions
# def calc_preds(coeffs, indeps): 
#     return torch.sigmoid((indeps*coeffs).sum(axis=1))

# Uses the matrix product to calculate the predictions, goes even faster
def calc_preds(coeffs, indeps): 
    return torch.sigmoid(indeps@coeffs)

# Defines the loss function as the mean absolute error, ie the difference between the predictions and if the person survived or not
def calc_loss(coeffs, indeps, deps): 
    return torch.abs(calc_preds(coeffs, indeps)-deps).mean()

#----------------- Gradient descent step -----------------

# Indicated pytorch that the coefficients require gradients
coeffs.requires_grad_()

# Calculates the loss
loss = calc_loss(coeffs, t_indep, t_dep)

# Calculates the gradients
loss.backward()

with torch.no_grad():
    # Updates the coefficients with the gradients
    coeffs.sub_(coeffs.grad * 0.1) # 0.1 is the learning rate
    coeffs.grad.zero_() # Resets the gradients
    # print(calc_loss(coeffs, t_indep, t_dep))

#----------------- Training the model -----------------

# Split the data into training and validation sets
trn_split,val_split=RandomSplitter(seed=42)(df)
trn_indep,val_indep = t_indep[trn_split],t_indep[val_split]
trn_dep,val_dep = t_dep[trn_split],t_dep[val_split]


# Converts the dependent variables to a column tensor
trn_dep = trn_dep[:,None]
val_dep = val_dep[:,None]

# Updates the coefficients with the gradients and resets the gradients
def update_coeffs(coeffs, lr):
    coeffs.sub_(coeffs.grad * lr)
    coeffs.grad.zero_()

# Trains the model for one epoch
def one_epoch(coeffs, lr):
    loss = calc_loss(coeffs, trn_indep, trn_dep)
    loss.backward()
    with torch.no_grad(): 
        update_coeffs(coeffs, lr)
    #print(f"{loss:.3f}", end="; ")

# Initializes the coefficients, the ( ,1) indicates that we wat a column tensor
def init_coeffs(): 
    return (torch.rand(n_coeff, 1)-0.5).requires_grad_()

# Trains the model for a number of epochs
def train_model(epochs=30, lr=0.01):
    torch.manual_seed(442)
    coeffs = init_coeffs()
    for i in range(epochs): 
        one_epoch(coeffs, lr)
    return coeffs

# Shows the coefficients
def show_coeffs(): 
    return dict(zip(indep_cols, coeffs.requires_grad_(False)))

coeffs = train_model(18, lr=0.2)

#----------------- Measuring Accuracy -----------------

# Calculates the predictions for the validation set
preds = calc_preds(coeffs, val_indep)

# We assume that any passenger with a score greater than 0.5 survived, and we calculate our accuracy thanks to this assumption
results = val_dep.bool()==(preds>0.5)
acc = results.float().mean()

# Calculates the accuracy of the model
def acc(coeffs): 
    return (val_dep.bool()==(calc_preds(coeffs, val_indep)>0.5)).float().mean()

#----------------- Using Sigmoid -----------------

coeffs = train_model(lr=100)
# print(show_coeffs())

#----------------- Neural Network ----------------

# Initializes the coefficients, depending on the number of hidden layers
def init_coeffs(n_hidden=20):
    layer1 = (torch.rand(n_coeff, n_hidden)-0.5)/n_hidden # We define the first layer as a matrix
    layer2 = torch.rand(n_hidden, 1)-0.3 # We define the second layer as a column tensor
    const = torch.rand(1)[0]
    return layer1.requires_grad_(),layer2.requires_grad_(),const.requires_grad_()

# Calculates the predictions for the neural network
def calc_preds(coeffs, indeps):
    l1,l2,const = coeffs # Learnings rates for the first and second layers
    res = F.relu(indeps@l1) # Applies the ReLU function to the first layer (Rectified Linear Unit)
    res = res@l2 + const # Applies the second layer and adds the constant
    return torch.sigmoid(res)

# Updates the coefficients with the gradients and resets the gradients
def update_coeffs(coeffs, lr):
    for layer in coeffs:
        layer.sub_(layer.grad * lr)
        layer.grad.zero_()

coeffs = train_model(lr=1.4)
coeffs = train_model(lr=20)

# print(acc(coeffs))

#----------------- Deep learning ----------------

def init_coeffs():
    hiddens = [10, 10]
    sizes = [n_coeff] + hiddens + [1]
    n = len(sizes)
    layers = [(torch.rand(sizes[i], sizes[i+1])-0.3)/sizes[i+1]*4 for i in range(n-1)]
    consts = [(torch.rand(1)[0]-0.5)*0.1 for i in range(n-1)]
    for l in layers+consts: l.requires_grad_()
    return layers,consts

def calc_preds(coeffs, indeps):
    layers,consts = coeffs
    n = len(layers)
    res = indeps
    for i,l in enumerate(layers):
        res = res@l + consts[i]
        if i!=n-1: res = F.relu(res)
    return torch.sigmoid(res)

def update_coeffs(coeffs, lr):
    layers,consts = coeffs
    for layer in layers+consts:
        layer.sub_(layer.grad * lr)
        layer.grad.zero_()

coeffs = train_model(lr=4)

print(acc(coeffs))