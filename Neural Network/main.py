import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Change the backend to 'Agg' or another compatible backend
import pandas as pd
import matplotlib.pyplot as plt
import torch, numpy as np

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

df['Fare'].hist()

print()