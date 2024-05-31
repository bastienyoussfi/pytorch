import os
from pathlib import Path
import torch, numpy as np, pandas as pd

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

np.set_printoptions(linewidth=140)
torch.set_printoptions(linewidth=140, sci_mode=False, edgeitems=7)
pd.set_option('display.width', 140)

df = pd.read_csv(path/'train.csv')