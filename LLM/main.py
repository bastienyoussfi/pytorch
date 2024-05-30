import os
from pathlib import Path
import pandas as pd
from datasets import Dataset,DatasetDict
from transformers import AutoModelForSequenceClassification,AutoTokenizer

iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')

creds = '{"username":"bastienyoussfi","key":"ca27a6b0de4f0e70eafa6558ada1e15f"}'

cred_path = Path('~/.kaggle/kaggle.json').expanduser()
if not cred_path.exists():
    cred_path.parent.mkdir(exist_ok=True)
    cred_path.write_text(creds)
    cred_path.chmod(0o600)

path = Path('us-patent-phrase-to-phrase-matching')

# Dowload and unzip the data
if not iskaggle and not path.exists():
    import zipfile,kaggle
    kaggle.api.competition_download_cli(str(path))
    zipfile.ZipFile(f'{path}.zip').extractall(path)

df = pd.read_csv(path/'train.csv')
stat = df.describe(include='object')

df['input'] = 'TEXT1: ' + df.context + '; TEXT2: ' + df.target + '; ANC1: ' + df.anchor
head = df.input.head()

# Create the dataset
ds = Dataset.from_pandas(df)

# Import the model
model_nm = 'microsoft/deberta-v3-small'
tokz = AutoTokenizer.from_pretrained(model_nm)

def tok_func(x): return tokz(x["input"])

tok_ds = ds.map(tok_func, batched=True)
row = tok_ds[0]
tokens = row['input'], row['input_ids']
tok_ds = tok_ds.rename_columns({'score':'labels'})

eval_df = pd.read_csv(path/'test.csv')
stats = eval_df.describe()

print(stats)