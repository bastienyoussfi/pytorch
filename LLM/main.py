import os
from pathlib import Path
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from transformers import TrainingArguments,Trainer

# Training parameters
bs = 128
epochs = 4
lr = 8e-5

# Environment variable
iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')

# Compute the correlation coefficient between the two columns x and y
def corr(x, y):
    return np.corrcoef(x,y)[0][1]

# Compute the Pearson coefficient, used as the metrics for the training
def corr_d(eval_pred): 
    return {'pearson': corr(*eval_pred)}

# Credentials
creds = '{"username":"bastienyoussfi","key":"ca27a6b0de4f0e70eafa6558ada1e15f"}'

# Kaggle login
cred_path = Path('~/.kaggle/kaggle.json').expanduser()
if not cred_path.exists():
    cred_path.parent.mkdir(exist_ok=True)
    cred_path.write_text(creds)
    cred_path.chmod(0o600)

# Path to the competition
path = Path('us-patent-phrase-to-phrase-matching')

# Dowload and unzip the data
if not iskaggle and not path.exists():
    import zipfile,kaggle
    kaggle.api.competition_download_cli(str(path))
    zipfile.ZipFile(f'{path}.zip').extractall(path)

# Create the Dataframe object and its statistics
df = pd.read_csv(path/'train.csv')
stat = df.describe(include='object')

# Formats the Dataframe and creates a preview
df['input'] = 'TEXT1: ' + df.context + '; TEXT2: ' + df.target + '; ANC1: ' + df.anchor
head = df.input.head()

# Create the dataset object
ds = Dataset.from_pandas(df)

# Import the model
model_nm = 'microsoft/deberta-v3-small'
tokz = AutoTokenizer.from_pretrained(model_nm)

# Tokenize the dataset["input"]
def tok_func(x): 
    return tokz(x["input"])

# Map the dataset using tokenization, allowing batch makes the mapping faster
tok_ds = ds.map(tok_func, batched=True)
row = tok_ds[0]

# Filters only the tokenization and their ids, renames them
tokens = row['input'], row['input_ids']
tok_ds = tok_ds.rename_columns({'score':'labels'})

# Loads the test set and its statistics
eval_df = pd.read_csv(path/'test.csv')
stats = eval_df.describe()

# Splits the dataset into training, evaluation and test sets
dds = tok_ds.train_test_split(0.25, seed=42)

# Formats the Dataframe and creates a preview
eval_df['input'] = 'TEXT1: ' + eval_df.context + '; TEXT2: ' + eval_df.target + '; ANC1: ' + eval_df.anchor
eval_head = eval_df.head()

# Map the dataset using tokenization, allowing batch makes the mapping faster
eval_ds = Dataset.from_pandas(eval_df).map(tok_func, batched=True)

# Creates the TrainingArguments object necessary for the training
args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=False,
    evaluation_strategy="epoch", per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2,
    num_train_epochs=epochs, weight_decay=0.01, report_to='none')

# Instantiate the model and the trainer objects
model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)
trainer = Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'],
                  tokenizer=tokz, compute_metrics=corr_d)

# Trains the model
trainer.train()

# Predicts label values on the test set
preds = trainer.predict(eval_ds).predictions.astype(float)
preds = np.clip(preds, 0, 1)