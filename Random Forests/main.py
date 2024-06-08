from fastai.imports import *
import pandas as pd
import os
import seaborn as sns

np.set_printoptions(linewidth=130)

iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')

# Path to the competition, downloads the data
path = Path('titanic')

df = pd.read_csv(path/'train.csv')
tst_df = pd.read_csv(path/'test.csv')
modes = df.mode().iloc[0]

#-------------- Data Preprocessing --------------#

# Function to process the data
def proc_data(df):
    df['Fare'] = df.Fare.fillna(0)
    df.fillna(modes, inplace=True)
    df['LogFare'] = np.log1p(df['Fare'])
    df['Embarked'] = pd.Categorical(df.Embarked)
    df['Sex'] = pd.Categorical(df.Sex)

# Categorical and continuous variables
cats=["Sex","Embarked"]
conts=['Age', 'SibSp', 'Parch', 'LogFare',"Pclass"]
dep="Survived"

# Process the data
proc_data(df)

# Shows the differences between the original and processed data
head = df.Sex.head()
realHead = df.Sex.cat.codes.head()

palette=['red', 'blue', 'green', 'purple']
fig,axs = plt.subplots(1,2, figsize=(11,5))
sns.barplot(data=df, y=dep, x="Sex", ax=axs[0], palette=palette).set(title="Survival rate")
sns.countplot(data=df, x="Sex", ax=axs[1]).set(title="Histogram");
plt.savefig("plot.png")