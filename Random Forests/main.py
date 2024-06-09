from fastai.imports import *
import pandas as pd
import os
import seaborn as sns
from numpy import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
from sklearn.ensemble import RandomForestClassifier

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

# Defines the palette and the axis
palette=['red', 'blue', 'green', 'purple']
fig,axs = plt.subplots(1,2, figsize=(11,5))

# Barplot and countplot of the survival rate depending on the sex
sns.barplot(data=df, y=dep, x="Sex", ax=axs[0], hue="Sex").set(title="Survival rate")
sns.countplot(data=df, x="Sex", ax=axs[1], hue="Sex").set(title="Histogram");

# Save the plot
# plt.savefig("plot.png")

#-------------- Binary Split --------------#

# Split the data into training and validation
random.seed(42)
trn_df,val_df = train_test_split(df, test_size=0.25)

# Replace the categorical variables with their codes to better manipulate them
trn_df[cats] = trn_df[cats].apply(lambda x: x.cat.codes)
val_df[cats] = val_df[cats].apply(lambda x: x.cat.codes)

# Creates the xs and y, which are the dependent and independent variables
def xs_y(df):
    xs = df[cats+conts].copy()
    return xs,df[dep] if dep in df else None

# Assigns the xs and y to the training and validation data
trn_xs,trn_y = xs_y(trn_df)
val_xs,val_y = xs_y(val_df)

# Extremely simple model, where all the women survive and all the men die
preds = val_xs.Sex==0

# Gets the mean absolute error to test our simple model
mae = mean_absolute_error(val_y, preds)

# Some more graphs
df_fare = trn_df[trn_df.LogFare>0]
fig,axs = plt.subplots(1,2, figsize=(11,5))
sns.boxenplot(data=df_fare, x=dep, y="LogFare", hue="Survived", ax=axs[0])
sns.kdeplot(data=df_fare, x="LogFare", ax=axs[1])
# plt.savefig("plot2.png")

# Considering the observations, people with a higher fare are more likely to survive
preds = val_xs.LogFare>2.7

# Gets the mean absolute error to test our simple model
mae = mean_absolute_error(val_y, preds)

# Calculates the score for one side of the split
def _side_score(side, y):
    tot = side.sum()
    if tot<=1: 
        return 0
    return y[side].std()*tot

# Calculates the score for the split, called impurity
def score(col, y, split):
    lhs = col<=split
    return (_side_score(lhs,y) + _side_score(~lhs,y))/len(y)

# score = score(trn_xs["Sex"], trn_y, 0.5)
# score = score(trn_xs["LogFare"], trn_y, 2.7)

# Let's try to find the best split for any variable

# Gets the unique values of the variable age
nm = "Age"
col = trn_xs[nm]
unq = col.unique()
unq.sort()

# Calculates the best split for the variable age
scores = np.array([score(col, trn_y, o) for o in unq if not np.isnan(o)])
bestsplit = unq[scores.argmin()]

# Calculates the best split for any column
def min_col(df, nm):
    col = df[nm]
    y = df[dep]
    unq = col.dropna().unique()
    unq.sort()
    scores = np.array([score(col, y, o) for o in unq if not np.isnan(o)])
    idx = scores.argmin()
    return unq[idx], scores[idx]

# Calculates the best split for all the columns
# cols = cats+conts
# cols_scores = { nm:min_col(trn_df, nm) for nm in cols}

#-------------- Decision Tree --------------#

# Calculates the best split for all the columns
# cols = cats+conts
# cols_scores = { nm:min_col(trn_df, nm) for nm in cols }

# # Considering that Sex is the best first split, we can remove it from the list and split the data in two
# cols.remove('Sex')
# ismale = trn_df.Sex == 1
# males, females = trn_df[ismale], trn_df[~ismale]

# # We calculate the best split for both males and females and so on, we create a decision tree
# cols_scores_males = { nm:min_col(males, nm) for nm in cols }
# cols_scores_females = { nm:min_col(females, nm) for nm in cols }

# Let's automate the process of creating the decision tree
m = DecisionTreeClassifier(max_leaf_nodes=4).fit(trn_xs, trn_y)

# Draw the decision tree
def draw_tree(t, df, size=10, ratio=0.6, precision=4, **kwargs):
    s=export_graphviz(t, out_file="bigger_tree.dot", feature_names=df.columns, filled=True, rounded=True,
                      special_characters=True, rotate=False, precision=precision, **kwargs)
    return graphviz.Source(re.sub('Tree {', f'Tree {{ size={size}; ratio={ratio}', s))

# draw_tree(m, trn_xs, size=10)

# Another measure of impurity, used to create the above decision tree
def gini(cond):
    act = df.loc[cond, dep]
    return 1 - act.mean()**2 - (1-act).mean()**2

# Calculates the mean absolute error for the decision tree
mae = mean_absolute_error(val_y, m.predict(val_xs))

# Let's try to create a bigger tree
m = DecisionTreeClassifier(min_samples_leaf=50)
m.fit(trn_xs, trn_y)
# draw_tree(m, trn_xs, size=12)

# Calculates the mean absolute error for the decision tree
mae = mean_absolute_error(val_y, m.predict(val_xs))

#-------------- Random Forest --------------#

# Creates a random tree from our data
def get_tree(prop=0.75):
    n = len(trn_y)
    idxs = random.choice(n, int(n*prop)) # Randomly selects a subset of the data of size prop
    return DecisionTreeClassifier(min_samples_leaf=5).fit(trn_xs.iloc[idxs], trn_y.iloc[idxs])

# Creates a random forest from our data
trees = [get_tree() for t in range(100)]

all_probs = [t.predict(val_xs) for t in trees]
avg_probs = np.stack(all_probs).mean(0)

mae = mean_absolute_error(val_y, avg_probs)

# Now with sklearn
rf = RandomForestClassifier(100, min_samples_leaf=5)
rf.fit(trn_xs, trn_y);
mae = mean_absolute_error(val_y, rf.predict(val_xs))

# Let's see the importance of each variable
pd.DataFrame(dict(cols=trn_xs.columns, imp=m.feature_importances_)).plot('cols', 'imp', 'barh')
# plt.savefig("plot3.png")