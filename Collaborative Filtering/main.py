from fastai.collab import *
from fastai.tabular.all import *
import pandas as pd

set_seed(42)

path = untar_data(URLs.ML_100k)

# Retrieves the ratings of the movies by every user
ratings = pd.read_csv(path/'u.data', delimiter='\t', header=None,
                      names=['user','movie','rating','timestamp'])
head = ratings.head()

#------------- Learning the latent factors -------------

# Retrieves the movies and their titles
movies = pd.read_csv(path/'u.item',  delimiter='|', encoding='latin-1',
                     usecols=(0,1), names=('movie','title'), header=None)
# head = movies.head()

# Merge the ratings and movies dataframes
ratings = ratings.merge(movies)
# head = ratings.head()

# Create the DataLoaders object
dls = CollabDataLoaders.from_df(ratings, item_name='title', bs=64)
# dls.show_batch()

# Defining the model variables, with 5 latent factors
n_users  = len(dls.classes['user'])
n_movies = len(dls.classes['title'])
n_factors = 5

# Generates a tensor of shape (n_users, n_factors) filled with random numbers drawn from a standard normal distribution (mean 0, variance 1) 
user_factors = torch.randn(n_users, n_factors)
movie_factors = torch.randn(n_movies, n_factors)

#------------- Collaborative filtering from scratch -------------

# Defines a class for our dot product model
class DotProduct(Module):
    def __init__(self, n_users, n_movies, n_factors, y_range=(0,5.5)): # Adding a y_range parameter to the model to make it more precise
        self.user_factors = Embedding(n_users, n_factors)
        self.user_bias = Embedding(n_users, 1)
        self.movie_factors = Embedding(n_movies, n_factors)
        self.movie_bias = Embedding(n_movies, 1)
        self.y_range = y_range
        
    def forward(self, x):
        users = self.user_factors(x[:,0]) # x[:,0] contains the user ids
        movies = self.movie_factors(x[:,1]) # x[:,1] contains the movie ids
        res = (users * movies).sum(dim=1, keepdim=True) # Multiplies the user and movie factors and sums them up
        res += self.user_bias(x[:,0]) + self.movie_bias(x[:,1]) # Adds the user and movie biases
        return sigmoid_range(res, *self.y_range) # Returns the dot product of the user and movie factors
    
# Creates the model and the learner
model = DotProduct(n_users, n_movies, 50)
learn = Learner(dls, model, loss_func=MSELossFlat())

# Trains the model
learn.fit_one_cycle(5, 5e-3, wd=0.1)