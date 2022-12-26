```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from recsys_utils import *
```


```python
#Load data
X, W, b, num_movies, num_features, num_users = load_precalc_params_small()
Y, R = load_ratings_small()

print("Y", Y.shape, "R", R.shape)
print("X", X.shape)
print("W", W.shape)
print("b", b.shape)
print("num_features", num_features)
print("num_movies",   num_movies)
print("num_users",    num_users)
```

    Y (4778, 443) R (4778, 443)
    X (4778, 10)
    W (443, 10)
    b (1, 443)
    num_features 10
    num_movies 4778
    num_users 443
    


```python
#  From the matrix, we can compute statistics like average rating.
tsmean =  np.mean(Y[0, R[0, :].astype(bool)])
print(f"Average rating for movie 1 : {tsmean:0.3f} / 5" )
```

    Average rating for movie 1 : 3.400 / 5
    


```python
def cofi_cost_func(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering
    Args:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    nm, nu = Y.shape
    J = 0
    for j in range(nu):
        w = W[j,:]
        b_j = b[0,j]
        for i in range(nm):
            x = X[i,:]
            y = Y[i,j]
            r = R[i,j]
            J += np.square(r * (np.dot(w,x) + b_j - y ) ) 
    J = J/2             
    
    ### Regularization ### 
    J += (lambda_/2) * (np.sum(np.square(W)) + np.sum(np.square(X)))

    return J
```


```python
# Reduce the data set size so that this runs faster
num_users_r = 4
num_movies_r = 5 
num_features_r = 3

X_r = X[:num_movies_r, :num_features_r]
W_r = W[:num_users_r,  :num_features_r]
b_r = b[0, :num_users_r].reshape(1,-1)
Y_r = Y[:num_movies_r, :num_users_r]
R_r = R[:num_movies_r, :num_users_r]

# Evaluate cost function
J = cofi_cost_func(X_r, W_r, b_r, Y_r, R_r, 0);
print(f"Cost: {J:0.2f}")
```

    Cost: 13.67
    


```python
# Evaluate cost function with regularization 
J = cofi_cost_func(X_r, W_r, b_r, Y_r, R_r, 1.5);
print(f"Cost (with regularization): {J:0.2f}")
```

    Cost (with regularization): 28.09
    

**Vectorized Implementation**


```python
def cofi_cost_func_v(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering
    Vectorized for speed. Uses tensorflow operations to be compatible with custom training loop.
    Args:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y)*R
    J = 0.5 * tf.reduce_sum(j**2) + (lambda_/2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
    return J
```


```python
# Evaluate cost function
J = cofi_cost_func_v(X_r, W_r, b_r, Y_r, R_r, 0);
print(f"Cost: {J:0.2f}")

# Evaluate cost function with regularization 
J = cofi_cost_func_v(X_r, W_r, b_r, Y_r, R_r, 1.5);
print(f"Cost (with regularization): {J:0.2f}")
```

    Cost: 13.67
    Cost (with regularization): 28.09
    

## Learning movie recommendations


```python
movieList, movieList_df = load_Movie_List_pd()

my_ratings = np.zeros(num_movies)          #  Initialize my ratings

# Check the file small_movie_list.csv for id of each movie in our dataset
# For example, Toy Story 3 (2010) has ID 2700, so to rate it "5", you can set
my_ratings[2700] = 5 

#Or suppose you did not enjoy Persuasion (2007), you can set
my_ratings[2609] = 2;

# We have selected a few movies we liked / did not like and the ratings we
# gave are as follows:
my_ratings[929]  = 5   # Lord of the Rings: The Return of the King, The
my_ratings[246]  = 5   # Shrek (2001)
my_ratings[2716] = 3   # Inception
my_ratings[1150] = 5   # Incredibles, The (2004)
my_ratings[382]  = 2   # Amelie (Fabuleux destin d'Amélie Poulain, Le)
my_ratings[366]  = 5   # Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)
my_ratings[622]  = 5   # Harry Potter and the Chamber of Secrets (2002)
my_ratings[988]  = 3   # Eternal Sunshine of the Spotless Mind (2004)
my_ratings[2925] = 1   # Louis Theroux: Law & Disorder (2008)
my_ratings[2937] = 1   # Nothing to Declare (Rien à déclarer)
my_ratings[793]  = 5   # Pirates of the Caribbean: The Curse of the Black Pearl (2003)
my_rated = [i for i in range(len(my_ratings)) if my_ratings[i] > 0]

print('\nNew user ratings:\n')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0 :
        print(f'Rated {my_ratings[i]} for  {movieList_df.loc[i,"title"]}');
```

    
    New user ratings:
    
    Rated 5.0 for  Shrek (2001)
    Rated 5.0 for  Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)
    Rated 2.0 for  Amelie (Fabuleux destin d'Amélie Poulain, Le) (2001)
    Rated 5.0 for  Harry Potter and the Chamber of Secrets (2002)
    Rated 5.0 for  Pirates of the Caribbean: The Curse of the Black Pearl (2003)
    Rated 5.0 for  Lord of the Rings: The Return of the King, The (2003)
    Rated 3.0 for  Eternal Sunshine of the Spotless Mind (2004)
    Rated 5.0 for  Incredibles, The (2004)
    Rated 2.0 for  Persuasion (2007)
    Rated 5.0 for  Toy Story 3 (2010)
    Rated 3.0 for  Inception (2010)
    Rated 1.0 for  Louis Theroux: Law & Disorder (2008)
    Rated 1.0 for  Nothing to Declare (Rien à déclarer) (2010)
    

Now, let's add these reviews to $Y$ and $R$ and normalize the ratings.


```python
# Reload ratings and add new ratings
Y, R = load_ratings_small()
Y    = np.c_[my_ratings, Y]
R    = np.c_[(my_ratings != 0).astype(int), R]

# Normalize the Dataset
Ynorm, Ymean = normalizeRatings(Y, R)
```

Let's prepare to train the model. Initialize the parameters and select the Adam optimizer.


```python
#  Useful Values
num_movies, num_users = Y.shape
num_features = 100

# Set Initial Parameters (W, X), use tf.Variable to track these variables
tf.random.set_seed(1234) # for consistent results
W = tf.Variable(tf.random.normal((num_users,  num_features),dtype=tf.float64),  name='W')
X = tf.Variable(tf.random.normal((num_movies, num_features),dtype=tf.float64),  name='X')
b = tf.Variable(tf.random.normal((1,          num_users),   dtype=tf.float64),  name='b')

# Instantiate an optimizer.
optimizer = keras.optimizers.Adam(learning_rate=1e-1)
```

Let's now train the collaborative filtering model. This will learn the parameters $\mathbf{X}$, $\mathbf{W}$, and $\mathbf{b}$. 


```python
iterations = 200
lambda_ = 1
for iter in range(iterations):
    # Use TensorFlow’s GradientTape
    # to record the operations used to compute the cost 
    with tf.GradientTape() as tape:

        # Compute the cost (forward pass included in cost)
        cost_value = cofi_cost_func_v(X, W, b, Ynorm, R, lambda_)

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss
    grads = tape.gradient( cost_value, [X,W,b] )

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients( zip(grads, [X,W,b]) )

    # Log periodically.
    if iter % 20 == 0:
        print(f"Training loss at iteration {iter}: {cost_value:0.1f}")
```

    Training loss at iteration 0: 2321191.3
    Training loss at iteration 20: 136168.7
    Training loss at iteration 40: 51863.3
    Training loss at iteration 60: 24598.8
    Training loss at iteration 80: 13630.4
    Training loss at iteration 100: 8487.6
    Training loss at iteration 120: 5807.7
    Training loss at iteration 140: 4311.6
    Training loss at iteration 160: 3435.2
    Training loss at iteration 180: 2902.1
    

## Recommendations


```python
# Make a prediction using trained weights and biases
p = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()

#restore the mean
pm = p + Ymean

my_predictions = pm[:,0]

# sort predictions
ix = tf.argsort(my_predictions, direction='DESCENDING')

for i in range(17):
    j = ix[i]
    if j not in my_rated:
        print(f'Predicting rating {my_predictions[j]:0.2f} for movie {movieList[j]}')

print('\n\nOriginal vs Predicted ratings:\n')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print(f'Original {my_ratings[i]}, Predicted {my_predictions[i]:0.2f} for {movieList[i]}')
```

    Predicting rating 4.49 for movie My Sassy Girl (Yeopgijeogin geunyeo) (2001)
    Predicting rating 4.48 for movie Martin Lawrence Live: Runteldat (2002)
    Predicting rating 4.48 for movie Memento (2000)
    Predicting rating 4.47 for movie Delirium (2014)
    Predicting rating 4.47 for movie Laggies (2014)
    Predicting rating 4.47 for movie One I Love, The (2014)
    Predicting rating 4.46 for movie Particle Fever (2013)
    Predicting rating 4.45 for movie Eichmann (2007)
    Predicting rating 4.45 for movie Battle Royale 2: Requiem (Batoru rowaiaru II: Chinkonka) (2003)
    Predicting rating 4.45 for movie Into the Abyss (2011)
    
    
    Original vs Predicted ratings:
    
    Original 5.0, Predicted 4.90 for Shrek (2001)
    Original 5.0, Predicted 4.84 for Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)
    Original 2.0, Predicted 2.13 for Amelie (Fabuleux destin d'Amélie Poulain, Le) (2001)
    Original 5.0, Predicted 4.88 for Harry Potter and the Chamber of Secrets (2002)
    Original 5.0, Predicted 4.87 for Pirates of the Caribbean: The Curse of the Black Pearl (2003)
    Original 5.0, Predicted 4.89 for Lord of the Rings: The Return of the King, The (2003)
    Original 3.0, Predicted 3.00 for Eternal Sunshine of the Spotless Mind (2004)
    Original 5.0, Predicted 4.90 for Incredibles, The (2004)
    Original 2.0, Predicted 2.11 for Persuasion (2007)
    Original 5.0, Predicted 4.80 for Toy Story 3 (2010)
    Original 3.0, Predicted 3.00 for Inception (2010)
    Original 1.0, Predicted 1.41 for Louis Theroux: Law & Disorder (2008)
    Original 1.0, Predicted 1.26 for Nothing to Declare (Rien à déclarer) (2010)
    


```python
filter=(movieList_df["number of ratings"] > 20)
movieList_df["pred"] = my_predictions
movieList_df = movieList_df.reindex(columns=["pred", "mean rating", "number of ratings", "title"])
movieList_df.loc[ix[:300]].loc[filter].sort_values("mean rating", ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pred</th>
      <th>mean rating</th>
      <th>number of ratings</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1743</th>
      <td>4.030965</td>
      <td>4.252336</td>
      <td>107</td>
      <td>Departed, The (2006)</td>
    </tr>
    <tr>
      <th>2112</th>
      <td>3.985287</td>
      <td>4.238255</td>
      <td>149</td>
      <td>Dark Knight, The (2008)</td>
    </tr>
    <tr>
      <th>211</th>
      <td>4.477792</td>
      <td>4.122642</td>
      <td>159</td>
      <td>Memento (2000)</td>
    </tr>
    <tr>
      <th>929</th>
      <td>4.887053</td>
      <td>4.118919</td>
      <td>185</td>
      <td>Lord of the Rings: The Return of the King, The...</td>
    </tr>
    <tr>
      <th>2700</th>
      <td>4.796530</td>
      <td>4.109091</td>
      <td>55</td>
      <td>Toy Story 3 (2010)</td>
    </tr>
    <tr>
      <th>653</th>
      <td>4.357304</td>
      <td>4.021277</td>
      <td>188</td>
      <td>Lord of the Rings: The Two Towers, The (2002)</td>
    </tr>
    <tr>
      <th>1122</th>
      <td>4.004469</td>
      <td>4.006494</td>
      <td>77</td>
      <td>Shaun of the Dead (2004)</td>
    </tr>
    <tr>
      <th>1841</th>
      <td>3.980647</td>
      <td>4.000000</td>
      <td>61</td>
      <td>Hot Fuzz (2007)</td>
    </tr>
    <tr>
      <th>3083</th>
      <td>4.084633</td>
      <td>3.993421</td>
      <td>76</td>
      <td>Dark Knight Rises, The (2012)</td>
    </tr>
    <tr>
      <th>2804</th>
      <td>4.434171</td>
      <td>3.989362</td>
      <td>47</td>
      <td>Harry Potter and the Deathly Hallows: Part 1 (...</td>
    </tr>
    <tr>
      <th>773</th>
      <td>4.289679</td>
      <td>3.960993</td>
      <td>141</td>
      <td>Finding Nemo (2003)</td>
    </tr>
    <tr>
      <th>1771</th>
      <td>4.344993</td>
      <td>3.944444</td>
      <td>81</td>
      <td>Casino Royale (2006)</td>
    </tr>
    <tr>
      <th>2649</th>
      <td>4.133482</td>
      <td>3.943396</td>
      <td>53</td>
      <td>How to Train Your Dragon (2010)</td>
    </tr>
    <tr>
      <th>2455</th>
      <td>4.175746</td>
      <td>3.887931</td>
      <td>58</td>
      <td>Harry Potter and the Half-Blood Prince (2009)</td>
    </tr>
    <tr>
      <th>361</th>
      <td>4.135291</td>
      <td>3.871212</td>
      <td>132</td>
      <td>Monsters, Inc. (2001)</td>
    </tr>
    <tr>
      <th>3014</th>
      <td>3.967901</td>
      <td>3.869565</td>
      <td>69</td>
      <td>Avengers, The (2012)</td>
    </tr>
    <tr>
      <th>246</th>
      <td>4.897137</td>
      <td>3.867647</td>
      <td>170</td>
      <td>Shrek (2001)</td>
    </tr>
    <tr>
      <th>151</th>
      <td>3.971888</td>
      <td>3.836364</td>
      <td>110</td>
      <td>Crouching Tiger, Hidden Dragon (Wo hu cang lon...</td>
    </tr>
    <tr>
      <th>1150</th>
      <td>4.898892</td>
      <td>3.836000</td>
      <td>125</td>
      <td>Incredibles, The (2004)</td>
    </tr>
    <tr>
      <th>793</th>
      <td>4.874935</td>
      <td>3.778523</td>
      <td>149</td>
      <td>Pirates of the Caribbean: The Curse of the Bla...</td>
    </tr>
    <tr>
      <th>366</th>
      <td>4.843375</td>
      <td>3.761682</td>
      <td>107</td>
      <td>Harry Potter and the Sorcerer's Stone (a.k.a. ...</td>
    </tr>
    <tr>
      <th>754</th>
      <td>4.021774</td>
      <td>3.723684</td>
      <td>76</td>
      <td>X2: X-Men United (2003)</td>
    </tr>
    <tr>
      <th>79</th>
      <td>4.242984</td>
      <td>3.699248</td>
      <td>133</td>
      <td>X-Men (2000)</td>
    </tr>
    <tr>
      <th>622</th>
      <td>4.878342</td>
      <td>3.598039</td>
      <td>102</td>
      <td>Harry Potter and the Chamber of Secrets (2002)</td>
    </tr>
  </tbody>
</table>
</div>


