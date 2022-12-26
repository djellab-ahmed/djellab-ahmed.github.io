# Deep Learning for Content-Based Filtering


```python
import numpy as np
import numpy.ma as ma
from numpy import genfromtxt
from collections import defaultdict
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tabulate
from recsysNN_utils import *
pd.set_option("display.precision", 1)
```


```python
# Load Data, set configuration variables
item_train, user_train, y_train, item_features, user_features, item_vecs, movie_dict, user_to_genre = load_data()

num_user_features = user_train.shape[1] - 3  # remove userid, rating count and ave rating during training
num_item_features = item_train.shape[1] - 1  # remove movie id at train time
uvs = 3  # user genre vector start
ivs = 3  # item genre vector start
u_s = 3  # start of columns to use in training, user
i_s = 1  # start of columns to use in training, items
scaledata = True  # applies the standard scalar to data if true
print(f"Number of training vectors: {len(item_train)}")
```

    Number of training vectors: 58187
    


```python
pprint_train(user_train, user_features, uvs,  u_s, maxcount=5)
```




<table>
<thead>
<tr><th style="text-align: center;"> [user id] </th><th style="text-align: center;"> [rating count] </th><th style="text-align: center;"> [rating ave] </th><th style="text-align: center;"> Act ion </th><th style="text-align: center;"> Adve nture </th><th style="text-align: center;"> Anim ation </th><th style="text-align: center;"> Chil dren </th><th style="text-align: center;"> Com edy </th><th style="text-align: center;"> Crime </th><th style="text-align: center;"> Docum entary </th><th style="text-align: center;"> Drama </th><th style="text-align: center;"> Fan tasy </th><th style="text-align: center;"> Hor ror </th><th style="text-align: center;"> Mys tery </th><th style="text-align: center;"> Rom ance </th><th style="text-align: center;"> Sci -Fi </th><th style="text-align: center;"> Thri ller </th></tr>
</thead>
<tbody>
<tr><td style="text-align: center;">     2     </td><td style="text-align: center;">       16       </td><td style="text-align: center;">     4.1      </td><td style="text-align: center;">   3.9   </td><td style="text-align: center;">    5.0     </td><td style="text-align: center;">    0.0     </td><td style="text-align: center;">    0.0    </td><td style="text-align: center;">   4.0   </td><td style="text-align: center;">  4.2  </td><td style="text-align: center;">     4.0      </td><td style="text-align: center;">  4.0  </td><td style="text-align: center;">   0.0    </td><td style="text-align: center;">   3.0   </td><td style="text-align: center;">   4.0    </td><td style="text-align: center;">   0.0    </td><td style="text-align: center;">   4.2   </td><td style="text-align: center;">    3.9    </td></tr>
<tr><td style="text-align: center;">     2     </td><td style="text-align: center;">       16       </td><td style="text-align: center;">     4.1      </td><td style="text-align: center;">   3.9   </td><td style="text-align: center;">    5.0     </td><td style="text-align: center;">    0.0     </td><td style="text-align: center;">    0.0    </td><td style="text-align: center;">   4.0   </td><td style="text-align: center;">  4.2  </td><td style="text-align: center;">     4.0      </td><td style="text-align: center;">  4.0  </td><td style="text-align: center;">   0.0    </td><td style="text-align: center;">   3.0   </td><td style="text-align: center;">   4.0    </td><td style="text-align: center;">   0.0    </td><td style="text-align: center;">   4.2   </td><td style="text-align: center;">    3.9    </td></tr>
<tr><td style="text-align: center;">     2     </td><td style="text-align: center;">       16       </td><td style="text-align: center;">     4.1      </td><td style="text-align: center;">   3.9   </td><td style="text-align: center;">    5.0     </td><td style="text-align: center;">    0.0     </td><td style="text-align: center;">    0.0    </td><td style="text-align: center;">   4.0   </td><td style="text-align: center;">  4.2  </td><td style="text-align: center;">     4.0      </td><td style="text-align: center;">  4.0  </td><td style="text-align: center;">   0.0    </td><td style="text-align: center;">   3.0   </td><td style="text-align: center;">   4.0    </td><td style="text-align: center;">   0.0    </td><td style="text-align: center;">   4.2   </td><td style="text-align: center;">    3.9    </td></tr>
<tr><td style="text-align: center;">     2     </td><td style="text-align: center;">       16       </td><td style="text-align: center;">     4.1      </td><td style="text-align: center;">   3.9   </td><td style="text-align: center;">    5.0     </td><td style="text-align: center;">    0.0     </td><td style="text-align: center;">    0.0    </td><td style="text-align: center;">   4.0   </td><td style="text-align: center;">  4.2  </td><td style="text-align: center;">     4.0      </td><td style="text-align: center;">  4.0  </td><td style="text-align: center;">   0.0    </td><td style="text-align: center;">   3.0   </td><td style="text-align: center;">   4.0    </td><td style="text-align: center;">   0.0    </td><td style="text-align: center;">   4.2   </td><td style="text-align: center;">    3.9    </td></tr>
<tr><td style="text-align: center;">     2     </td><td style="text-align: center;">       16       </td><td style="text-align: center;">     4.1      </td><td style="text-align: center;">   3.9   </td><td style="text-align: center;">    5.0     </td><td style="text-align: center;">    0.0     </td><td style="text-align: center;">    0.0    </td><td style="text-align: center;">   4.0   </td><td style="text-align: center;">  4.2  </td><td style="text-align: center;">     4.0      </td><td style="text-align: center;">  4.0  </td><td style="text-align: center;">   0.0    </td><td style="text-align: center;">   3.0   </td><td style="text-align: center;">   4.0    </td><td style="text-align: center;">   0.0    </td><td style="text-align: center;">   4.2   </td><td style="text-align: center;">    3.9    </td></tr>
</tbody>
</table>




```python
pprint_train(item_train, item_features, ivs, i_s, maxcount=5, user=False)
```




<table>
<thead>
<tr><th style="text-align: center;"> [movie id] </th><th style="text-align: center;"> year </th><th style="text-align: center;"> ave rating </th><th style="text-align: center;"> Act ion </th><th style="text-align: center;"> Adve nture </th><th style="text-align: center;"> Anim ation </th><th style="text-align: center;"> Chil dren </th><th style="text-align: center;"> Com edy </th><th style="text-align: center;"> Crime </th><th style="text-align: center;"> Docum entary </th><th style="text-align: center;"> Drama </th><th style="text-align: center;"> Fan tasy </th><th style="text-align: center;"> Hor ror </th><th style="text-align: center;"> Mys tery </th><th style="text-align: center;"> Rom ance </th><th style="text-align: center;"> Sci -Fi </th><th style="text-align: center;"> Thri ller </th></tr>
</thead>
<tbody>
<tr><td style="text-align: center;">    6874    </td><td style="text-align: center;"> 2003 </td><td style="text-align: center;">    4.0     </td><td style="text-align: center;">    1    </td><td style="text-align: center;">     0      </td><td style="text-align: center;">     0      </td><td style="text-align: center;">     0     </td><td style="text-align: center;">    0    </td><td style="text-align: center;">   0   </td><td style="text-align: center;">      0       </td><td style="text-align: center;">   0   </td><td style="text-align: center;">    0     </td><td style="text-align: center;">    0    </td><td style="text-align: center;">    0     </td><td style="text-align: center;">    0     </td><td style="text-align: center;">    0    </td><td style="text-align: center;">     0     </td></tr>
<tr><td style="text-align: center;">    6874    </td><td style="text-align: center;"> 2003 </td><td style="text-align: center;">    4.0     </td><td style="text-align: center;">    0    </td><td style="text-align: center;">     0      </td><td style="text-align: center;">     0      </td><td style="text-align: center;">     0     </td><td style="text-align: center;">    0    </td><td style="text-align: center;">   1   </td><td style="text-align: center;">      0       </td><td style="text-align: center;">   0   </td><td style="text-align: center;">    0     </td><td style="text-align: center;">    0    </td><td style="text-align: center;">    0     </td><td style="text-align: center;">    0     </td><td style="text-align: center;">    0    </td><td style="text-align: center;">     0     </td></tr>
<tr><td style="text-align: center;">    6874    </td><td style="text-align: center;"> 2003 </td><td style="text-align: center;">    4.0     </td><td style="text-align: center;">    0    </td><td style="text-align: center;">     0      </td><td style="text-align: center;">     0      </td><td style="text-align: center;">     0     </td><td style="text-align: center;">    0    </td><td style="text-align: center;">   0   </td><td style="text-align: center;">      0       </td><td style="text-align: center;">   0   </td><td style="text-align: center;">    0     </td><td style="text-align: center;">    0    </td><td style="text-align: center;">    0     </td><td style="text-align: center;">    0     </td><td style="text-align: center;">    0    </td><td style="text-align: center;">     1     </td></tr>
<tr><td style="text-align: center;">    8798    </td><td style="text-align: center;"> 2004 </td><td style="text-align: center;">    3.8     </td><td style="text-align: center;">    1    </td><td style="text-align: center;">     0      </td><td style="text-align: center;">     0      </td><td style="text-align: center;">     0     </td><td style="text-align: center;">    0    </td><td style="text-align: center;">   0   </td><td style="text-align: center;">      0       </td><td style="text-align: center;">   0   </td><td style="text-align: center;">    0     </td><td style="text-align: center;">    0    </td><td style="text-align: center;">    0     </td><td style="text-align: center;">    0     </td><td style="text-align: center;">    0    </td><td style="text-align: center;">     0     </td></tr>
<tr><td style="text-align: center;">    8798    </td><td style="text-align: center;"> 2004 </td><td style="text-align: center;">    3.8     </td><td style="text-align: center;">    0    </td><td style="text-align: center;">     0      </td><td style="text-align: center;">     0      </td><td style="text-align: center;">     0     </td><td style="text-align: center;">    0    </td><td style="text-align: center;">   1   </td><td style="text-align: center;">      0       </td><td style="text-align: center;">   0   </td><td style="text-align: center;">    0     </td><td style="text-align: center;">    0    </td><td style="text-align: center;">    0     </td><td style="text-align: center;">    0     </td><td style="text-align: center;">    0    </td><td style="text-align: center;">     0     </td></tr>
</tbody>
</table>




```python
print(f"y_train[:5]: {y_train[:5]}")
```

    y_train[:5]: [4.  4.  4.  3.5 3.5]
    


```python
# scale training data
if scaledata:
    item_train_save = item_train
    user_train_save = user_train

    scalerItem = StandardScaler()
    scalerItem.fit(item_train)
    item_train = scalerItem.transform(item_train)

    scalerUser = StandardScaler()
    scalerUser.fit(user_train)
    user_train = scalerUser.transform(user_train)

    print(np.allclose(item_train_save, scalerItem.inverse_transform(item_train)))
    print(np.allclose(user_train_save, scalerUser.inverse_transform(user_train)))
```

    True
    True
    


```python
item_train, item_test = train_test_split(item_train, train_size=0.80, shuffle=True, random_state=1)
user_train, user_test = train_test_split(user_train, train_size=0.80, shuffle=True, random_state=1)
y_train, y_test       = train_test_split(y_train,    train_size=0.80, shuffle=True, random_state=1)
print(f"movie/item training data shape: {item_train.shape}")
print(f"movie/item test  data shape: {item_test.shape}")
```

    movie/item training data shape: (46549, 17)
    movie/item test  data shape: (11638, 17)
    

The scaled, shuffled data now has a mean of zero.


```python
pprint_train(user_train, user_features, uvs, u_s, maxcount=5)
```




<table>
<thead>
<tr><th style="text-align: center;"> [user id] </th><th style="text-align: center;"> [rating count] </th><th style="text-align: center;"> [rating ave] </th><th style="text-align: center;"> Act ion </th><th style="text-align: center;"> Adve nture </th><th style="text-align: center;"> Anim ation </th><th style="text-align: center;"> Chil dren </th><th style="text-align: center;"> Com edy </th><th style="text-align: center;"> Crime </th><th style="text-align: center;"> Docum entary </th><th style="text-align: center;"> Drama </th><th style="text-align: center;"> Fan tasy </th><th style="text-align: center;"> Hor ror </th><th style="text-align: center;"> Mys tery </th><th style="text-align: center;"> Rom ance </th><th style="text-align: center;"> Sci -Fi </th><th style="text-align: center;"> Thri ller </th></tr>
</thead>
<tbody>
<tr><td style="text-align: center;">     1     </td><td style="text-align: center;">       0        </td><td style="text-align: center;">     0.6      </td><td style="text-align: center;">   0.7   </td><td style="text-align: center;">    0.6     </td><td style="text-align: center;">    0.6     </td><td style="text-align: center;">    0.7    </td><td style="text-align: center;">   0.7   </td><td style="text-align: center;">  0.5  </td><td style="text-align: center;">     0.7      </td><td style="text-align: center;">  0.2  </td><td style="text-align: center;">   0.3    </td><td style="text-align: center;">   0.3   </td><td style="text-align: center;">   0.5    </td><td style="text-align: center;">   0.5    </td><td style="text-align: center;">   0.8   </td><td style="text-align: center;">    0.5    </td></tr>
<tr><td style="text-align: center;">     0     </td><td style="text-align: center;">       0        </td><td style="text-align: center;">     1.6      </td><td style="text-align: center;">   1.5   </td><td style="text-align: center;">    1.7     </td><td style="text-align: center;">    0.9     </td><td style="text-align: center;">    1.0    </td><td style="text-align: center;">   1.4   </td><td style="text-align: center;">  0.8  </td><td style="text-align: center;">     -1.2     </td><td style="text-align: center;">  1.2  </td><td style="text-align: center;">   1.2    </td><td style="text-align: center;">   1.6   </td><td style="text-align: center;">   0.9    </td><td style="text-align: center;">   1.4    </td><td style="text-align: center;">   1.2   </td><td style="text-align: center;">    1.0    </td></tr>
<tr><td style="text-align: center;">     0     </td><td style="text-align: center;">       0        </td><td style="text-align: center;">     0.8      </td><td style="text-align: center;">   0.6   </td><td style="text-align: center;">    0.7     </td><td style="text-align: center;">    0.5     </td><td style="text-align: center;">    0.6    </td><td style="text-align: center;">   0.6   </td><td style="text-align: center;">  0.3  </td><td style="text-align: center;">     -1.2     </td><td style="text-align: center;">  0.7  </td><td style="text-align: center;">   0.8    </td><td style="text-align: center;">   0.9   </td><td style="text-align: center;">   0.6    </td><td style="text-align: center;">   0.2    </td><td style="text-align: center;">   0.6   </td><td style="text-align: center;">    0.6    </td></tr>
<tr><td style="text-align: center;">     1     </td><td style="text-align: center;">       0        </td><td style="text-align: center;">     -0.1     </td><td style="text-align: center;">   0.2   </td><td style="text-align: center;">    -0.1    </td><td style="text-align: center;">    0.3     </td><td style="text-align: center;">    0.7    </td><td style="text-align: center;">   0.3   </td><td style="text-align: center;">  0.2  </td><td style="text-align: center;">     1.0      </td><td style="text-align: center;"> -0.5  </td><td style="text-align: center;">   -0.7   </td><td style="text-align: center;">  -2.1   </td><td style="text-align: center;">   0.5    </td><td style="text-align: center;">   0.7    </td><td style="text-align: center;">   0.3   </td><td style="text-align: center;">    0.0    </td></tr>
<tr><td style="text-align: center;">    -1     </td><td style="text-align: center;">       0        </td><td style="text-align: center;">     -1.3     </td><td style="text-align: center;">  -0.8   </td><td style="text-align: center;">    -0.8    </td><td style="text-align: center;">    0.1     </td><td style="text-align: center;">   -0.1    </td><td style="text-align: center;">  -1.1   </td><td style="text-align: center;"> -0.9  </td><td style="text-align: center;">     -1.2     </td><td style="text-align: center;"> -1.5  </td><td style="text-align: center;">   -0.6   </td><td style="text-align: center;">  -0.5   </td><td style="text-align: center;">   -0.6   </td><td style="text-align: center;">   -0.9   </td><td style="text-align: center;">  -0.4   </td><td style="text-align: center;">   -0.9    </td></tr>
</tbody>
</table>




```python
scaler = MinMaxScaler((-1, 1))
scaler.fit(y_train.reshape(-1, 1))
ynorm_train = scaler.transform(y_train.reshape(-1, 1))
ynorm_test = scaler.transform(y_test.reshape(-1, 1))
print(ynorm_train.shape, ynorm_test.shape)
```

    (46549, 1) (11638, 1)
    

#### Neural Network for content-based filtering


```python
num_outputs = 32
tf.random.set_seed(1)
user_NN = tf.keras.models.Sequential([
    
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_outputs),
])

item_NN = tf.keras.models.Sequential([
    
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_outputs), 
])

# create the user input and point to the base network
input_user = tf.keras.layers.Input(shape=(num_user_features))
vu = user_NN(input_user)
vu = tf.linalg.l2_normalize(vu, axis=1)

# create the item input and point to the base network
input_item = tf.keras.layers.Input(shape=(num_item_features))
vm = item_NN(input_item)
vm = tf.linalg.l2_normalize(vm, axis=1)

# compute the dot product of the two vectors vu and vm
output = tf.keras.layers.Dot(axes=1)([vu, vm])

# specify the inputs and output of the model
model = Model([input_user, input_item], output)

model.summary()
```

    Model: "model"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     input_1 (InputLayer)           [(None, 14)]         0           []                               
                                                                                                      
     input_2 (InputLayer)           [(None, 16)]         0           []                               
                                                                                                      
     sequential (Sequential)        (None, 32)           40864       ['input_1[0][0]']                
                                                                                                      
     sequential_1 (Sequential)      (None, 32)           41376       ['input_2[0][0]']                
                                                                                                      
     tf.math.l2_normalize (TFOpLamb  (None, 32)          0           ['sequential[0][0]']             
     da)                                                                                              
                                                                                                      
     tf.math.l2_normalize_1 (TFOpLa  (None, 32)          0           ['sequential_1[0][0]']           
     mbda)                                                                                            
                                                                                                      
     dot (Dot)                      (None, 1)            0           ['tf.math.l2_normalize[0][0]',   
                                                                      'tf.math.l2_normalize_1[0][0]'] 
                                                                                                      
    ==================================================================================================
    Total params: 82,240
    Trainable params: 82,240
    Non-trainable params: 0
    __________________________________________________________________________________________________
    

We'll use a mean squared error loss and an Adam optimizer.


```python
tf.random.set_seed(1)
cost_fn = tf.keras.losses.MeanSquaredError()
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt,
              loss=cost_fn)
```


```python
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

    Num GPUs Available:  1
    


```python
tf.random.set_seed(1)
model.fit([user_train[:, u_s:], item_train[:, i_s:]], ynorm_train, epochs=30)
```

    Epoch 1/30
    1455/1455 [==============================] - 21s 11ms/step - loss: 0.1249
    Epoch 2/30
    1455/1455 [==============================] - 16s 11ms/step - loss: 0.1183
    Epoch 3/30
    1455/1455 [==============================] - 16s 11ms/step - loss: 0.1167
    Epoch 4/30
    1455/1455 [==============================] - 16s 11ms/step - loss: 0.1148
    Epoch 5/30
    1455/1455 [==============================] - 16s 11ms/step - loss: 0.1131
    Epoch 6/30
    1455/1455 [==============================] - 16s 11ms/step - loss: 0.1110
    Epoch 7/30
    1455/1455 [==============================] - 16s 11ms/step - loss: 0.1099
    Epoch 8/30
    1455/1455 [==============================] - 15s 11ms/step - loss: 0.1090
    Epoch 9/30
    1455/1455 [==============================] - 16s 11ms/step - loss: 0.1081
    Epoch 10/30
    1455/1455 [==============================] - 16s 11ms/step - loss: 0.1075
    Epoch 11/30
    1455/1455 [==============================] - 15s 10ms/step - loss: 0.1066
    Epoch 12/30
    1455/1455 [==============================] - 15s 10ms/step - loss: 0.1065
    Epoch 13/30
    1455/1455 [==============================] - 16s 11ms/step - loss: 0.1058
    Epoch 14/30
    1455/1455 [==============================] - 15s 11ms/step - loss: 0.1051
    Epoch 15/30
    1455/1455 [==============================] - 15s 10ms/step - loss: 0.1045
    Epoch 16/30
    1455/1455 [==============================] - 15s 10ms/step - loss: 0.1039
    Epoch 17/30
    1455/1455 [==============================] - 16s 11ms/step - loss: 0.1033
    Epoch 18/30
    1455/1455 [==============================] - 16s 11ms/step - loss: 0.1030
    Epoch 19/30
    1455/1455 [==============================] - 16s 11ms/step - loss: 0.1023
    Epoch 20/30
    1455/1455 [==============================] - 16s 11ms/step - loss: 0.1020
    Epoch 21/30
    1455/1455 [==============================] - 16s 11ms/step - loss: 0.1014
    Epoch 22/30
    1455/1455 [==============================] - 16s 11ms/step - loss: 0.1010
    Epoch 23/30
    1455/1455 [==============================] - 16s 11ms/step - loss: 0.1005
    Epoch 24/30
    1455/1455 [==============================] - 16s 11ms/step - loss: 0.1001
    Epoch 25/30
    1455/1455 [==============================] - 16s 11ms/step - loss: 0.0999
    Epoch 26/30
    1455/1455 [==============================] - 16s 11ms/step - loss: 0.0993
    Epoch 27/30
    1455/1455 [==============================] - 16s 11ms/step - loss: 0.0990
    Epoch 28/30
    1455/1455 [==============================] - 16s 11ms/step - loss: 0.0988
    Epoch 29/30
    1455/1455 [==============================] - 16s 11ms/step - loss: 0.0983
    Epoch 30/30
    1455/1455 [==============================] - 16s 11ms/step - loss: 0.0980
    




    <keras.callbacks.History at 0x2430f6b67f0>



Evaluate the model to determine loss on the test data. It is comparable to the training loss indicating the model has not substantially overfit the training data.


```python
model.evaluate([user_test[:, u_s:], item_test[:, i_s:]], ynorm_test)
```

    364/364 [==============================] - 2s 5ms/step - loss: 0.1045
    




    0.10449469089508057



### Predictions for a new user

First, we'll create a new user and have the model suggest movies for that user. After you have tried this example on the example user content, feel free to change the user content to match your own preferences and see what the model suggests. Note that ratings are between 0.5 and 5.0, inclusive, in half-step increments.


```python
new_user_id = 5000
new_rating_ave = 1.0
new_action = 1.0
new_adventure = 1
new_animation = 1
new_childrens = 1
new_comedy = 5
new_crime = 1
new_documentary = 1
new_drama = 1
new_fantasy = 1
new_horror = 1
new_mystery = 1
new_romance = 5
new_scifi = 5
new_thriller = 1
new_rating_count = 3

user_vec = np.array([[new_user_id, new_rating_count, new_rating_ave,
                      new_action, new_adventure, new_animation, new_childrens,
                      new_comedy, new_crime, new_documentary,
                      new_drama, new_fantasy, new_horror, new_mystery,
                      new_romance, new_scifi, new_thriller]])
```

Let's look at the top-rated movies for the new user. Recall, the user vector had genres that favored Comedy and Romance.
Below, we'll use a set of movie/item vectors, `item_vecs` that have a vector for each movie in the training/test set. This is matched with the user vector above and the scaled vectors are used to predict ratings for all the movies for our new user above.


```python
# generate and replicate the user vector to match the number movies in the data set.
user_vecs = gen_user_vecs(user_vec,len(item_vecs))

# scale the vectors and make predictions for all movies. Return results sorted by rating.
sorted_index, sorted_ypu, sorted_items, sorted_user = predict_uservec(user_vecs,  item_vecs, model, u_s, i_s, 
                                                                       scaler, scalerUser, scalerItem, scaledata=scaledata)

print_pred_movies(sorted_ypu, sorted_user, sorted_items, movie_dict, maxcount = 10)
```

    59/59 [==============================] - 0s 4ms/step
    




<table>
<thead>
<tr><th style="text-align: right;">    y_p</th><th style="text-align: right;">  movie id</th><th style="text-align: right;">  rating ave</th><th>title                                     </th><th>genres                </th></tr>
</thead>
<tbody>
<tr><td style="text-align: right;">4.84793</td><td style="text-align: right;">     76293</td><td style="text-align: right;">     3.31818</td><td>Date Night (2010)                         </td><td>Action|Comedy|Romance </td></tr>
<tr><td style="text-align: right;">4.82386</td><td style="text-align: right;">     69406</td><td style="text-align: right;">     3.5    </td><td>Proposal, The (2009)                      </td><td>Comedy|Romance        </td></tr>
<tr><td style="text-align: right;">4.8197 </td><td style="text-align: right;">     58047</td><td style="text-align: right;">     3.42857</td><td>Definitely, Maybe (2008)                  </td><td>Comedy|Drama|Romance  </td></tr>
<tr><td style="text-align: right;">4.81538</td><td style="text-align: right;">     62155</td><td style="text-align: right;">     3.35   </td><td>Nick and Norah&#x27;s Infinite Playlist (2008) </td><td>Comedy|Drama|Romance  </td></tr>
<tr><td style="text-align: right;">4.80771</td><td style="text-align: right;">     99007</td><td style="text-align: right;">     3.5    </td><td>Warm Bodies (2013)                        </td><td>Comedy|Horror|Romance </td></tr>
<tr><td style="text-align: right;">4.80431</td><td style="text-align: right;">     86882</td><td style="text-align: right;">     3.56   </td><td>Midnight in Paris (2011)                  </td><td>Comedy|Fantasy|Romance</td></tr>
<tr><td style="text-align: right;">4.80238</td><td style="text-align: right;">     56949</td><td style="text-align: right;">     3.3    </td><td>27 Dresses (2008)                         </td><td>Comedy|Romance        </td></tr>
<tr><td style="text-align: right;">4.79867</td><td style="text-align: right;">     54004</td><td style="text-align: right;">     3.45455</td><td>I Now Pronounce You Chuck and Larry (2007)</td><td>Comedy|Romance        </td></tr>
<tr><td style="text-align: right;">4.77714</td><td style="text-align: right;">      5377</td><td style="text-align: right;">     3.71591</td><td>About a Boy (2002)                        </td><td>Comedy|Drama|Romance  </td></tr>
<tr><td style="text-align: right;">4.77429</td><td style="text-align: right;">      5992</td><td style="text-align: right;">     3.7    </td><td>Hours, The (2002)                         </td><td>Drama|Romance         </td></tr>
</tbody>
</table>



### Predictions for an existing user.

Let's look at the predictions for "user 36", one of the users in the data set. We can compare the predicted ratings with the model's ratings. Note that movies with multiple genre's show up multiple times in the training data. For example,'The Time Machine' has three genre's: Adventure, Action, Sci-Fi


```python
uid =  36 
# form a set of user vectors. This is the same vector, transformed and repeated.
user_vecs, y_vecs = get_user_vecs(uid, scalerUser.inverse_transform(user_train), item_vecs, user_to_genre)

# scale the vectors and make predictions for all movies. Return results sorted by rating.
sorted_index, sorted_ypu, sorted_items, sorted_user = predict_uservec(user_vecs, item_vecs, model, u_s, i_s, scaler, 
                                                                      scalerUser, scalerItem, scaledata=scaledata)
sorted_y = y_vecs[sorted_index]

#print sorted predictions
print_existing_user(sorted_ypu, sorted_y.reshape(-1,1), sorted_user, sorted_items, item_features, ivs, uvs, movie_dict, maxcount = 10)
```

    59/59 [==============================] - 0s 4ms/step
    




<table>
<thead>
<tr><th style="text-align: right;">  y_p</th><th style="text-align: right;">  y</th><th style="text-align: right;">  user</th><th style="text-align: right;">  user genre ave</th><th style="text-align: right;">  movie rating ave</th><th>title                   </th><th>genres   </th></tr>
</thead>
<tbody>
<tr><td style="text-align: right;">  3.2</td><td style="text-align: right;">3.0</td><td style="text-align: right;">    36</td><td style="text-align: right;">            3.00</td><td style="text-align: right;">              2.86</td><td>Time Machine, The (2002)</td><td>Adventure</td></tr>
<tr><td style="text-align: right;">  3.1</td><td style="text-align: right;">3.0</td><td style="text-align: right;">    36</td><td style="text-align: right;">            3.00</td><td style="text-align: right;">              2.86</td><td>Time Machine, The (2002)</td><td>Action   </td></tr>
<tr><td style="text-align: right;">  3.0</td><td style="text-align: right;">3.0</td><td style="text-align: right;">    36</td><td style="text-align: right;">            3.00</td><td style="text-align: right;">              2.86</td><td>Time Machine, The (2002)</td><td>Sci-Fi   </td></tr>
<tr><td style="text-align: right;">  2.0</td><td style="text-align: right;">1.0</td><td style="text-align: right;">    36</td><td style="text-align: right;">            1.50</td><td style="text-align: right;">              4.00</td><td>Beautiful Mind, A (2001)</td><td>Drama    </td></tr>
<tr><td style="text-align: right;">  1.9</td><td style="text-align: right;">1.5</td><td style="text-align: right;">    36</td><td style="text-align: right;">            1.75</td><td style="text-align: right;">              3.52</td><td>Road to Perdition (2002)</td><td>Crime    </td></tr>
<tr><td style="text-align: right;">  1.9</td><td style="text-align: right;">2.0</td><td style="text-align: right;">    36</td><td style="text-align: right;">            1.75</td><td style="text-align: right;">              3.52</td><td>Gangs of New York (2002)</td><td>Crime    </td></tr>
<tr><td style="text-align: right;">  1.8</td><td style="text-align: right;">1.0</td><td style="text-align: right;">    36</td><td style="text-align: right;">            1.00</td><td style="text-align: right;">              4.00</td><td>Beautiful Mind, A (2001)</td><td>Romance  </td></tr>
<tr><td style="text-align: right;">  1.6</td><td style="text-align: right;">1.5</td><td style="text-align: right;">    36</td><td style="text-align: right;">            1.50</td><td style="text-align: right;">              3.52</td><td>Road to Perdition (2002)</td><td>Drama    </td></tr>
<tr><td style="text-align: right;">  1.6</td><td style="text-align: right;">2.0</td><td style="text-align: right;">    36</td><td style="text-align: right;">            1.50</td><td style="text-align: right;">              3.52</td><td>Gangs of New York (2002)</td><td>Drama    </td></tr>
</tbody>
</table>




```python
def sq_dist(a,b):
    """
    Returns the squared distance between two vectors
    Args:
      a (ndarray (n,)): vector with n features
      b (ndarray (n,)): vector with n features
    Returns:
      d (float) : distance
    """
    
    d = np.sum(np.square(a - b))
     
    return (d)
```


```python
input_item_m = tf.keras.layers.Input(shape=(num_item_features))    # input layer
vm_m = item_NN(input_item_m)                                       # use the trained item_NN
vm_m = tf.linalg.l2_normalize(vm_m, axis=1)                        # incorporate normalization as was done in the original model
model_m = Model(input_item_m, vm_m)                                
model_m.summary()
```

    Model: "model_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_3 (InputLayer)        [(None, 16)]              0         
                                                                     
     sequential_1 (Sequential)   (None, 32)                41376     
                                                                     
     tf.math.l2_normalize_2 (TFO  (None, 32)               0         
     pLambda)                                                        
                                                                     
    =================================================================
    Total params: 41,376
    Trainable params: 41,376
    Non-trainable params: 0
    _________________________________________________________________
    


```python
scaled_item_vecs = scalerItem.transform(item_vecs)
vms = model_m.predict(scaled_item_vecs[:,i_s:])
print(f"size of all predicted movie feature vectors: {vms.shape}")
```

    59/59 [==============================] - 0s 2ms/step
    size of all predicted movie feature vectors: (1883, 32)
    


```python
count = 50
dim = len(vms)
dist = np.zeros((dim,dim))

for i in range(dim):
    for j in range(dim):
        dist[i,j] = sq_dist(vms[i, :], vms[j, :])
        
m_dist = ma.masked_array(dist, mask=np.identity(dist.shape[0]))  # mask the diagonal

disp = [["movie1", "genres", "movie2", "genres"]]
for i in range(count):
    min_idx = np.argmin(m_dist[i])
    movie1_id = int(item_vecs[i,0])
    movie2_id = int(item_vecs[min_idx,0])
    genre1,_  = get_item_genre(item_vecs[i,:], ivs, item_features)
    genre2,_  = get_item_genre(item_vecs[min_idx,:], ivs, item_features)

    disp.append( [movie_dict[movie1_id]['title'], genre1,
                  movie_dict[movie2_id]['title'], genre2]
               )
table = tabulate.tabulate(disp, tablefmt='html', headers="firstrow", floatfmt=[".1f", ".1f", ".0f", ".2f", ".2f"])
table
```




<table>
<thead>
<tr><th>movie1                                </th><th>genres   </th><th>movie2                          </th><th>genres   </th></tr>
</thead>
<tbody>
<tr><td>Save the Last Dance (2001)            </td><td>Drama    </td><td>John Q (2002)                   </td><td>Drama    </td></tr>
<tr><td>Save the Last Dance (2001)            </td><td>Romance  </td><td>Wedding Planner, The (2001)     </td><td>Romance  </td></tr>
<tr><td>Wedding Planner, The (2001)           </td><td>Comedy   </td><td>Spy Kids (2001)                 </td><td>Comedy   </td></tr>
<tr><td>Wedding Planner, The (2001)           </td><td>Romance  </td><td>Sweetest Thing, The (2002)      </td><td>Romance  </td></tr>
<tr><td>Hannibal (2001)                       </td><td>Horror   </td><td>Resident Evil: Apocalypse (2004)</td><td>Horror   </td></tr>
<tr><td>Hannibal (2001)                       </td><td>Thriller </td><td>Sum of All Fears, The (2002)    </td><td>Thriller </td></tr>
<tr><td>Saving Silverman (Evil Woman) (2001)  </td><td>Comedy   </td><td>Cats &amp; Dogs (2001)              </td><td>Comedy   </td></tr>
<tr><td>Saving Silverman (Evil Woman) (2001)  </td><td>Romance  </td><td>Save the Last Dance (2001)      </td><td>Romance  </td></tr>
<tr><td>Down to Earth (2001)                  </td><td>Comedy   </td><td>Joe Dirt (2001)                 </td><td>Comedy   </td></tr>
<tr><td>Down to Earth (2001)                  </td><td>Fantasy  </td><td>Haunted Mansion, The (2003)     </td><td>Fantasy  </td></tr>
<tr><td>Down to Earth (2001)                  </td><td>Romance  </td><td>Joe Dirt (2001)                 </td><td>Romance  </td></tr>
<tr><td>Mexican, The (2001)                   </td><td>Action   </td><td>Knight&#x27;s Tale, A (2001)         </td><td>Action   </td></tr>
<tr><td>Mexican, The (2001)                   </td><td>Comedy   </td><td>Knight&#x27;s Tale, A (2001)         </td><td>Comedy   </td></tr>
<tr><td>15 Minutes (2001)                     </td><td>Thriller </td><td>Panic Room (2002)               </td><td>Thriller </td></tr>
<tr><td>Heartbreakers (2001)                  </td><td>Comedy   </td><td>Animal, The (2001)              </td><td>Comedy   </td></tr>
<tr><td>Heartbreakers (2001)                  </td><td>Crime    </td><td>Stepford Wives, The (2004)      </td><td>Thriller </td></tr>
<tr><td>Heartbreakers (2001)                  </td><td>Romance  </td><td>Bewitched (2005)                </td><td>Romance  </td></tr>
<tr><td>Spy Kids (2001)                       </td><td>Action   </td><td>Lara Croft: Tomb Raider (2001)  </td><td>Action   </td></tr>
<tr><td>Spy Kids (2001)                       </td><td>Adventure</td><td>Lara Croft: Tomb Raider (2001)  </td><td>Adventure</td></tr>
<tr><td>Spy Kids (2001)                       </td><td>Children </td><td>Princess Diaries, The (2001)    </td><td>Children </td></tr>
<tr><td>Spy Kids (2001)                       </td><td>Comedy   </td><td>Wedding Planner, The (2001)     </td><td>Comedy   </td></tr>
<tr><td>Along Came a Spider (2001)            </td><td>Action   </td><td>Swordfish (2001)                </td><td>Action   </td></tr>
<tr><td>Along Came a Spider (2001)            </td><td>Crime    </td><td>Swordfish (2001)                </td><td>Crime    </td></tr>
<tr><td>Along Came a Spider (2001)            </td><td>Mystery  </td><td>Ring, The (2002)                </td><td>Mystery  </td></tr>
<tr><td>Along Came a Spider (2001)            </td><td>Thriller </td><td>Signs (2002)                    </td><td>Thriller </td></tr>
<tr><td>Blow (2001)                           </td><td>Crime    </td><td>Training Day (2001)             </td><td>Crime    </td></tr>
<tr><td>Blow (2001)                           </td><td>Drama    </td><td>Training Day (2001)             </td><td>Drama    </td></tr>
<tr><td>Bridget Jones&#x27;s Diary (2001)          </td><td>Comedy   </td><td>Super Troopers (2001)           </td><td>Comedy   </td></tr>
<tr><td>Bridget Jones&#x27;s Diary (2001)          </td><td>Drama    </td><td>Others, The (2001)              </td><td>Drama    </td></tr>
<tr><td>Bridget Jones&#x27;s Diary (2001)          </td><td>Romance  </td><td>Punch-Drunk Love (2002)         </td><td>Romance  </td></tr>
<tr><td>Joe Dirt (2001)                       </td><td>Adventure</td><td>Bulletproof Monk (2003)         </td><td>Adventure</td></tr>
<tr><td>Joe Dirt (2001)                       </td><td>Comedy   </td><td>Dr. Dolittle 2 (2001)           </td><td>Comedy   </td></tr>
<tr><td>Joe Dirt (2001)                       </td><td>Mystery  </td><td>Grudge, The (2004)              </td><td>Mystery  </td></tr>
<tr><td>Joe Dirt (2001)                       </td><td>Romance  </td><td>Down to Earth (2001)            </td><td>Romance  </td></tr>
<tr><td>Crocodile Dundee in Los Angeles (2001)</td><td>Comedy   </td><td>Heartbreakers (2001)            </td><td>Comedy   </td></tr>
<tr><td>Crocodile Dundee in Los Angeles (2001)</td><td>Drama    </td><td>Scary Movie 4 (2006)            </td><td>Horror   </td></tr>
<tr><td>Mummy Returns, The (2001)             </td><td>Action   </td><td>Swordfish (2001)                </td><td>Action   </td></tr>
<tr><td>Mummy Returns, The (2001)             </td><td>Adventure</td><td>Rundown, The (2003)             </td><td>Adventure</td></tr>
<tr><td>Mummy Returns, The (2001)             </td><td>Comedy   </td><td>American Pie 2 (2001)           </td><td>Comedy   </td></tr>
<tr><td>Mummy Returns, The (2001)             </td><td>Thriller </td><td>Fast and the Furious, The (2001)</td><td>Thriller </td></tr>
<tr><td>Knight&#x27;s Tale, A (2001)               </td><td>Action   </td><td>Mexican, The (2001)             </td><td>Action   </td></tr>
<tr><td>Knight&#x27;s Tale, A (2001)               </td><td>Comedy   </td><td>Mexican, The (2001)             </td><td>Comedy   </td></tr>
<tr><td>Knight&#x27;s Tale, A (2001)               </td><td>Romance  </td><td>Monster&#x27;s Ball (2001)           </td><td>Romance  </td></tr>
<tr><td>Shrek (2001)                          </td><td>Adventure</td><td>Monsters, Inc. (2001)           </td><td>Adventure</td></tr>
<tr><td>Shrek (2001)                          </td><td>Animation</td><td>Monsters, Inc. (2001)           </td><td>Animation</td></tr>
<tr><td>Shrek (2001)                          </td><td>Children </td><td>Monsters, Inc. (2001)           </td><td>Children </td></tr>
<tr><td>Shrek (2001)                          </td><td>Comedy   </td><td>Monsters, Inc. (2001)           </td><td>Comedy   </td></tr>
<tr><td>Shrek (2001)                          </td><td>Fantasy  </td><td>Monsters, Inc. (2001)           </td><td>Fantasy  </td></tr>
<tr><td>Shrek (2001)                          </td><td>Romance  </td><td>Monsoon Wedding (2001)          </td><td>Romance  </td></tr>
<tr><td>Animal, The (2001)                    </td><td>Comedy   </td><td>Heartbreakers (2001)            </td><td>Comedy   </td></tr>
</tbody>
</table>


