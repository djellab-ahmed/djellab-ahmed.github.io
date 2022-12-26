# Character level language model - Dinosaurus Island


```python
import numpy as np
from utils import *
import random
import pprint
import copy
```


```python
data = open('dinos.txt', 'r').read()
data= data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))
```

    There are 19909 total characters and 27 unique characters in your data.
    


```python
chars = sorted(chars)
print(chars)
```

    ['\n', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    


```python
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(ix_to_char)
```

    {   0: '\n',
        1: 'a',
        2: 'b',
        3: 'c',
        4: 'd',
        5: 'e',
        6: 'f',
        7: 'g',
        8: 'h',
        9: 'i',
        10: 'j',
        11: 'k',
        12: 'l',
        13: 'm',
        14: 'n',
        15: 'o',
        16: 'p',
        17: 'q',
        18: 'r',
        19: 's',
        20: 't',
        21: 'u',
        22: 'v',
        23: 'w',
        24: 'x',
        25: 'y',
        26: 'z'}
    


```python
def clip(gradients, maxValue):
    '''
    Clips the gradients' values between minimum and maximum.
    
    Arguments:
    gradients -- a dictionary containing the gradients "dWaa", "dWax", "dWya", "db", "dby"
    maxValue -- everything above this number is set to this number, and everything less than -maxValue is set to -maxValue
    
    Returns: 
    gradients -- a dictionary with the clipped gradients.
    '''
    gradients = copy.deepcopy(gradients)
    
    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']

    # Clip to mitigate exploding gradients, loop over [dWax, dWaa, dWya, db, dby]. (≈2 lines)
    for gradient in gradients:
        np.clip(gradients[gradient], -maxValue, maxValue, out = gradients[gradient])
    
    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}
    
    return gradients
```


```python
def sample(parameters, char_to_ix, seed):
    """
    Sample a sequence of characters according to a sequence of probability distributions output of the RNN

    Arguments:
    parameters -- Python dictionary containing the parameters Waa, Wax, Wya, by, and b. 
    char_to_ix -- Python dictionary mapping each character to an index.
    seed -- Used for grading purposes. Do not worry about it.

    Returns:
    indices -- A list of length n containing the indices of the sampled characters.
    """

    # Retrieve parameters and relevant shapes from "parameters" dictionary
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]

    # Step 1: Create the a zero vector x that can be used as the one-hot vector 
    # Representing the first character (initializing the sequence generation). (≈1 line)
    x = np.zeros((vocab_size,1))
    # Step 1': Initialize a_prev as zeros (≈1 line)
    a_prev = np.zeros((n_a ,1))

    # Create an empty list of indices. This is the list which will contain the list of indices of the characters to generate (≈1 line)
    indices = []

    # idx is the index of the one-hot vector x that is set to 1
    # All other positions in x are zero.
    # Initialize idx to -1
    idx = -1

    # Loop over time-steps t. At each time-step:
    # Sample a character from a probability distribution 
    # And append its index (`idx`) to the list "indices". 
    # You'll stop if you reach 50 characters 
    # (which should be very unlikely with a well-trained model).
    # Setting the maximum number of characters helps with debugging and prevents infinite loops. 
    counter = 0
    newline_character = char_to_ix['\n']
    
    while (idx != newline_character and counter != 50):

        # Step 2: Forward propagate x using the equations (1), (2) and (3)
        a = np.tanh(np.dot(Wax,x) + np.dot(Waa,a_prev) + b)
        z = np.dot(Wya,a) + by
        y = softmax(z)

        # For grading purposes
        np.random.seed(counter + seed) 

        # Step 3: Sample the index of a character within the vocabulary from the probability distribution y
        # (see additional hints above)
        idx = np.random.choice(range(len(y)), p = np.squeeze(y) )

        # Append the index to "indices"
        indices.append(idx)

        # Step 4: Overwrite the input x with one that corresponds to the sampled index `idx`.
        # (see additional hints above)
        x = np.zeros((vocab_size,1))
        x[idx] = 1

        # Update "a_prev" to be "a"
        a_prev = a

        # for grading purposes
        seed += 1

        counter +=1


    if (counter == 50):
        indices.append(char_to_ix['\n'])
    
    return indices
```


```python
def optimize(X, Y, a_prev, parameters, learning_rate = 0.01):
    """
    Execute one step of the optimization to train the model.
    
    Arguments:
    X -- list of integers, where each integer is a number that maps to a character in the vocabulary.
    Y -- list of integers, exactly the same as X but shifted one index to the left.
    a_prev -- previous hidden state.
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        b --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    learning_rate -- learning rate for the model.
    
    Returns:
    loss -- value of the loss function (cross-entropy)
    gradients -- python dictionary containing:
                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dWya -- Gradients of hidden-to-output weights, of shape (n_y, n_a)
                        db -- Gradients of bias vector, of shape (n_a, 1)
                        dby -- Gradients of output bias vector, of shape (n_y, 1)
    a[len(X)-1] -- the last hidden state, of shape (n_a, 1)
    """
    
    
    # Forward propagate through time (≈1 line)
    loss, cache = rnn_forward(X, Y, a_prev, parameters)
    
    # Backpropagate through time (≈1 line)
    gradients, a = rnn_backward(X, Y, parameters, cache)
    
    # Clip your gradients between -5 (min) and 5 (max) (≈1 line)
    gradients = clip(gradients, 5)
    
    # Update parameters (≈1 line)
    parameters = update_parameters(parameters, gradients, learning_rate)
    
    
    return loss, gradients, a[len(X)-1]
```


```python
def model(data_x, ix_to_char, char_to_ix, num_iterations = 35000, n_a = 50, dino_names = 7, vocab_size = 27, verbose = False):
    """
    Trains the model and generates dinosaur names. 
    
    Arguments:
    data_x -- text corpus, divided in words
    ix_to_char -- dictionary that maps the index to a character
    char_to_ix -- dictionary that maps a character to an index
    num_iterations -- number of iterations to train the model for
    n_a -- number of units of the RNN cell
    dino_names -- number of dinosaur names you want to sample at each iteration. 
    vocab_size -- number of unique characters found in the text (size of the vocabulary)
    
    Returns:
    parameters -- learned parameters
    """
    
    # Retrieve n_x and n_y from vocab_size
    n_x, n_y = vocab_size, vocab_size
    
    # Initialize parameters
    parameters = initialize_parameters(n_a, n_x, n_y)
    
    # Initialize loss (this is required because we want to smooth our loss)
    loss = get_initial_loss(vocab_size, dino_names)
    
    # Build list of all dinosaur names (training examples).
    examples = [x.strip() for x in data_x]
    
    # Shuffle list of all dinosaur names
    np.random.seed(0)
    np.random.shuffle(examples)
    
    # Initialize the hidden state of your LSTM
    a_prev = np.zeros((n_a, 1))
    
    # for grading purposes
    last_dino_name = "abc"
    
    # Optimization loop
    for j in range(num_iterations):
        
        
        # Set the index `idx` (see instructions above)
        idx = j%len(examples)
        
        # Set the input X (see instructions above)
        single_example_chars = examples[idx]
        single_example_ix = [char_to_ix[c] for c in single_example_chars]

        # if X[t] == None, we just have x[t]=0. This is used to set the input for the first timestep to the zero vector. 
        X = [None] + single_example_ix
        
        # Set the labels Y (see instructions above)
        # The goal is to train the RNN to predict the next letter in the name
        # So the labels are the list of characters that are one time-step ahead of the characters in the input X
        Y = X[1:] 
        # The RNN should predict a newline at the last letter, so add ix_newline to the end of the labels
        ix_newline = [char_to_ix["\n"]]
        Y = Y + ix_newline

        # Perform one optimization step: Forward-prop -> Backward-prop -> Clip -> Update parameters
        # Choose a learning rate of 0.01
        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters, learning_rate = 0.01)
       
        
        # debug statements to aid in correctly forming X, Y
        if verbose and j in [0, len(examples) -1, len(examples)]:
            print("j = " , j, "idx = ", idx,) 
        if verbose and j in [0]:
            #print("single_example =", single_example)
            print("single_example_chars", single_example_chars)
            print("single_example_ix", single_example_ix)
            print(" X = ", X, "\n", "Y =       ", Y, "\n")
        
        # Use a latency trick to keep the loss smooth. It happens here to accelerate the training.
        loss = smooth(loss, curr_loss)

        # Every 1000 Iteration, generate "n" characters thanks to sample() to check if the model is learning properly
        if j % 1000 == 0:
            
            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')
            
            # The number of dinosaur names to print
            seed = 0
            for name in range(dino_names):
                
                # Sample indices and print them
                sampled_indices = sample(parameters, char_to_ix, seed)
                last_dino_name = get_sample(sampled_indices, ix_to_char)
                print(last_dino_name.replace('\n', ''))
                
                seed += 1  # To get the same result (for grading purposes), increment the seed by one. 
      
            print('\n')
        
    return parameters, last_dino_name
```


```python
parameters, last_name = model(data.split("\n"), ix_to_char, char_to_ix, 22001, verbose = True)

assert last_name == 'Trodonosaurus\n', "Wrong expected output"
print("\033[92mAll tests passed!")
```

    j =  0 idx =  0
    single_example_chars turiasaurus
    single_example_ix [20, 21, 18, 9, 1, 19, 1, 21, 18, 21, 19]
     X =  [None, 20, 21, 18, 9, 1, 19, 1, 21, 18, 21, 19] 
     Y =        [20, 21, 18, 9, 1, 19, 1, 21, 18, 21, 19, 0] 
    
    Iteration: 0, Loss: 23.087336
    
    Nkzxwtdmfqoeyhsqwasjkjvu
    Kneb
    Kzxwtdmfqoeyhsqwasjkjvu
    Neb
    Zxwtdmfqoeyhsqwasjkjvu
    Eb
    Xwtdmfqoeyhsqwasjkjvu
    
    
    Iteration: 1000, Loss: 28.712699
    
    Nivusahidoraveros
    Ioia
    Iwtroeoirtaurusabrngeseaosawgeanaitafeaolaeratohop
    Nac
    Xtroeoirtaurusabrngeseaosawgeanaitafeaolaeratohopr
    Ca
    Tseeohnnaveros
    
    
    j =  1535 idx =  1535
    j =  1536 idx =  0
    Iteration: 2000, Loss: 27.884160
    
    Liusskeomnolxeros
    Hmdaairus
    Hytroligoraurus
    Lecalosapaus
    Xusicikoraurus
    Abalpsamantisaurus
    Tpraneronxeros
    
    
    Iteration: 3000, Loss: 26.863598
    
    Niusos
    Infa
    Iusrtendor
    Nda
    Wtrololos
    Ca
    Tps
    
    
    Iteration: 4000, Loss: 25.901815
    
    Mivrosaurus
    Inee
    Ivtroplisaurus
    Mbaaisaurus
    Wusichisaurus
    Cabaselachus
    Toraperlethosdarenitochusthiamamumamaon
    
    
    Iteration: 5000, Loss: 25.290275
    
    Ngyusedonis
    Klecagropechus
    Lytosaurus
    Necagropechusangotmeeycerum
    Xuskangosaurus
    Da
    Tosaurus
    
    
    Iteration: 6000, Loss: 24.608779
    
    Onwusceomosaurus
    Lieeaerosaurus
    Lxussaurus
    Oma
    Xusteonosaurus
    Eeahosaurus
    Toreonosaurus
    
    
    Iteration: 7000, Loss: 24.425330
    
    Ngytromiasaurus
    Ingabcosaurus
    Kyusichiropurusanrasauraptous
    Necamithachusidinysaus
    Yusodon
    Caaesaurus
    Tosaurus
    
    
    Iteration: 8000, Loss: 24.070350
    
    Onxusichepriuon
    Kilabersaurus
    Lutrodon
    Omaaerosaurus
    Xutrcheps
    Edaksoje
    Trodiktonus
    
    
    Iteration: 9000, Loss: 23.730944
    
    Onyusaurus
    Klecanotal
    Kyuspang
    Ogaacosaurus
    Xutrasaurus
    Dabcosaurus
    Troching
    
    
    Iteration: 10000, Loss: 23.844446
    
    Onyusaurus
    Klecalosaurus
    Lustodon
    Ola
    Xusodonia
    Eeaeosaurus
    Troceosaurus
    
    
    Iteration: 11000, Loss: 23.581901
    
    Leutosaurus
    Inda
    Itrtoplerosherotarangos
    Lecalosaurus
    Xutogolosaurus
    Babator
    Trodonosaurus
    
    
    Iteration: 12000, Loss: 23.291971
    
    Onyxosaurus
    Kica
    Lustrepiosaurus
    Olaagrraiansaurus
    Yuspangosaurus
    Eealosaurus
    Trognesaurus
    
    
    Iteration: 13000, Loss: 23.547611
    
    Nixrosaurus
    Indabcosaurus
    Jystolong
    Necalosaurus
    Yuspangosaurus
    Daagosaurus
    Usndicirax
    
    
    Iteration: 14000, Loss: 23.382338
    
    Meutromodromurus
    Inda
    Iutroinatorsaurus
    Maca
    Yusteratoptititan
    Ca
    Troclosaurus
    
    
    Iteration: 15000, Loss: 23.049663
    
    Phyus
    Lica
    Lustrapops
    Padaeron
    Yuspcheosaurus
    Eeagosaurus
    Trochirnathus
    
    
    Iteration: 16000, Loss: 23.265759
    
    Meustoloplohus
    Imeda
    Iutosaurus
    Maca
    Yuspanenphurus
    Daaisicachtitan
    Trodon
    
    
    Iteration: 17000, Loss: 23.151825
    
    Oruston
    Kiacaissceerthsaurus
    Lustonomoraviosiadon
    Olaadrus
    Yustarisaurus
    Egajosaurus
    Trtarhiomungos
    
    
    Iteration: 18000, Loss: 22.901372
    
    Phytrohaesaurus
    Melaaisaurus
    Mystoosaurus
    Peeamosaurus
    Ytronosaurus
    Eiakosaurus
    Trogonosaurus
    
    
    Iteration: 19000, Loss: 23.018827
    
    Niwushangosaurus
    Klecahps
    Kystolongheus
    Nedaisur
    Yussargosaurus
    Ehahrsegassdriimus
    Trrirhis
    
    
    Iteration: 20000, Loss: 22.924847
    
    Nixusaurus
    Llecalosaurus
    Lxusodon
    Necalosaurus
    Ystrengosaurus
    Eg
    Trochkosaurus
    
    
    Iteration: 21000, Loss: 22.716377
    
    Opusaurus
    Lmecalosaurus
    Lyutonnashaycosaurus
    Ola
    Yusodomasaurus
    Efaispia
    Trocomadaurosedilidos
    
    
    Iteration: 22000, Loss: 22.759659
    
    Piustolonosaurus
    Migbaeron
    Myrrocepholus
    Peeadosaurus
    Yusodomincteros
    Eiadosaurus
    Trocephods
    
    
    


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    Cell In [10], line 3
          1 parameters, last_name = model(data.split("\n"), ix_to_char, char_to_ix, 22001, verbose = True)
    ----> 3 assert last_name == 'Trodonosaurus\n', "Wrong expected output"
          4 print("\033[92mAll tests passed!")
    

    AssertionError: Wrong expected output


## Writing like Shakespeare


```python
from __future__ import print_function
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.sequence import pad_sequences
from shakespeare_utils import *
import sys
import io
```

    Loading text data...
    Creating training set...
    number of training examples: 31412
    Vectorizing training set...
    Loading model...
    WARNING:tensorflow:Layer lstm_5 will not use cuDNN kernels since it doesn't meet the criteria. It will use a generic GPU kernel as fallback when running on GPU.
    WARNING:tensorflow:Layer lstm_6 will not use cuDNN kernels since it doesn't meet the criteria. It will use a generic GPU kernel as fallback when running on GPU.
    


```python
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y, batch_size=128, epochs=1, callbacks=[print_callback])
```

    246/246 [==============================] - 354s 1s/step - loss: 2.5694
    




    <keras.callbacks.History at 0x20ae0b60b20>




```python
# Run this cell to try with different inputs without having to re-train the model 
generate_output()
```

    Write the beginning of your poem, the Shakespeare machine will complete it. Your input is: Machine learning
    
    
    Here is your poem: 
    
    Machine learning,
    bit a dofl the that the my hawenth past ever laks,
    for thus caf bey pomsuon dorse to kont anr.
    would your spart co bri ear fortwliell chere asele.
     live love me of athore's hy, goot dement,
    in whichst the yours worwn mreele anfowht's ans dave.
    so glich a limet thought ay his holoakey?
    srie in'thiing of lope in gost as rakned thee braus, and astady winth,
    with my mispecion tiell ou wront i that,
    
