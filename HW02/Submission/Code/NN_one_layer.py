import autograd.numpy as np
import autograd
from autograd.util import flatten
import matplotlib.pyplot as plt
import seaborn as sns 
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import kaggle

# Helper variables to store the validation accuracy and training time and training error
training_time = []
train_acc = {5:[], 40:[], 70:[]}
train_err = {5:[], 40:[], 70:[]}
mean_logistic_loss = {5:[], 40:[], 70:[]}
val_error = []

# Function to compute classification accuracy
def mean_zero_one_loss(weights, x, y_integers, unflatten):
	(W, b, V, c) = unflatten(weights)
	out = feedForward(W, b, V, c, x)
	pred = np.argmax(out, axis=1)
	return(np.mean(pred != y_integers))

# Feed forward output i.e. L = -O[y] + log(sum(exp(O[j])))
def feedForward(W, b, V, c, train_x):
        hid = np.tanh(np.dot(train_x, W) + b)
        out = np.dot(hid, V) + c
        return out

# Logistic Loss function
def logistic_loss_batch(weights, x, y, unflatten):
	# regularization penalty
        lambda_pen = 10

        # unflatten weights into W, b, V and c respectively 
        (W, b, V, c) = unflatten(weights)

        # Predict output for the entire train data
        out  = feedForward(W, b, V, c, x)
        pred = np.argmax(out, axis=1)

	# True labels
        true = np.argmax(y, axis=1)
        # Mean accuracy
        class_err = np.mean(pred != true)

        # Computing logistic loss with l2 penalization
        logistic_loss = np.sum(-np.sum(out * train_y, axis=1) + np.log(np.sum(np.exp(out),axis=1))) + lambda_pen * np.sum(weights**2)
        
        # returning loss. Note that outputs can only be returned in the below format
        return (logistic_loss, [autograd.util.getval(logistic_loss), autograd.util.getval(class_err)])




# Loading the dataset
print('Reading image data ...')
temp = np.load('../../Data/data_train.npz')
train_x = temp['data_train']
temp = np.load('../../Data/labels_train.npz')
train_y_integers = temp['labels_train']
temp = np.load('../../Data/data_test.npz')
test_x = temp['data_test']
X_train, X_test, y_train_integer, y_test = train_test_split(train_x, train_y_integers, test_size = .20, shuffle = True)

# Mean Normalization
X_train -= .5
X_test  -= .5

# training neural netwrok with 5, 40, 70 hidden layers.
for m in [5, 40, 70]:
    # Make inputs approximately zero mean (improves optimization backprob algorithm in NN)
    x1 = time.time()
    
    
    # Number of output dimensions
    dims_out = 4
    # Number of hidden units
    dims_hid = m # Make this 5 , 40, 70
    # Learning rate
    epsilon = 0.0001
    # Momentum of gradients update
    momentum = 0.1
    # Number of epochs
    nEpochs = 1000
    # Number of train examples
    nTrainSamples = X_train.shape[0]
    # Number of input dimensions
    dims_in = X_train.shape[1]
    t1 = time.time()
    # Convert integer labels to one-hot vectors
    # i.e. convert label 2 to 0, 0, 1, 0
    train_y = np.zeros((nTrainSamples, dims_out))
    train_y[np.arange(nTrainSamples), y_train_integer] = 1
    
    assert momentum <= 1
    assert epsilon <= 1
    
    # Batch compute the gradients (partial derivatives of the loss function w.r.t to all NN parameters)
    grad_fun = autograd.grad_and_aux(logistic_loss_batch)
    
    # Initializing weights
    W = np.random.randn(dims_in, dims_hid)
    b = np.random.randn(dims_hid)
    V = np.random.randn(dims_hid, dims_out)
    c = np.random.randn(dims_out)
    smooth_grad = 0
    
    # Compress all weights into one weight vector using autograd's flatten
    all_weights = (W, b, V, c)
    weights, unflatten = flatten(all_weights)
    
    for epoch in range(nEpochs):
        print(epoch)
        # Compute gradients (partial derivatives) using autograd toolbox
        weight_gradients, returned_values = grad_fun(weights, X_train, train_y, unflatten)
        print('logistic loss: ', returned_values[0], 'Train error =', returned_values[1])
        mean_logistic_loss[m].append(returned_values[0])
        train_err[m].append(returned_values[1])
        # Update weight vector
        smooth_grad = (1 - momentum) * smooth_grad + momentum * weight_gradients
        weights = weights - epsilon * smooth_grad
        
        print('Train accuracy =', 1-mean_zero_one_loss(weights, X_train, y_train_integer, unflatten))
        train_acc[m].append(returned_values[1])
    print('Validation Accuracy =',1-mean_zero_one_loss(weights, X_train, y_train_integer, unflatten))
    val_error.append([m, 1- mean_zero_one_loss(weights, X_test, y_test, unflatten)])
    training_time.append([m, (time.time() - x1)*1000])

# Plotting the mean logistic error against the total number of epoch.
df = pd.DataFrame(mean_logistic_loss)
labels = df.mean().index
plt.figure(2, figsize=(6,4))  #6x4 is the aspect ratio for the plot
plt.plot(df[5],'or-', linewidth=3) #Plot the first series in red with circle marker
plt.plot(df[40], linewidth=3)
plt.plot(df[70], linewidth=3)
plt.ylabel("Mean Logistic Error") #Y-axis label
plt.xlabel("Number of Epoch") #X-axis label
plt.title("Mean Logistic Error vs Number of Epoch")
plt.legend(labels)
plt.show()

# Printing the Time for each model.
print(training_time)
# Printing the Validation error
print(val_error)
# training model with the full dataset and reporting the predicted output.


print('Reading image data ...')
temp = np.load('../../Data/data_train.npz')
train_x = temp['data_train']
temp = np.load('../../Data/labels_train.npz')
train_y_integers = temp['labels_train']
temp = np.load('../../Data/data_test.npz')
test_x = temp['data_test']

# Make inputs approximately zero mean (improves optimization backprob algorithm in NN)
train_x -= .5
test_x  -= .5

# Number of output dimensions
dims_out = 4
# Number of hidden units
dims_hid = 70
# Learning rate
epsilon = 0.0001
# Momentum of gradients update
momentum = 0.1
# Number of epochs
nEpochs = 5000
# Number of train examples
nTrainSamples = train_x.shape[0]
# Number of input dimensions
dims_in = train_x.shape[1]

# Convert integer labels to one-hot vectors
# i.e. convert label 2 to 0, 0, 1, 0
train_y = np.zeros((nTrainSamples, dims_out))
train_y[np.arange(nTrainSamples), train_y_integers] = 1

assert momentum <= 1
assert epsilon <= 1

# Batch compute the gradients (partial derivatives of the loss function w.r.t to all NN parameters)
grad_fun = autograd.grad_and_aux(logistic_loss_batch)

# Initializing weights
W = np.random.randn(dims_in, dims_hid)
b = np.random.randn(dims_hid)
V = np.random.randn(dims_hid, dims_out)
c = np.random.randn(dims_out)
smooth_grad = 0

# Compress all weights into one weight vector using autograd's flatten
all_weights = (W, b, V, c)
weights, unflatten = flatten(all_weights)
for epoch in range(nEpochs):
    # Compute gradients (partial derivatives) using autograd toolbox
    print("Epoch", epoch)
    weight_gradients, returned_values = grad_fun(weights, train_x, train_y, unflatten)
    print('logistic loss: ', returned_values[0], 'Train error =', returned_values[1])
    
    # Update weight vector
    smooth_grad = (1 - momentum) * smooth_grad + momentum * weight_gradients
    weights = weights - epsilon * smooth_grad
    
    print('Train accuracy =', 1-mean_zero_one_loss(weights, train_x, train_y_integers, unflatten))
(W, b, V, c) = unflatten(weights)
#import pickle
#all_weights = (W, b, V, c)
#output = open('allWeight.txt', 'wb')
## Pickle dictionary using protocol 0.
#pickle.dump(all_weights, output)
#pkl_file = open('allWeight.txt', 'rb')
#data1 = pickle.load(pkl_file)
(W, b, V, c) = unflatten(weights)
out  = feedForward(W, b, V, c, test_x)
pred_y = np.argmax(out, axis=1)
file_name = '../Predictions/best.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(pred_y, file_name)