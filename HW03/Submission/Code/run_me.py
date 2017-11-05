# Import python modules
import numpy as np
import kaggle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

def fxn():
    warnings.warn("deprecated", category=RuntimeWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
    from scipy import *
    from sklearn.linear_model.ridge import Ridge
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.model_selection import KFold
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import mean_squared_error
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import make_scorer
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    
############################################################################
# Read in train and test synthetic data
def read_synthetic_data():
	print('Reading synthetic data ...')
	train_x = np.loadtxt('../../Data/Synthetic/data_train.txt', delimiter = ',', dtype=float)
	train_y = np.loadtxt('../../Data/Synthetic/label_train.txt', delimiter = ',', dtype=float)
	test_x = np.loadtxt('../../Data/Synthetic/data_test.txt', delimiter = ',', dtype=float)
	test_y = np.loadtxt('../../Data/Synthetic/label_test.txt', delimiter = ',', dtype=float)

	return (train_x, train_y, test_x, test_y)

############################################################################
# Read in train and test credit card data
def read_creditcard_data():
	print('Reading credit card data ...')
	train_x = np.loadtxt('../../Data/CreditCard/data_train.txt', delimiter = ',', dtype=float)
	train_y = np.loadtxt('../../Data/CreditCard/label_train.txt', delimiter = ',', dtype=float)
	test_x = np.loadtxt('../../Data/CreditCard/data_test.txt', delimiter = ',', dtype=float)

	return (train_x, train_y, test_x)

############################################################################
# Read in train and test tumor data
def read_tumor_data():
    print('Reading tumor data ...')
    train_x = np.loadtxt('../../Data/Tumor/data_train.txt', delimiter = ',', dtype=float)
    train_y = np.loadtxt('../../Data/Tumor/label_train.txt', delimiter = ',', dtype=float)
    test_x = np.loadtxt('../../Data/Tumor/data_test.txt', delimiter = ',', dtype=float)
    return (train_x, train_y, test_x)

############################################################################
# Compute MSE
def compute_MSE(y, y_hat):
	# mean squared error
	return np.mean(np.power(y - y_hat, 2))

############################################################################

train_x, train_y, test_x, test_y = read_synthetic_data()
print('Train=', train_x.shape)
print('Test=', test_x.shape)

##############################################################################
# reshaping train_x and train_y
train_x = np.expand_dims(train_x, axis = 1)
test_x = np.expand_dims(test_x, axis = 1)

# printing train_x and test_x
train_x.shape
test_x.shape

# Lambda equals 0.1
lambd = 0.1

# implementing h for the first problem
def k1(X1, X2, i):
    tmp1 = (1 + np.dot(X1.T, X2))**i
    return tmp1

# Trignometry h implementation for the KRRS
def k2(X1, X2, i):
    result = 1
    for k in range(i):
        result += np.dot(np.sin(k*0.5*X1),np.sin(k*0.5*X2)) + np.dot(np.cos(k*0.5*X1),np.cos(k*0.5*X2))
    return result

# Basis Expension for polynomial function.
def BasisExpension(X1, i):
    sig1 = [1]
    for p in range(i + 1):
        sig1.append(X1**p)
    return sig1

# Basis function for the trignometry function
def BasisExpension_trig(X1,i):
    sig1 = [1]
    for k in range(i):
        sig1.append(np.sin(k*0.5*X1))
        sig1.append(np.cos(k*0.5*X1))
    return sig1

# Calculating K for the kernel ridge regression 
def kernel_ridge_Basis(X, h, power):
    # Calculate alpha
    K = []
    for i in range(len(X)):
        K.append(h(X[i],power))
    return K

# Prediction alpha for the kernel ridge regression
def kernel_ridge_linear(X, Y, h, lambd, power):
    # Calculate alpha
    K = np.empty([len(X), len(X)])
    lambd_identity = lambd*np.identity(X.shape[0])
    for i in range(len(X)):
        for j in range(len(X)):
            K[i][j] = h(X[i], X[j], power)
    alpha  = np.dot(np.linalg.inv(K + lambd_identity), Y)
    return alpha 
 
# prediting new data for kernel ridge regression
def predit_kernel(train_x, test_x, train_y, lambd, power, h):
    alpha = kernel_ridge_linear(train_x, train_y,h, lambd, power) 
    predicted = []
    for i in range(len(test_x)):
        result = 0
        for j in  range(len(train_x)):
            result += alpha[j]*h(test_x[i], train_x[j], power)
        predicted.append(result)
    predicted = np.asarray(predicted)
    return predicted
  
# Kernel ridge regression 
def kernel_prediction(degree, kernel):
    res_pred = []
    res_mse = []
    for d in degree:
       #title = 'KRSS, ' + str(typ) + ", degree: " + str(d) + ", lambda: 0.1"
       pred_y = predit_kernel(train_x, test_x, train_y, lambd, d, kernel)
       res_pred.append(pred_y)
       mse = compute_MSE(test_y, pred_y)
       print('mse: ', mse)
       res_mse.append(mse)
    return res_pred, res_mse

def BasisPrediction(degree, function):
    # Basis function Prediction
    res_pred = []
    res_mse = []
    for d in degree:
       #title = 'KRSS, ' + str(typ) + ", degree: " + str(d) + ", lambda: 0.1"
       K = kernel_ridge_Basis(train_x, function, d)
       K_test = kernel_ridge_Basis(test_x, function, d)
       clf = Ridge()
       clf.fit(K, train_y)
       pred_y = clf.predict(K_test)    
       res_pred.append(pred_y)
       mse = compute_MSE(test_y, pred_y)
       print('mse: ', mse)
       res_mse.append(mse)
    return res_pred, res_mse

# Printing the Mean Squared errors
print("Kernel ridge regression scratch (KRRS) Polynomial order i [1,2,4,6] Mean Squared errors are")
predk1, mse1 = kernel_prediction([1, 2, 4, 6], k1)

print()

print("Kernel ridge regression scratch (KRRS) Trigonometric order i [3, 5, 10] Mean Squared errors are")
predk2, mse2 = kernel_prediction([3, 5, 10], k2)
print()
print("Basis expansion + ridge regression (BERR) Polynomial order i [1,2,4,6] Mean Squared errors are")
predB1, mse3 = BasisPrediction([1, 2, 4, 6], BasisExpension)
print()
print("Basis expansion + ridge regression (BERR) Trigonometric order i [3, 5, 10] Mean Squared errors are")
predB2, mse4 = BasisPrediction([3, 5, 10], BasisExpension_trig)
print()
# Taking the values of Polynomial orders 2 and 6
predk12, predk16, predB12, predB16 = predk1[1], predk1[3], predB1[1], predB1[3]

# Taking the values of Trignometric orders  and 6
predk22, predk26, predB22, predB26 = predk2[1], predk2[2], predB2[1], predB2[2]

prediction = [[predk12, predk16], [predk22, predk26], [predB12, predB16], [predB22, predB26]]
Titles = [["KRRS, Polynomial, degree = 2, lambda = 0.1", "KRRS, Polynomial, degree = 6, lambda = 0.1"],
          ["KRRS, Trignometric, degree = 5, lambda = 0.1", "KRRS, Trignometric, degree = 10, lambda = 0.1"],
          ["BERR, Polynomial, degree = 2, lambda = 0.1", "BERR, Polynomial, degree = 6, lambda = 0.1"],
          ["BERR, Trignometric, degree = 5, lambda = 0.1", "BERR, Trignometric, degree = 10, lambda = 0.1"]]

def plot():
    fig, axes = plt.subplots(4, 2)
    for i in range(4):
        for j in range(2):
            axes[i, j].scatter(test_x, test_y, c = 'b', marker = "*")
            axes[i, j].scatter(test_x, prediction[i][j], c = 'r',marker = "o")
            axes[i, j].set_title(Titles[i][j])
            axes[i, j].set_xlabel("Test X")
            axes[i, j].set_ylabel("True/Predicted Y")  
    plt.tight_layout(pad=0.1, w_pad=0.2, h_pad=-1.5)
    plt.savefig('../Figures/myfig.png') 
    plt.show()      
# Printing the plot and saving it to figure directory     
print("Printing the Graph and saving into the figures folder")
plot()


# Reading Credit card data
train_x, train_y, test_x  = read_creditcard_data()
print('Train=', train_x.shape)
print('Test=', test_x.shape)

# Grid Search Function 
def GridSearch(clf, parameter, folds, score = None):
    if score == "Accuracy":
        print("Accuracy")
        scoring = make_scorer(accuracy_score)
        grid = GridSearchCV(clf, param_grid= parameter, cv = folds, scoring = "accuracy") # performing gridsearch
    else:
        #scoring = {'accuracy': make_scorer(mean_squared_error)}
        print("Mean Squared Error")
        scoring = make_scorer(mean_squared_error, greater_is_better=False)
        grid = GridSearchCV(clf, param_grid= parameter, cv = folds, scoring = "neg_mean_squared_error")
    grid.fit(train_x, train_y) # fitting the data on gridsearch
    best_clf = grid.best_estimator_ #taking the best estimator
    best_clf.fit(train_x, train_y) # fitting the training data in the best estimator
    predicted_y = best_clf.predict(test_x)  # predicting the test data
    return grid.best_params_, grid.grid_scores_, predicted_y

            
clf = KernelRidge()
parameter = {"kernel": ['rbf', 'poly', 'linear'], "alpha": [1, 0.0001], "gamma": [ None, 1, 0.001]}
best_param, grid_scores, predicted_y = GridSearch(clf, parameter, 10, score = "MeanSquaredError")
print("Printing the Best classifier for Credit Card Dataset Using Kernel Regression: ", best_param)
# Output file location
file_name = '../Predictions/CreditCard/best.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name, True)


# Extra credit 
clf = KernelRidge()
parameter = {"kernel": ['rbf', 'poly', 'linear', 'sigmoid'], 
             "alpha": [1,0.001, 0.00001, 0.0001], 
             "gamma": [ None, 1, 0.001, 0.01],
             'degree': [1, 3, 4, 8]}

best_param, grid_scores, predicted_y = GridSearch(clf, parameter, 5, score = "MeanSquaredError")
print("Printing the Best classifier for Credit Card Dataset Using Kernel Regression: ", best_param)
# Output file location
file_name = '../Predictions/CreditCard/bestextra.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name, True)


# Tumor dataset
print("Now about to start to working on Tumor Dataset")
train_x, train_y, test_x  = read_tumor_data()
print('Train=', train_x.shape)
print('Test=', test_x.shape)


# performing the grid search for the following parameters with different clf
parameter = {"C" : [1, 0.01, 0.0001], "gamma": [1, 0.01, 0.001],"kernel" : ["rbf", 'poly','linear'], 'degree':[3, 5]}
clf = SVC()
best_param, grid_scores, predicted_y = GridSearch(clf, parameter, 4, score = "Accuracy")
print("Printing the Best classifier for Tumor using SVM: ", best_param)
# Writing output in Kaggle format
file_name = '../Predictions/Tumor/best.csv'
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name, False)

