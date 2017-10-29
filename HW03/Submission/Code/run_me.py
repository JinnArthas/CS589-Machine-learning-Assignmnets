# Import python modules
import numpy as np
import kaggle
from sklearn.metrics import accuracy_score
from GridSearch import GridSearch
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

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

train_x, train_y, test_x  = read_creditcard_data()
print('Train=', train_x.shape)
print('Test=', test_x.shape)

# Create dummy test output values to compute MSE
test_y = np.random.rand(test_x.shape[0], train_y.shape[1])
predicted_y = np.random.rand(test_x.shape[0], train_y.shape[1])
print('DUMMY MSE=%0.4f' % compute_MSE(test_y, predicted_y))

# Output file location
file_name = '../Predictions/CreditCard/best.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name, True)

train_x, train_y, test_x  = read_tumor_data()
print('Train=', train_x.shape)
print('Test=', test_x.shape)

def GridSearch(clf, parameter, folds):
    scoring = {'accuracy': make_scorer(accuracy_score)} # scoring = Accuracy
    grid = GridSearchCV(clf, param_grid= parameter, cv = folds, scoring= "accuracy") # performing gridsearch
    grid.fit(train_x, train_y) # fitting the data on gridsearch
    best_clf = grid.best_estimator_ #taking the best estimator
    best_clf.fit(train_x, train_y) # fitting the training data in the best estimator
    predicted_y = best_clf.predict(test_x)  # predicting the test data
    #print("The best parameter: ", grid.best_params_)
    #print("grid Score for the best parameter: ", grid.grid_scores_)
    return grid.best_params_, grid.grid_scores_, predicted_y

# performing the grid search for the following parameters with different clf

parameter = {"C" : [1, 0.01, 0.0001], "gamma": [1, 0.01, 0.0001],"kernel" : ['linear', "rbf", 'poly'], 'degree':[3, 5]}
clf = SVC()
best_param, grid_scores, predicted_y = GridSearch(clf, parameter, 5)

# Writing output in Kaggle format
file_name = '../Predictions/Tumor/best.csv'
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name, False)

