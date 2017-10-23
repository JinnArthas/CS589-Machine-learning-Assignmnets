# Import python modules
import numpy as np
import kaggle
# All the helper function (GridSearch, regression) are implement here
from helperfile import * 


#from Tree import full_decision_tree
# Read in train and test data
def read_data_power_plant():
	print('Reading power plant dataset ...')
	train_x = np.loadtxt('../../Data/PowerOutput/data_train.txt')
	train_y = np.loadtxt('../../Data/PowerOutput/labels_train.txt')
	test_x = np.loadtxt('../../Data/PowerOutput/data_test.txt')

	return (train_x, train_y, test_x)

def read_data_localization_indoors():
	print('Reading indoor localization dataset ...')
	train_x = np.loadtxt('../../Data/IndoorLocalization/data_train.txt')
	train_y = np.loadtxt('../../Data/IndoorLocalization/labels_train.txt')
	test_x = np.loadtxt('../../Data/IndoorLocalization/data_test.txt')

	return (train_x, train_y, test_x)

# Compute MAE
def compute_error(y_hat, y):
	# mean absolute error
	return np.abs(y_hat - y).mean()

# For power plant data
############################################################################
train_x, train_y, test_x = read_data_power_plant()
print('Train=', train_x.shape)
print('Test=', test_x.shape)

############################################################################
# Decision Tree
############################################################################

parameters = [3, 6, 9, 12, 15]
# Calling grid search function from the helper file 
# This function returns best parameters and list of error for the parameters
# and time it took for model to train with each parameters
k, errorlistDTp, TimeDTp = GridSearch(parameters, "DT", 5, train_x, train_y, True)
# Plot fucntion takes 
plot(TimeDTp, x_axis_label= "Tree Depth")
# for the best k training the full model and saving the kagglizied result in the DT.csv file
# Calling the function Decision Tree from the helper file and training the model
predicted_y = DecisionTree(k, train_x, train_y, test_x)
file_name = '../Predictions/PowerOutput/DT.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)


############################################################################
# KNN
############################################################################
parameters = [3, 5, 10, 20, 25]
k, errorlistKNNp, TimeKNNp = GridSearch(parameters, "KNN", 5, train_x, train_y, True)
predicted_y = knn(k, train_x, train_y, test_x)
file_name = '../Predictions/PowerOutput/KNN.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)


############################################################################
# Ridge
############################################################################


parameters = [10**(-6), 10**(-4), 10**(-2), 1, 10]
k, errorlistRp, TimeRp = GridSearch(parameters, "Ridge", 5, train_x, train_y, True)
predicted_y = ridge(0.01, train_x, train_y, test_x)
file_name = '../Predictions/PowerOutput/ridge.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)


############################################################################
# Lasso
############################################################################
parameters = [10**(-6), 10**(-4), 10**(-2), 1, 10]
k, errorlistLp, TimeLp = GridSearch(parameters, "Lasso", 5, train_x, train_y, True)


predicted_y = lasso(k, train_x, train_y, test_x)

file_name = '../Predictions/PowerOutput/Lasso.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)


"""

Power data ends here

"""


# for in door localization data
###########################################################################

train_x, train_y, test_x = read_data_localization_indoors()
print('Train=', train_x.shape)
print('Test=', test_x.shape)

############################################################################
# Decision Tree
############################################################################

parameters = [20, 25, 30, 35, 40]
k, errorlistDTi, TimeDTi = GridSearch(parameters, "DT", 5, train_x, train_y, True)

plot(TimeDTi, x_axis_label= "Tree Depth")
predicted_y = DecisionTree(k, train_x, train_y, test_x)

file_name = '../Predictions/IndoorLocalization/DT.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)

############################################################################
# KNN
############################################################################

parameters = [3, 5, 10, 20, 25]
k, errorlistKNNi = GridSearch(parameters, "KNN", 5, train_x, train_y, False)
errorlistKNNi
predicted_y = knn(k, train_x, train_y, test_x)

file_name = '../Predictions/IndoorLocalization/knn.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)
############################################################################
# Ridge
############################################################################

parameters = [10**(-4), 10**(-2), 1, 10]
k, errorlistRi, TimeRi = GridSearch(parameters, "Ridge", 5, train_x, train_y, True)
predicted_y = ridge(k, train_x, train_y, test_x)
file_name = '../Predictions/IndoorLocalization/Ridge.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)

############################################################################
# Lasso
############################################################################
parameters = [10**(-4), 10**(-2), 1, 10]
k, errorlistLpi, TimeLpi = GridSearch(parameters, "Lasso", 5, train_x, train_y, True)
predicted_y = lasso(k, train_x, train_y, test_x)
file_name = '../Predictions/IndoorLocalization/lasso.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)




# Experiment for the best model

""" The Best Modal for power plant"""

# Reading data and initializing decision tree regressor. All the experiment test is 
# mentioned in the report.
train_x, train_y, test_x = read_data_power_plant()
clf = DecisionTreeRegressor(max_depth= 9, presort= True, criterion='mae')
"""The Error is increasing with increasing max_ depth and also with decrease in max depth
having max_depth 10, presort true and criterion as mae the error is turn out to be .1595
"""
clf.fit(train_x, train_y)
predicted_y = clf.predict(test_x)
file_name = '../Predictions/PowerOutput/best.csv'
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)



# Test for the best model for the indoor localization dataset
train_x, train_y, test_x = read_data_localization_indoors()
clf = KNeighborsRegressor(n_neighbors= 7, weights= "distance", algorithm= "ball_tree", p= 1)  
# with n = 5 error is 8.87
# with n = 7 error is 8.76
# with n = 10 erorr is 8.74
clf.fit(train_x, train_y)
predicted_y = clf.predict(test_x)
file_name = '../Predictions/IndoorLocalization/best.csv'
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)

