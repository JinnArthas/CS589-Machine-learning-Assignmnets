# Import python modules
import numpy as np
import kaggle


def p_j_alpha(j, alpha):
    # Function to calculate the probability pr(J|alpha) 
    # takes input j (list) and alpha (int)
    # returns the probability (float)
    X_i = 1 if j[0] == 0 else 1e-100 
    X_i = X_i*alpha if (j[0] == j[1]) else X_i*(1- alpha)
    for i in range(1, len(j) -1):
        X_i = X_i*alpha if j[i] == j[i+1] else X_i*(1 - alpha)
    return X_i


def p_b_j(b, j, dic):
    # Function to calculate the probability pr(B|J) 
    # takes input b (list), J(list) and dic (dictionary of probability)
    # returns the X_i probability (float)
    X_i = 1
    for i in range(len(b)):
        X_i *= dic[b[i], j[i]]
    return X_i


def p_alpha(alpha):
    # Function to calculate alpha 
    # takes alpha (float ) as input 
    # returns 1 if 0 <= alpha <= 1, 0 otherwise
    if 0 <= alpha <= 1:
        return 1
    else:
        return 0
    
    
def p_alpha_j_b(b, j, alpha):
    # Function to calculate pr(alpha,j,B)
    # takes alpha(int), j(list) and B(list)
    # returns pr(alpha,j,B) (int)
    return p_alpha(alpha)*p_b_j(b, j, dic)*p_j_alpha(j, alpha)
################################################
if __name__ == '__main__':

	
    
    # Working on the Question1 a
    print("\n*************************Solution for question 1 a *******************************")
    test_case = [[[0,1,1,0,1], 0.75], [[0,0,1,0,1], 0.2], [[1,1,0,1,0,1], 0.2], [[0,1,0,1,0,0], 0.2]]
    for i,test in enumerate(test_case):
        j, alpha = test[0],test[1] 
        print("P_J_alpha for test case " + str(i + 1) + " " + str(j)  + " "+ str(alpha)  +": " + str(p_j_alpha(j, alpha)))
    
    # Working on the Question 1 b
    print("\n*************************Solution for question 1 b *******************************")
    dic = {(0 ,0) : 0.20, (0, 1): 0.90, (1, 0): 0.80, (1,1) : .10}
    test_case = [[[0,1,1,0,1], [1,0,0,1,1]], [[0,1,0,0,1], [0,0,1,0,1]], 
                 [[0,1,1,0,0,1], [1,0,1,1,1,0]], [[1,1,0,0,1,1], [0,1,1,0,1,1]]]
    for i,test in enumerate(test_case):
        j, b = test[0],test[1] 
        print("P_b_j for test case " + str(i + 1) + " " + str(j)  + " "+ str(b)  +": " + str(p_b_j(b, j, dic)))
    
    # Working on the Question 1 d
    print("\n*************************Solution for question 1 d *******************************")
    dic = {(0 ,0) : 0.20, (0, 1): 0.90, (1, 0): 0.80, (1,1) : .10}
    test_case = [[[0,1,1,0,1], [1,0,0,1,1], 0.75], [[0,1,0,0,1], [0,0,1,0,1], 0.3],
                 [[0,0,0,0,0,1], [0,1,1,1,0,1], 0.63], [[0,0,1,0,0,1,1], [1,1,0,0,1,1,1], 0.23]]
    for i,test in enumerate(test_case):
        j, b, alpha = test[0],test[1], test[2] 
        print("P_alpha_j_b for test case " + str(i + 1) + ", " + str(j) + " " + str(b) + ", " +str(alpha) + ": " + str(p_alpha_j_b(b, j, alpha)))
        
        
        
        
        
    """
    print('1a through 1l computation goes here ...')
    
	################################################
    #
	lengths = [10, 15, 20, 25]
	prediction_prob = list()
	for l in lengths:
		B_array = np.loadtxt('../../Data/B_sequences_%s.txt' % (l), delimiter=',', dtype=float)
		for b in np.arange(B_array.shape[0]):
			prediction_prob.append(np.random.rand(1))
			print('Prob of next entry in ', B_array[b, :], 'is black is', prediction_prob[-1])
    
	"""
    """# Output file location
	file_name = '../Predictions/best.csv'

	# Writing output in Kaggle format
	print('Writing output to ', file_name)
	kaggle.kaggleize(np.array(prediction_prob), file_name)"""
