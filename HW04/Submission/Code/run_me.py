# Import python modules
import numpy as np
import kaggle
import random
from random import randrange

def p_j_alpha(j, alpha):
    # Function to calculate the probability pr(J|alpha) 
    # takes input j (list) and alpha (int)
    # returns the probability (float)
    X_i = 1.0
    if j[0] !=0:
        X_i = 1e-80
    else:
        for i in range(1, len(j)):
            X_i = X_i*alpha if j[i] == j[i-1] else X_i*(1.0 - alpha)
    return X_i


def p_b_j(b, j, dic):
    # Function to calculate the probability pr(B|J) 
    # takes input b (list), J(list) and dic (dictionary of probability)
    # returns the X_i probability (float)
    X_i = 1.0
    for i in range(len(b)):
        X_i *= dic[j[i], b[i]]
    return X_i


def p_alpha(alpha):
    # Function to calculate alpha 
    # takes alpha (float ) as input 
    if alpha >= 0.0 and alpha <= 1.0:
        return 1.0
    else:
        return 1e-80
    
    
def p_alpha_j_b(b, j, alpha):
    # Function to calculate pr(alpha,j,B)
    # takes alpha(int), j(list) and B(list)
    # returns pr(alpha,j,B) (float) 
    return p_alpha(alpha)*p_b_j(b, j, dic)*p_j_alpha(j, alpha)

def random_j(j):
    # Function to flip certain index of j
    length  = len(j)
    index = np.random.randint(0, length)
    j_new = np.copy(j)
    j_new[index] = int(not j_new[index])
    return j_new

def p_j_alpha_b(b, alpha, iteration):
    # Implementation of Metropolis Hastings algorithm 
    #to draw samples from P (J|α, B) and then 
    # calculate hen calculate the mean value of J in those samples.
    # Takes b (list), alpha (float) and iteration (int)
    j = np.array([0 for i in range(len(b))])
    j_mean = np.array([0 for i in range(len(b))])
    for i in range(iteration):
        j_new = np.array(random_j(j))
        acceprance_ratio = p_alpha_j_b(b, j_new, alpha)*1.0/p_alpha_j_b(b, j, alpha)*1.0
        if random.random() <= acceprance_ratio:
            j = j_new
        j_mean += j
    return j_mean/(1.0*iteration)
 
def random_alpha(alpha):
    # Function to randomly calculate random alpha
    # input alpha (flaot)
    # return random alpha value between 0.0 and 1.0
    return np.random.rand()

def p_alpha_given_j_b(j, b, iteration):
    #function for Metropolis Hastings algorithm 
    #to draw samples from P (α|J, B) and
    #then calculate the mean value of α in those samples
    #Takes input j (list), b (list) and iteration (int)
    # returns probability (float)
    alpha_mean =[]
    alpha = random.random()
    for i in range(int(iteration)):
        alpha_new = random_alpha(alpha)
        #print(p_alpha_j_b(b, j, alpha))
        acceprance_ratio = p_alpha_j_b(b, j, alpha_new)/p_alpha_j_b(b, j, alpha)
        if random.random() <= acceprance_ratio:
            alpha = alpha_new
        alpha_mean.append(alpha)
    return sum(alpha_mean)/(1.0*iteration)

def proposal_alpha_j(j, alpha):
    # returns proposal alpha and j 
    # input j (list) and alpha (float)
    # return flipped j (list) and random alpha (alpha)
    return random_j(j), random_alpha(alpha)
    

def p_new_alpha_new_j_b(b, iteration):
    #function for Metropolis Hastings algorithm
    #to draw samples from P (α, J|B) and
    #then calculate the mean values of α and J in those Samples.
    # Input b (list) and iteration (int)
    # outputs P (J|α, B), P (α|J, B)
    alpha_mean = 0.0
    alpha = random.random()
    j = np.array([0 for i in range(len(b))])
    j_mean = np.array([0 for i in range(len(b))])
    for i in range(int(iteration)):
        alpha_new = random_alpha(alpha)
        j_new = np.array(random_j(j))
        acceprance_ratio = p_alpha_j_b(b, j_new, alpha_new)/p_alpha_j_b(b, j, alpha)
        if random.random() <= acceprance_ratio:
            alpha = alpha_new
            j = j_new
        alpha_mean += alpha
        j_mean += j
    return j_mean/(1.0*iteration), alpha_mean/(1.0*iteration)

def p_n_1_given_alpha(j_n, alpha):
    # returns input P (B n+1 |J n , α)
    # Input j_n and alpha both float
    # returns probability float
    if j_n == 0:
        return alpha*0.8 + (1 - alpha)*0.1
    if j_n == 1:
        return alpha*0.1 + (1 - alpha)*0.8


def p_j_alpha_given_b(b, iteration):
    #function to draw samples using Metropolis Hastings algorithm 
    #from P (J, α|B) and then perform an exact computation 
    #of P (B n+1 |J n , α) for each of those samples
    # Takes b (list) and iteration (int)
    # Returns probability
    j_proposed, alpha_proposed = p_new_alpha_new_j_b(b, iteration)
    j = 1*(j_proposed > 0.5)
    return p_n_1_given_alpha(j[-1], alpha_proposed)
    
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
    dic = {(0 ,0) : 0.20, (0, 1): 0.80, (1, 0): 0.90, (1,1) : 0.10}
    test_case = [[[0,1,1,0,1], [1,0,0,1,1]], [[0,1,0,0,1], [0,0,1,0,1]], 
                 [[0,1,1,0,0,1], [1,0,1,1,1,0]], [[1,1,0,0,1,1], [0,1,1,0,1,1]]]
    for i,test in enumerate(test_case):
        j, b = test[0],test[1] 
        print("P_b_j for test case " + str(i + 1) + " " + str(j)  + " "+ str(b)  +": " + str(p_b_j(b, j, dic)))
    
    # Working on the Question 1 d
    print("\n*************************Solution for question 1 d *******************************")
    #dic = {(0 ,0) : 0.20, (0, 1): 0.90, (1, 0): 0.80, (1,1) : .10}
    test_case = [[[0,1,1,0,1], [1,0,0,1,1], 0.75], [[0,1,0,0,1], [0,0,1,0,1], 0.3],
                 [[0,0,0,0,0,1], [0,1,1,1,0,1], 0.63], [[0,0,1,0,0,1,1], [1,1,0,0,1,1,1], 0.23]]
    for i,test in enumerate(test_case):
        j, b, alpha = test[0],test[1], test[2] 
        print("P_alpha_j_b for test case " + str(i + 1) + ", " + str(j) + " " + str(b) + ", " +str(alpha) + ": " + str(p_alpha_j_b(b, j, alpha)))
    
    # Working on the Question 1 f
    print("\n*************************Solution for question 1 f *******************************")
    #dic = {(0 ,0) : 0.20, (0, 1): 0.90, (1, 0): 0.80, (1,1) : .10}
    iteration = 10000
    test_case = [[[1,0,0,1,1], 0.5], [[1,0,0,0,1,0,1,1], 0.11], [[1,0,0,1,1,0,0], 0.75]]
    for i,test in enumerate(test_case):
        b, alpha = test[0],test[1] 
        print("p_j_alpha_b for test case " + str(i + 1) +  ": " + str(p_j_alpha_b(b, alpha, iteration)))
        
    print("\n*************************Solution for question 1 h *******************************")
    #dic = {(0 ,0) : 0.20, (0, 1): 0.90, (1, 0): 0.80, (1,1) : .10}
    test_cases = [[[0,1,0,1,0], [1,0,1,0,1]], [[0,0,0,0,0], [1,1,1,1,1]],
                  [[0,1,1,1,1,1,1,0], [1,0,0,1,1,0,0,1]], 
                  [[0,1,1,0,1,0], [1,0,0,1,1,1]]]
    iteration = 10000.0
    for i, test_case in enumerate(test_cases):
        j, b = test_case[0], test_case[1]
        print("p_alpha_given_j_b for test case " + str(i + 1) +  ": " + str(p_alpha_given_j_b(j, b, iteration)))
        
    print("\n*************************Solution for question 1 j *******************************")
    b = np.array([1,1,0,1,1,0,0,0])
    print("p_new_alpha_new_j_b: " + str(p_new_alpha_new_j_b(b, iteration)))
    
    print("\n*************************Solution for question 1 k *******************************")
    test_cases = [[0.33456, 0.0], [0.5019, 1.0], [0.33456, 0.0], [0.5019, 1.0]]
    for i, test_case in enumerate(test_cases):
        alpha, j = test_case[0], test_case[1]
        print("p_n_1_given_alpha for test case " + str(i + 1) + " : " + str(p_n_1_given_alpha(j, alpha)))
    
    print("\n*************************Solution for question 1 l *******************************")
    iteration = 10000
    for i, b in enumerate([[0, 0, 1], [0, 1, 0, 1, 0, 1], [0, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]):
        print("p_j_alpha_given_b for test case " +str(i + 1) + " : " + str(p_j_alpha_given_b(b, iteration)))
    
    print("\n*************************Solution for question 1 m *******************************")
    prediction_prob = [] 
    lengths = [10, 15, 20, 25]
    for l in lengths:
        B_array = np.loadtxt('../../Data/B_sequences_%s.txt' %(l), delimiter=',', dtype=float)
        for b in B_array:
            #print(p_j_alpha_given_b(b, iteration))
            prediction_prob.append(p_j_alpha_given_b(b, iteration))
            print('Prob of next entry in ', b, 'is black is', prediction_prob[-1])

    # Output file location
    file_name = "../Predictions/best.csv"
	# Writing output in Kaggle format
    print('Writing output to ', file_name)
    kaggle.kaggleize(np.array(prediction_prob), file_name)