# functions used in debugging
import numpy as np 

# function that prints number of non-zero columns in transition matrix and returns those columns 
def transition_check (trans_mat, tol):
    # trans_mat: transition matrix
    # tol: probability tolerance
    
    counter = 0
    non_triv_trans = []
    for col in range(len(trans_mat[0,:])):
        if all(i <= tol for i in trans_mat[:,col]):
            pass
        else:
            non_triv_trans.append(trans_mat[:,col])
            counter += 1
        
    print ('There are', counter ,'columns out of ', len(trans_mat[0,:]), 'with probabilities >', tol)
    
    return non_triv_trans


        
        
    