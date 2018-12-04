# functions used in debugging
import numpy as np 

# function that prints number of non-zero columns in a matrix and returns those columns 
def trivialcol_check (mat, tol):
    # trans_mat: transition matrix
    # tol: probability tolerance
    
    counter = 0
    non_triv = []
    for col in range(len(mat[0,:])):
        if all(i <= tol for i in mat[:,col]):
            pass
        else:
            non_triv.append(mat[:,col])
            counter += 1
        
    print ('There are', counter ,'columns out of ', len(mat[0,:]), 'with probabilities >', tol)
    
    return non_triv


        
        
    