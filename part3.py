# coding: utf-8

from part2 import * 
from part3_fun import *
from check_fun import *
from preprocess import *
from collections import defaultdict
import numpy as np
import time



# main code block
if __name__ == '__main__':
    outfile = '/dev.p3.out'
    
#     languages = ['FR']
    # loop over languages
    for lang in languages:
        
        
        print ('============================',lang , '============================')
        # ====================================== training ======================================
        em_mat, trans_mat, sents, word_dict = train_params (lang)
        # ======================================================================================
                
        # ========================================= validation set =========================================
        # A list of list of tuples of size 1. Each list in test is a tweet. 
        ptest = data_from_file(lang + '/dev.in')
        # test is a list of list. Each sublist is an array of words, 1 tweet
        ptest = [[word[0] for word in line] for line in ptest]
        test = mod_test(ptest) # modified with start and stop words
        # ==================================================================================================
        
        start = time.time()
        # ============================================ getting predictions ============================================
        # initializing list of optimal sentiment lists corresponding to each tweet 
        optimal_sentiments = []
        
        # loop that runs over all tweets for a given language to predict optimal sentiments
        for tweet in range(len(test)):
            
            # running Viterbi algorithm
            base_scores = np.ones([len(sents.keys()),1]) # initializing base case scores
            opt_ind_list = viterbi_algo (em_mat, trans_mat, word_dict, test[tweet], base_scores, 1, []) 
            
            # generating list of optimal sentiments for a given sentence
            inv_sents = dict (zip(sents.values(), sents.keys())) # swapping keys and values
            opt_sents = [inv_sents[opt_ind_list[i]] for i in range(len(opt_ind_list))]

            optimal_sentiments.append(opt_sents) # populating parent optimal sentiment list
            
            # printing iteration checks
            if (tweet % 100 == 0):
                print (tweet, ' out of ', len(test), ' tweets have been predicted.')
        
        predictions = []
        for tweet in range(len(optimal_sentiments)):
            predictions.append([(ptest[tweet][i], optimal_sentiments[tweet][i]) for i in range(len(optimal_sentiments[tweet]))])
        
        write_predictions(predictions, lang, outfile) # writing predictions to outfile
        # =============================================================================================================
        end = time.time()
        print('time to run the Viterbi algorithm for', lang, ': ', end - start)
    
    print ('============================ Predictions Complete ============================')
    
    