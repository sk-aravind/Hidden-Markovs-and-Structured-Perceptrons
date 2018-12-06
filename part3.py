# coding: utf-8

from preprocess import *
from part2 import * 
from part3_fun import *
from collections import defaultdict
import numpy as np
import time




# main code block
if __name__ == '__main__':
    outfile = '/dev.p3.out'
    
    # loop over languages
    for lang in languages:
        
        
        print ('============================',lang , '============================')
        # ====================================== training ======================================
        a, b, tags, words = train_phase (lang, 1.1)
        # ======================================================================================
        print (tags)
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
        for tweet in test:
            optimal_sentiments.append(Viterbi (a, b, tags, words, tweet))

        predictions = []
        for tweet in range(len(optimal_sentiments)):
            predictions.append([(ptest[tweet][i], optimal_sentiments[tweet][i]) for i in range(len(optimal_sentiments[tweet]))])
        
        write_predictions(predictions, lang, outfile) # writing predictions to outfile
        # =============================================================================================================
        end = time.time()
        print('time to get predictions for', lang, ': ', end - start)
    
    print ('============================ Predictions Complete ============================')
    
    