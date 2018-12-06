# coding: utf-8

# part 4 of ML project
from part4_fun import *
from check_fun import *


# main code block
if __name__ == '__main__':
    outfile = '/dev.p4.out'
    
    # loop over EN and FR languages
    for lang in ['EN', 'FR']:
        
        print ('============================',lang , '============================')
        
        # ======================================== training ========================================
        a, b, tags, words = train_phase2 (lang, 1.1)
        # ========================================================================================== 
        
        # ==================================== validation set ====================================
        # A list of list of tuples of size 1. Each list in test is a tweet. 
        ptest = data_from_file(lang + '/dev.in')
        # test is a list of list. Each sublist is an array of words, 1 tweet
        ptest = [[word[0] for word in line] for line in ptest]
        test = mod_test2(ptest) # modified with start and stop words
        # ========================================================================================
        
        # ==================================== getting predictions ====================================
        optimal_tags = [] # init list of optimal tag lists corresponding to each tweet 
        
        # loop that runs over all tweets for a given language to predict optimal sentiments
        for tweet in test:
            optimal_tags.append(Viterbi2 (a, b, tags, words, tweet))
            print (tweet)
        
        predictions = [] # init list of predictions
        for tweet in range(len(optimal_tags)):
            predictions.append([(ptest[tweet][i], optimal_tags[tweet][i]) for i in range(len(optimal_tags[tweet]))])
        
        # write_predictions(predictions, lang, outfile) # writing results to outfile
        # =============================================================================================
        
    
    print ('============================ Predictions Complete ============================')



