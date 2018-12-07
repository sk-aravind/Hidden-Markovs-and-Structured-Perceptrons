# coding: utf-8

# part 4 of ML project
from part4_fun import *
import time


# main code block
if __name__ == '__main__':
    outfile = '/dev.p4.out'
    
    # loop over EN and FR languages
    for lang in ['EN', 'FR']:
        
        print ('============================',lang , '============================')
        
        # ======================================== training ========================================
        k = 1 # regulator for unseen words
        # a, b, tags, words = train_phase2 (lang, k) # getting trained model parameters
        a, b, tags, words = train_phase_2nd_order (lang, k) # getting 2nd order trained model parameters
        # ========================================================================================== 
        
        # ==================================== validation set ====================================
        # A list of list of tuples of size 1. Each list in test is a tweet. 
        ptest = data_from_file(lang + '/dev.in')
        # test is a list of list. Each sublist is an array of words, 1 tweet
        ptest = [[word[0] for word in line] for line in ptest]
        test = mod_test2(ptest) # modified with start and stop words
        # ========================================================================================
        
        start = time.time()
        # ==================================== getting predictions ====================================
        optimal_tags = [] # init list of optimal tag lists corresponding to each tweet 
        
        # loop that runs over all tweets for a given language to predict optimal sentiments
        for tweet in test:
            # optimal_tags.append(Viterbi2 (a, b, tags, words, tweet))
            optimal_tags.append(Viterbi2_alt(a, b, tags, words, tweet))
            
        predictions = [] # init list of predictions
        for tweet in range(len(optimal_tags)):
            predictions.append([(ptest[tweet][i], optimal_tags[tweet][i]) for i in range(len(optimal_tags[tweet]))])
            
        write_predictions(predictions, lang, outfile) # writing results to outfile
        # =============================================================================================
        end = time.time()
        print('time to get predictions for', lang, ': ', end - start)
        print ()        
        pred = get_entities(open(lang+outfile, encoding='utf-8'))
        gold = get_entities(open(lang+'/dev.out', encoding='utf-8'))
        print (lang)
        compare_result(gold, pred)
        print ()
     
    print ('============================ Predictions Complete ============================')




