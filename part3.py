# coding: utf-8

from preprocess import *
from part2 import * 
from part3_fun import *
from collections import defaultdict
import numpy as np
import time




# main code block
outfile = '/dev.p3.out'
# languages = ['EN', 'FR']
# loop over languages
for lang in languages:

    print ('============================',lang , '============================')
    # ====================================== training ======================================
    k = 1
    a, b = train_phase (lang, k)
    # ======================================================================================
    
    # ========================================= validation set =========================================
    # A list of list of tuples of size 1. Each list in test is a tweet. 
    test = data_from_file(lang + '/dev.in')
    # test is a list of list. Each sublist is an array of words, 1 tweet
    test = [[word[0] for word in line] for line in test]
    # ==================================================================================================

    start = time.time()
    # ============================================ getting predictions ============================================
    predictions = []
    for tweet in test:
        words, tags = tuple(set(i) for i in zip(*b.keys()))
        prediction = list(zip(tweet, Viterbi(a, b, tags, words, tweet)))
        predictions.append(prediction)

    write_predictions(predictions, lang, outfile) # writing predictions to outfile
    # =============================================================================================================
    end = time.time()
    print('time to get predictions for', lang, ': ', end - start)
    
    print ()
    
    pred = get_entities(open(lang+outfile, encoding='utf-8'))
    gold = get_entities(open(lang+'/dev.out', encoding='utf-8'))
    print (lang)
    compare_result(gold, pred)
    print ()

print ('============================ Predictions Complete ============================')
    
    