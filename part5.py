# coding: utf-8

# part 4 of ML project
from part5_fun import *
import time


# main code block
outfile = '/dev.p5.out'
    
# loop over EN and FR languages
for lang in ['EN', 'FR']:

    print ('============================',lang , '============================')
    # ======================================== training ========================================
    k = 4.1284 if lang == 'EN' else 0 # regulator for unseen words
    a, b = train_phase_2order2 (lang, k) # getting 2nd order trained model parameters
    # ========================================================================================== 

    # ==================================== validation set ====================================
    # A list of list of tuples of size 1. Each list in test is a tweet. 
    test = data_from_file(lang + '/dev.in')
    # test is a list of list. Each sublist is an array of words, 1 tweet
    test = [[word[0] for word in line] for line in test]
    # ========================================================================================

    start = time.time()
    # ============================================ getting predictions ============================================
    predictions = []

    for tweet in test:
        words, tags = list(set(i) for i in zip(*b.keys()))
        prediction = list(zip(tweet, Viterbi_2order2 (a, b, tags, words, tweet)))
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
    
    # for k parameter tuning
    k_vals, F_e, F_et = k_tuning (test, lang, '/dev.p5_ktune.out', 0, 10, 110) 
    print ('k values:')
    print (k_vals, '\n')
    print ('Entity F Scores:')
    print (F_e, '\n')
    print ('Entity Type F Scores:')
    print (F_et, '\n')
    

print ('============================ Predictions Complete ============================')




