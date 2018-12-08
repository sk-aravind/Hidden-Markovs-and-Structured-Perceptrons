# coding: utf-8

# part 4 of ML project
from part5_fun import *
import time


# main code block
outfile = '/dev.p5.out'

# loop over EN and FR languages
for lang in ['EN', 'FR']:

    print ('============================',lang , '============================')
    
    k_vals = [0] # np.linspace(0,10,110).tolist()
    F0 = [] # F score for entities
    F1 = [] # F score for entity types
    
    for k in k_vals:
        # ======================================== training ========================================
        # k = 1 # regulator for unseen words
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
    
#         F0.append(F_scores(gold, pred)[0]) # F scores for entities
#         F1.append(F_scores(gold, pred)[1]) # F scores for entity types
        
#     print ('k values:', k_vals)
#     print ()
#     print ('entity F scores: ', F0)
#     print ()
#     print ('entity type F scores: ', F1)

print ('============================ Predictions Complete ============================')




