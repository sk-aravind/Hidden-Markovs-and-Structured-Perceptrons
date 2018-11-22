# coding: utf-8

from part2 import * 
from preprocess import *
from collections import defaultdict
import numpy as np
import copy


# function that collates a sorted list of different words, 
# and a dictionary of words as keys and associated indices as values 
def get_words(train):
    # train: training data 

    all_words = [] # list of all words in train data

    # populating all_words list
    for tweet in range(len(train)):
        for word in range(len(train[tweet])):
            all_words.append(train[tweet][word][0]) # list of all words in data


    diff_words = np.unique(all_words) # array of unique words

    # generating dictionary of words as keys and associated indices as values 
    word_dict = defaultdict(int)
    for i in range(len(diff_words)):
        word_dict[diff_words[i]] = i

    return diff_words, word_dict

# function that gets different sentiments (tags) from a data set
def get_tags(data):
    
    tags = {}
    Y = get_counts(data)[0] 
    
    len_tweet = 0
    for sent in range(len(Y.keys())):
        if (list(Y.keys())[sent] != 'start') and (list(Y.keys())[sent] != 'stop'):
            tags[list(Y.keys())[sent]] = sent+1
            len_tweet += 1
        else:
            pass
    
    tags['stop'] = len_tweet + 1
    tags['start'] = 0
    
    return tags
    
    

# function that converts emission dictionary into emission matrix
def em_matrix(emissions, diff_words, sents={}):
    # emissions: dictionary of emission probs
    # diff_words: diff_words: array of words arranged with associated indices
    # sents: dictionary of sentiments and associated indices

    em_mat = np.zeros([len(sents.keys()), len(diff_words)]) # emission matrix
    inv_sents = dict (zip(sents.values(),sents.keys())) # swapping keys and values

    # populating emission matrix
    for row in range(len(inv_sents.keys())):
        for col in range(len(diff_words)):
            if (diff_words[col], inv_sents[row]) in emissions.keys():
                em_mat[row, col] = emissions[(diff_words[col], inv_sents[row])]
            else:
                pass

    return em_mat


# function that adds start and stop nodes to training set
def mod_train (ptrain):
    
    train = copy.deepcopy(ptrain)
    # inserting start and stop nodes
    for tweet in train:
        tweet.insert(0, ('////','start'))
        tweet.append(('\\\\', 'stop'))
        
    return train


# function that adds start and stop nodes to validation set
def mod_test (ptest):
    
    test = copy.deepcopy(ptest)
    # inserting start and stop nodes
    for tweet in test:
        tweet.insert(0, '////')
        tweet.append('\\\\')
        
    return test


# function that computes transition parameter matrix
def transition_params(train, Y, sents):
    # train: processed training set of features and labels
    # Y: dictionary with sentiment and counts
    # sents: dictionary with sentiments and associated indices

    q_uv = np.zeros([len(Y.keys()), len(Y.keys())]) # 2D array transitions
    
    # counting u,v transitions for all u,v
    for tweet in range(len(train)):
        for y in range(1, len(train[tweet])):
            
            # comparing data labels with sentiment keys
            label_i = sents[train[tweet][y][1]]
            label_im1 = sents[train[tweet][y-1][1]]

            # filling up transition matrix
            q_uv[label_im1, label_i] += 1/Y[train[tweet][y-1][1]]

    return q_uv


# In[97]:



# function that runs the viterbi algorithm recursively 
def viterbi_algo (em_mat, trans_mat, 
                  word_dict, line, prev_scores, 
                  loop_ind=1, ind_list=[]):
    # emissions: matrix of emission parameters
    # transitions: matrix of transition parameters
    # word_dict: dictionary of words with associated indices
    # line: line of words (tweet)
    # prev_col: scores of previous column 
    # loop_ind: current loop iteration

    word_ind = word_dict[line[loop_ind][0]] # associated index of current word
    emissions = em_mat[:, word_ind].reshape((len(trans_mat[0]),1)) 
    scores = emissions*trans_mat*np.transpose(prev_scores) # matrix of all scores 
    
    current_scores = np.zeros([len(prev_scores), 1]) # init current word layer 
    # loop to fill current score column
    for row in range(len(prev_scores)):   
        current_scores[row, 0] = np.amax(scores[row,:])

    # check statements to terminate recursion
    if loop_ind < len(line)-1:
        
        loop_ind += 1 # setting next iterations index
        ind_list.append(np.argmax(current_scores)) # storing optimal path node indices  
        return viterbi_algo(em_mat, trans_mat, word_dict, line, current_scores, loop_ind, ind_list)
    
    else:
        return ind_list

    
    

# main code block
if __name__ == '__main__':
    outfile = '/dev.p3.out'
    
    # initializing dictionary where values are the nested list of optimal sentiments
    opt_sent_dict = {'EN':[], 'CN':[], 'FR':[], 'SG:':[]} 
    
    # loop over languages
    for lang in languages:
        
        # reading tweets for particular language
        ptrain = data_from_file(lang + '/train') # unmodified
        train = mod_train (ptrain) # modified w/ start and stop states

        # getting sentiments and associated indices (w/ start and stop)
        sents = get_tags(ptrain) 
        print (sents)

        Y = get_counts(train)[0] # dictionary of sentiments and their counts
        diff_words = get_words(train)[0] # array of unique words 
        word_dict = get_words(train)[1] # dictionary of unique words and indices

        # emission and transmission parameter matrices
        emission_dict = get_emission(train) # dictionary with keys as (x, y) and values as emission probabilities
        em_mat = em_matrix(emission_dict, diff_words, sents) # emission matrix
        trans_mat = transition_params(train, Y, sents) # transition matrix
        
        # A list of list of tuples of size 1. Each list in test is a tweet. 
        ptest = data_from_file(lang + '/dev.in')
        # test is a list of list. Each sublist is an array of words, 1 tweet
        ptest = [[word[0] for word in line] for line in ptest]
        test = mod_test(ptest) # modified with start and stop words
        
        # initializing list of optimal sentiment lists corresponding to each tweet 
        optimal_sentiments = []
        
        print ('============================',lang , '============================')
        # loop that runs over all tweets for a given language to predict optimal sentiments
        for tweet in range(len(test)):
            
            base_scores = np.ones([len(sents.keys()),1]) # initializing base case scores
            opt_ind_list = viterbi_algo (em_mat, trans_mat, word_dict, test[tweet], base_scores, 1, []) # running Viterbi
            
            # generating list of optimal sentiments for a given sentence
            inv_sents = dict (zip(sents.values(), sents.keys())) # swapping keys and values
            opt_sents = [inv_sents[opt_ind_list[i]] for i in range(len(opt_ind_list))]

            optimal_sentiments.append(opt_sents) # populating parent optimal sentiment list
            
            # printing iteration checks
            if (tweet % 100 == 0):
                print (tweet, ' out of ', len(test), ' tweets have been predicted.')
        
        opt_sent_dict[lang] = optimal_sentiments # populating optimal sentiment dictionary 
        
        predictions = []
        for tweet in range(len(optimal_sentiments)):
            predictions.append([(ptest[tweet][i], optimal_sentiments[tweet][i]) for i in range(len(optimal_sentiments[tweet]))])
        
        write_predictions(predictions, lang, outfile)
        
    
    print ('============================ Predictions Complete ============================')



