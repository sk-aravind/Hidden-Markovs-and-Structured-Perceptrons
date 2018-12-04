from part2 import * 
from preprocess import *
from collections import defaultdict
import numpy as np
import copy    

# function that collates a sorted list of different words, 
# and a dictionary of words as keys and associated indices as values 
def get_words (train):
    # train: training data 

    all_words = [] # list of all words in train data

    # populating all_words list
    for tweet in range(len(train)):
        for word in range(len(train[tweet])):
            all_words.append(train[tweet][word][0]) # list of all words in data

    diff_words = np.unique(all_words) # array of unique words
    diff_words = diff_words.tolist() # converting to list
    diff_words.append('#UNK#') # appending #UNK#
    diff_words = np.asarray(diff_words) # converting back to array
    
    # generating dictionary of words as keys and associated indices as values 
    word_dict = defaultdict(int)
    for i in range(len(diff_words)):
        word_dict[diff_words[i]] = i

    return diff_words, word_dict



# function that gets different sentiments (tags) from a data set
def get_tags (data):
    
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
    # emissions: dictionary of emission probs
    # word_dict: dictionary of different words
    # sents: dictionary of sentiments and associated indices
def em_matrix (emissions, word_dict, sents={}):
    
    em_mat = np.zeros([len(sents.keys()), len(word_dict.keys())]) # init emission matrix
    
    # populating emission parameters
    for tag in sents:
        for word in word_dict:
            em_mat[sents[tag], word_dict[word]] = emissions[(word, tag)] if emissions[(word, tag)] != 0 else 1e-30
    
    # ensuring zeros for emissions from start and stop nodes
    for i in range(len(em_mat[0, :])):
        em_mat[sents['start'], i] = (i == word_dict['~~~~|']) * 1.
        em_mat[sents['stop'], i] = (i == word_dict['|~~~~']) * 1.

    return em_mat



# function that adds start and stop nodes to training set
def mod_train (ptrain):
    
    train = copy.deepcopy(ptrain)
    # inserting start and stop nodes
    for tweet in train:
        tweet.insert(0, ('~~~~|','start'))
        tweet.append(('|~~~~', 'stop'))
        
    return train



# function that adds start and stop nodes to validation set
def mod_test (ptest):
    
    test = copy.deepcopy(ptest)
    # inserting start and stop nodes
    for tweet in test:
        tweet.insert(0, '~~~~|')
        tweet.append('|~~~~')
        
    return test





# function that computes transition parameter matrix
def transition_params (train, Y, sents):
    # train: processed training set of features and labels
    # Y: dictionary with sentiment and counts
    # sents: dictionary with sentiments and associated indices

    q_uv = np.ones([len(sents.keys()), len(sents.keys())]) * 1e-30 # 2D array transitions
    
    # counting u,v transitions for all u,v
    for tweet in range(len(train)):
        for y in range(1, len(train[tweet])):
            
            # comparing data labels with sentiment keys
            label_i = sents[train[tweet][y][1]]
            label_im1 = sents[train[tweet][y-1][1]]

            # filling up transition matrix
            q_uv[label_i, label_im1] += 1/Y[train[tweet][y-1][1]]
    
    # setting all transitions to start and from stop to 0
    for i in range(len(q_uv[:, 0])):
        q_uv[sents['start'], i] = 0.
        q_uv[i, sents['stop']] = 0. 

    return q_uv





# function that runs the viterbi algorithm recursively 
    # emissions: matrix of emission parameters
    # transitions: matrix of transition parameters
    # word_dict: dictionary of words with associated indices
    # line: line of words (tweet)
    # prev_col: scores of previous column 
    # loop_ind: current loop iteration
def viterbi_algo (em_mat, trans_mat, 
                  word_dict, line, prev_scores, 
                  loop_ind=1, ind_list=[]):
    
    # check statements to terminate recursion
    if loop_ind < len(line)-1:
        
        # associated index of current word (checks if word in training set, else #UNK#)
        word_ind = word_dict[line[loop_ind][0]] if line[loop_ind][0] in word_dict else word_dict['#UNK#']

        # populating current score column
        emissions = em_mat[:, word_ind].reshape([len(em_mat[:, 0]),1]) # column of emission matrix
        scores = emissions*trans_mat*np.transpose(prev_scores) # matrix of all possible scores 
        current_scores = np.asarray([np.amax(scores[row,:]) * 1e+3 for row in range(len(prev_scores))]).reshape([len(prev_scores[:,0]), 1])
    
        # appending optimal scores to list
        ind_list.append(np.argmax(current_scores[1:len(current_scores[:,0])-1,0]) + 1)
        
        return viterbi_algo(em_mat, trans_mat, word_dict, line, current_scores, loop_ind + 1, ind_list)
    
    else:
        return ind_list
    
    
    
    

# function that generates emission and transmission matrices, sentiment and word dictionaries
    # lang: language string (e.g. 'EN')
def train_params (lang):
    
    # reading tweets for particular language
    ptrain = data_from_file(lang + '/train') # unmodified
    train = mod_train (ptrain) # modified w/ start and stop states

    sents = get_tags(ptrain) # getting sentiments and associated indices (w/ start and stop)
    Y = get_counts(train)[0] # dictionary of sentiments and their counts
    diff_words = get_words(train)[0] # array of unique words 
    word_dict = get_words(train)[1] # dictionary of unique words and indices

    # emission and transmission parameter matrices
    emission_dict = get_emission2(train, 2) # dictionary with keys as (x, y) and values as emission probabilities
    em_mat = em_matrix(emission_dict, word_dict, sents) # emission matrix
    trans_mat = transition_params(train, Y, sents) # transition matrix
    
    return em_mat, trans_mat, sents, word_dict
        
        
        
        
        
        
    
