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
def em_matrix (emissions, diff_words, sents={}):
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
    
    # populating entries with base probabilities
    for row in range(len(em_mat[:, 0])):
        for col in range(len(em_mat[0, :])):
            em_mat[row, col] = em_mat[row, col] if em_mat[row, col] != 0 \
            else (row != 0 and row != len(em_mat[:, 0])-1) * 1e-30  

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
def transition_params (train, Y, sents):
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
            q_uv[label_i, label_im1] += 1/Y[train[tweet][y-1][1]]
    
    # populating entries with base probabilities
    for row in range(len(q_uv[:, 0])):
        for col in range(len(q_uv[0, :])):
            q_uv[row, col] = q_uv[row, col] if q_uv[row, col] != 0 \
            else (row != 0 and col != len(q_uv[0, :])-1) * 1e-30 

    return q_uv





# function that runs the viterbi algorithm recursively 
    # emissions: matrix of emission parameters
    # transitions: matrix of transition parameters
    # word_dict: dictionary of words with associated indices
    # line: line of words (tweet)
    # prev_col: scores of previous column 
    # loop_ind: current loop iteration
    # edge: default index for edge cases
def viterbi_algo (em_mat, trans_mat, 
                  word_dict, line, prev_scores, 
                  loop_ind=1, ind_list=[], edge=1):
    
    # associated index of current word (checks if word in training set, else #UNK#)
    word_ind = word_dict[line[loop_ind][0]] if line[loop_ind][0] in word_dict else word_dict['#UNK#']
    
    emissions = em_mat[:, word_ind].reshape((len(trans_mat[0]),1)) # column of emission matrix
    scores = emissions*trans_mat*np.transpose(prev_scores) # matrix of all scores 
    
    # populating current score column
    current_scores = np.zeros([len(prev_scores), 1]) # init current word layer 
    for row in range(len(prev_scores)): # mult by 100 to prevent underflow
        current_scores[row, 0] = np.amax(scores[row,:]) * 1e+3
    
    # check statements to terminate recursion
    if loop_ind < len(line)-1:
        loop_ind += 1 # setting next iterations index
        ind_list.append(np.argmax(current_scores[1:len(current_scores[:,0])-1, 0]) + 1)
        return viterbi_algo(em_mat, trans_mat, word_dict, line, current_scores, loop_ind, ind_list, edge)
    
    else:
        return ind_list
    
    