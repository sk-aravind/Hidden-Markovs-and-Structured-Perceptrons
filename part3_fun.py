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
    for tweet in train:
        for y in tweet:
            all_words.append(y[0]) # list of all words in data

    diff_words = np.unique(all_words).tolist() # list of unique words
    diff_words.append('#UNK#') # appending #UNK#
    
    # generating dictionary of words as keys and associated indices as values 
    word_dict = defaultdict(int)
    for i in range(len(diff_words)):
        word_dict[diff_words[i]] = i

    return np.asarray(diff_words), word_dict



# function that gets different tags (sentiments) from a data set
def get_tags (data):
    
    tags = defaultdict(int)
    Y = get_counts(data)[0] 
    
    len_tweet = 0
    for sent in Y:
        if (Y[sent] != 'start') and (Y[sent] != 'stop'):
            len_tweet += 1
            tags[sent] = len_tweet
        else: pass
    
    tags['stop'] = len_tweet + 1
    tags['start'] = 0
    
    return tags



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



# function that computes transition parameters 
    # train: processed training set of features and labels
    # Y: dictionary with sentiment and counts
    # sents: dictionary with sentiments and associated indices
def transition_dict (train, Y):
    
    a_uv = defaultdict(float)
    
    # counting u,v transitions for all u,v
    for tweet in train:
        for y_i in range(1, len(tweet)):
    
            # filling up transition matrix
            a_uv[(tweet[y_i - 1][1], tweet[y_i][1])] += 1/Y[tweet[y_i-1][1]]

    return a_uv

    
    
# function that runs the viterbi algorithm for each tweet
    # a: transition dictionary
    # b: emission dictionary
    # tags: dictionary of tags and indices
    # words: dictionary of words
    # tweet: tweet from data
def Viterbi (a, b, tags, words, tweet):
 
    optimal_tags = [] # optimal tags for given tweet
    
    pi = defaultdict(float) # initializing score dictionary
    pi[(0, 'start')] = 1. # base case
    
    for j in range(1,len(tweet)): # loop over all words in tweet
        
        u_opt, pi_j_max = ['O', 0.] # default tag and score
        x_j = tweet[j] if tweet[j] in words else '#UNK#' # j-th word in tweet

        for u in tags: # loop over all possible tags
            
            pi[(j, u)] = max([pi[(j-1, v)] * a[(v,u)] * b[(x_j, u)] for v in tags]) # max score finding
            u_opt, pi_j_max = [u, pi[(j, u)]] if pi[(j, u)] > pi_j_max else [u_opt, pi_j_max] # updating opt tag for x_j
            
        optimal_tags.append(u_opt) # appending optimal sentiments
    
    return optimal_tags[:-1]
        
    
    

# function that generates emission and transmission dicts, sentiment and word dictionaries
    # lang: language string (e.g. 'EN')
    # k: regulator for unseen words
def train_phase (lang, k):
    
    # reading tweets for particular language
    ptrain = data_from_file(lang + '/train') # unmodified
    train = mod_train (ptrain) # modified w/ start and stop states

    sents = get_tags(ptrain) # getting sentiments and associated indices (w/ start and stop)
    Y = get_counts(train)[0] # dictionary of sentiments and their counts
    word_dict = get_words(train)[1] # dictionary of unique words and indices

    # emission and transmission parameter matrices
    em_dict = get_emission2 (train, k) # dictionary with keys as (x, y) and values as emission probabilities
    trans_dict = transition_dict (train, Y) # transition matrix
    
    return trans_dict, em_dict, sents, word_dict



