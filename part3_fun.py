from part2 import * 
from preprocess import *
from collections import defaultdict
from functools import lru_cache
import numpy as np
import copy   


# function that adds start and stop nodes to training set
def mod_train (ptrain):
    
    train = copy.deepcopy(ptrain)
    # inserting start and stop nodes
    for tweet in train:
        tweet.insert(0, ('~~~~|','start'))
        tweet.append(('|~~~~', 'stop'))
        
    return train



# function that computes transition parameters 
    # train: processed training set of features and labels
    # Y: dictionary with tags and counts
    # sents: dictionary with tags and associated indices
def transition_dict (train, Y):
    
    a_uv = defaultdict(float)
    
    # counting u,v transitions for all u,v
    for tweet in train:
        for y_i in range(1, len(tweet)):
    
            # filling up transition matrix
            a_uv[(tweet[y_i - 1][1], tweet[y_i][1])] += 1/Y[tweet[y_i-1][1]]

    return a_uv


# # function that runs the viterbi algorithm for each tweet
#     # a: transition dictionary
#     # b: emission dictionary
#     # tags: dictionary of tags and indices
#     # words: dictionary of words
#     # tweet: tweet from data
def Viterbi(a, b, tags, words, tweet):
    
    pi = defaultdict(float) # init dictionary
    
    for j in range(len(tweet)): # loops over words in tweet, O(n) complexity
        x_j = tweet[j] if tweet[j] in words else '#UNK#' # word in tweet
        
        for u in tags: # loops over possible tags, O(T) complexity
            if j == 0:
                pi[(j, u)] = a[('start', u)] * b[(x_j, u)] # base case

            elif j > 0:
                pi[(j, u)] = max([pi[(j-1, v)] * a[(v, u)] * b[(x_j, u)] \
                                  for v in tags]) # finding max score for u, O(T) complexity
    
    n = len(tweet) # length of tweet
    pi[(n, 'stop')] = max([pi[n-1, v] * a[(v, 'stop')] for v in tags]) # stop state score
    
    # function that runs each backtracking iteration
    def backtrack(j, u):
        scores = {v: pi[(j-1, v)] * a[(v, u)] for v in tags} # scores for word j
        best_tag = max(scores, key=lambda key: scores[key]) \
                    if max(scores.values()) > 0 else 'O' # tag with the highest score
        return best_tag

    reverse_tags = [] # init list 
    u = 'stop' # setting first tag for back tracking
    
    # bactracking for Viterbi algorithm
    for j in range(len(tweet), 0, -1):
        v = backtrack(j, u)
        u = v  # moving to previous word
        reverse_tags.append(u)

    return reverse_tags[::-1]



# function that generates emission and transmission dicts, sentiment and word dictionaries
    # lang: language string (e.g. 'EN')
    # k: regulator for unseen words
def train_phase (lang, k):
    
    # reading tweets for particular language
    ptrain = data_from_file(lang + '/train') # unmodified
    train = mod_train (ptrain) # modified w/ start and stop states

    Y = get_count_y (train) # dictionary of sentiments and their counts
    
    # emission and transmission parameter dictionaries
    em_dict = get_emission2 (ptrain, k) # dictionary with keys as (x, y) and values as emission probabilities
    trans_dict = transition_dict (train, Y) # transition dictionary
    
    return trans_dict, em_dict



