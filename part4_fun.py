# functions for second order Viterbi algorithm (ML project part 4)

from part3_fun import *
from copy import deepcopy


# function that gets different sentiments/tags from a data set (adds the start and stop tags)
    # data: unmodified training data (ptrain)
def get_tags2(data):
    
    # generating dictionary of single tags
    tags = defaultdict(int)
    Y = get_counts(data)[0] 
    
    len_tweet = 0
    for sent in Y:
        if (Y[sent] != 'start0') and (Y[sent] != 'start1') and (Y[sent] != 'stop0') and (Y[sent] != 'stop1'):
            len_tweet += 1
            tags[sent] = len_tweet + 1
        else: pass
        
    # adding start and stop labels
    tags['stop0'] = len_tweet + 2 
    tags['stop1'] = len_tweet + 3
    tags['start0'] = 0
    tags['start1'] = 1
    
    return tags



# function that adds start and stop nodes to training data set
    # unmodified training data
def mod_train2 (ptrain):
    
    train = deepcopy(ptrain)
    # inserting start and stop nodes
    for tweet in train:
        tweet.insert(0, ('~~~~|_','start1'))
        tweet.insert(0, ('~~~~|','start0'))
        tweet.append(('|~~~~', 'stop0'))
        tweet.append(('_|~~~~', 'stop1'))
        
    return train


# function that adds start and stop words to validation/test data set (no labels)
    # ptest: unmodified testing data 
def mod_test2 (ptest):
    
    test = deepcopy(ptest)
    # inserting start and stop nodes
    for tweet in test:
        tweet.insert(0, '~~~~|_')
        tweet.insert(0, '~~~~|')
        tweet.append('|~~~~')
        tweet.append('_|~~~~')
        
    return test



# function that computes transition parameters 
    # train: processed training set of features and labels
    # YY: dictionary with tag pairs and counts
    # sents: dictionary with sentiments and associated indices
    # sent_pairs: dictionary with sentiment pairs and associated indices
def transition_dict2 (train, YY):
    
    a_uv = defaultdict(float)
    
    # counting u,v transitions for all u,v
    for tweet in train:
        for y_i in range(2, len(tweet)):
            
            # filling up transition matrix
            a_uv[((tweet[y_i - 2][1], tweet[y_i - 1][1]), tweet[y_i][1])] += 1/YY[(tweet[y_i-2][1], tweet[y_i - 1][1])]

    return a_uv    




# function that runs the viterbi algorithm for each tweet
    # a: transition dictionary
    # b: emission dictionary
    # tags: dictionary of tags and indices
    # words: dictionary of words
    # tweet: tweet from data
def Viterbi2 (a, b, tags, words, tweet):
 
    optimal_tags = [] # optimal tags for given tweet
    
    pi = defaultdict(float) # initializing score dictionary
    pi[(0, 'start0')] = 1. # base case 0
    pi[(1, 'start1')] = 1. # base case 1
    
    for j in range(2,len(tweet)): # loop over all words in tweet
        
        u_opt, pi_j_max = ['O', 0.] # default tag and score
        x_jm1 = tweet[j-1] if tweet[j-1] in words else '#UNK#' # j-th word in tweet
        x_j = tweet[j] if tweet[j] in words else '#UNK#' # j-th word in tweet
        
        
        for u in tags: # loop over all possible tags
            
            pi_ju = np.zeros([len(tags), len(tags)]) # matrix of possible scorings 
            for v0 in tags: # j-2 tag
                for v1 in tags: # j-1 tag
                    pi_ju[tags[v0], tags[v1]] = pi[(j-1, v1)]*pi[(j-2, v0)] * a[(v0, v1)]*a[(v1, u)] * b[(x_jm1, v1)]*b[(x_j, u)]
            
            pi[(j, u)] = np.amax(pi_ju)
            u_opt, pi_j_max = [u, pi[(j, u)]] if pi[(j, u)] > pi_j_max else [u_opt, pi_j_max] # updating opt tag for x_j
            
        optimal_tags.append(u_opt) # appending optimal sentiments
        
    return optimal_tags[:-2]



# function that generates emission and transmission matrices, sentiment and word dictionaries
    # lang: language string (e.g. 'EN')
    # k: regulator for unseen words
def train_phase2 (lang, k):
    
    # reading tweets for particular language
    ptrain = data_from_file(lang + '/train') # unmodified
    train = mod_train2 (ptrain) # modified w/ start and stop states

    # getting sentiments/sentiment pairs and associated indices (w/ start and stop)
    sents = get_tags2 (ptrain) 
    
    Y = get_counts(train)[0] # dictionary of sentiments and their counts
    word_dict = get_words(train)[1] # dictionary of unique words and indices

    # emission and transmission parameter matrices
    emission_dict = get_emission2 (train, k) # dictionary with keys as (x, y) and values as emission probabilities
    trans_dict = transition_dict (train, Y) # transition dictionary
    
    return trans_dict, emission_dict, sents, word_dict



    
    