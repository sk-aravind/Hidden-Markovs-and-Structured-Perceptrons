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


# function that gets counts of tag pairs
    # train: modified processed training set 
    # tags2: dictionary of sentiments and indices
def get_counts2(train, tags2):
    # count occurence of y,y
    count_yy = defaultdict(int)
    
    # getting the (y_i, y_{i+1}) counts
    for line in train:
        for obs_labeli in range(len(line)-1):
            count_yy[(line[obs_labeli][1], line[obs_labeli+1][1])] += 1
    
    # ensuring all possible tag pairs are in the dictionary 
    for pairs in range(len(tags2.keys())):
        if list(tags2.keys())[pairs] not in list(count_yy.keys()):
            count_yy[list(tags2.keys())[pairs]] = 0
        else: 
            pass
        
    return count_yy


# function that adds start and stop nodes to training data set
    # ptrain: unmodified training data
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



# function that computes second order transition parameters 
    # train: processed training set of features and labels
    # YY: dictionary with tag pairs and counts
def transition_dict2 (train, YY):
    
    a_uv = defaultdict(float) # lambda:1e-30 
    
    # counting (v0, v1), u transitions 
    for tweet in train:
        for y_i in range(2, len(tweet)):
            
            # filling up transition matrix
            a_uv[((tweet[y_i - 2][1], tweet[y_i - 1][1]), tweet[y_i][1])] += 1/YY[(tweet[y_i-2][1], tweet[y_i - 1][1])]

    return a_uv    



# function that runs the 2nd order viterbi algorithm for each tweet
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


# an alternative implementation of the 2nd order viterbi algorithm 
    # a: 2d order transition dictionary
    # b: emission dictionary
    # tags: dictionary of tags and indices
    # words: dictionary of words
    # tweet: tweet from data
def Viterbi2_alt (a, b, tags, words, tweet):
    
    optimal_tags = [] # optimal tags for given tweet
    
    pi = defaultdict(float) # initializing score dictionary
    pi[(0, 'start0')] = 1. # base case 0
    pi[(1, 'start1')] = 1. # base case 1
    
    for j in range(2,len(tweet)): # loop over all words in tweet
        
        u_opt, pi_j_max = ['O', 0.] # default tag and score
        x_jm1 = tweet[j-1] if tweet[j-1] in words else '#UNK#' # j-th word in tweet
        x_j = tweet[j] if tweet[j] in words else '#UNK#' # j-th word in tweet
        
        
        for u in tags: # loop over all possible tags
            
            pi[(j, u)] = max([pi[(j-1, v1)]*b[(x_jm1, v1)]*b[(x_j, u)]* \
                              max([pi[(j-2, v0)]*a[((v0, v1), u)] for v0 in tags]) \
                              for v1 in tags])
            u_opt, pi_j_max = [u, pi[(j, u)]] if pi[(j, u)] > pi_j_max else [u_opt, pi_j_max] # updating opt tag for x_j
            
        optimal_tags.append(u_opt) # appending optimal sentiments
    
    return optimal_tags[:-2]
    
    

# function that generates emission and transmission dicts, sentiment and word dictionaries
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


# function that generates 2nd order emission and transmission dicts, tags and word dictionaries
    # lang: language string (e.g. 'EN')
    # k: regulator for unseen words
def train_phase_2nd_order (lang, k):
    
    # reading tweets for particular language
    ptrain = data_from_file(lang + '/train') # unmodified
    train = mod_train2 (ptrain) # modified w/ start and stop states

    # getting sentiments/sentiment pairs and associated indices (w/ start and stop)
    tags2 = get_tags2 (ptrain) 
    
    YY = get_counts2(train, tags2) # dictionary of sentiments and their counts
    word_dict = get_words(train)[1] # dictionary of unique words and indices

    # emission and transmission parameter matrices
    emission_dict = get_emission2 (train, k) # dictionary with keys as (x, y) and values as emission probabilities
    trans_dict = transition_dict2 (train, YY) # 2nd order transition dictionary
    
    return trans_dict, emission_dict, tags2, word_dict
    
    