# functions for second order Viterbi algorithm (ML project part 4)

from part3_fun import *
from copy import deepcopy
import math


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


# function that gets counts of tag pairs
    # train: modified processed training set 
def get_counts2(train): # tags2
    # count occurence of y,y
    count_yy = defaultdict(int)
    
    # getting the (y_i, y_{i+1}) counts
    for line in train:
        for obs_labeli in range(len(line)-1):
            count_yy[(line[obs_labeli][1], line[obs_labeli+1][1])] += 1
        
    return count_yy




# function that computes second order transition parameters 
    # train: processed training set of features and labels
    # YY: dictionary with tag pairs and counts
def transition_dict2 (train, YY):
    
    a_v0v1u = defaultdict(lambda:1e-15) # defaultdict(float) # init dictionary 
    
    # counting (v0, v1), u transitions 
    for tweet in train:
        for y_i in range(2, len(tweet)):
            
            # filling up transition matrix
            a_v0v1u[((tweet[y_i - 2][1], tweet[y_i - 1][1]), tweet[y_i][1])] += 1 / YY[(tweet[y_i-2][1], tweet[y_i - 1][1])]

    return a_v0v1u    

    
    
# # function that runs the viterbi algorithm for each tweet
#     # a: 2nd order transition dictionary
#     # b: emission dictionary
#     # tags: dictionary of tags and indices
#     # words: dictionary of words
#     # tweet: tweet from data
def Viterbi_2nd_order (a, b, tags, words, tweet):
    
    pi = defaultdict(float)
    pi[(-2, 'start0')] = 1.
    pi[(-1, 'start1')] = 1.
                
    for j in range(len(tweet)): # loop over words in tweet
        
        x_j = tweet[j] if tweet[j] in words else '#UNK#' # word in tweet
        
        for u in tags: # loop over possible tags
            
            if j == 0:
                # pi[(j, u)] = a[(('start0', 'start1'), u)] * b[(x_j, 'start1', u)] # base case
                pi[(j, u)] = a[(('start0', 'start1'), u)] * b[(x_j, u)] # base case

            elif j > 0:
                pi[(j, u)] = max([pi[(j-1, v1)] * b[(x_j, u)] * \
                              max([a[((v0, v1), u)] for v0 in tags]) \
                              for v1 in tags]) # finding max score for u
                
#                 pi[(j, u)] = max([pi[(j-1, v1)] * b[(x_j, v1, u)] * \
#                               max([a[((v0, v1), u)] for v0 in tags]) \
#                               for v1 in tags]) # finding max score for u            
    
    # stop state scores
    n = len(tweet) # length of tweet
    pi[(n, 'stop0')] = max([pi[(n-1, v1)] * \
                              max([a[((v0, v1), 'stop0')] for v0 in tags]) \
                              for v1 in tags]) 
    pi[(n+1, 'stop1')] = pi[(n, 'stop0')] * max([a[((v0, 'stop0'), 'stop1')] for v0 in tags])
    
    # function that runs each backtracking iteration
    def backtrack (j, v1, u):
        scores = {v0: pi[(j-2, v0)] * a[((v0, v1), u)] for v0 in tags}
        
        # tag with the highest score
        best_tag = max(scores, key=lambda key: scores[key]) if max(scores.values()) > 0 else 'O' 
        return best_tag 

    reverse_tags = [] # init list 
    v1 = 'stop0' # setting stop0 for back tracking
    u = 'stop1' # setting stop1 for back tracking
    
    # bactracking for Viterbi algorithm
    for j in range(len(tweet)+1, 1, -1):
        v0 = backtrack(j, v1, u)
        u = v1
        v1 = v0  # moving to previous word
        reverse_tags.append(v0)

    return reverse_tags[::-1]



# function that generates 2nd order emission and transmission dicts, tags and word dictionaries
    # lang: language string (e.g. 'EN')
    # k: regulator for unseen words
def train_phase_2nd_order (lang, k):
    
    # reading tweets for particular language
    ptrain = data_from_file(lang + '/train') # unmodified
    train = mod_train2 (ptrain) # modified w/ start and stop states 
    
    YY = get_counts2 (train) # dictionary of tag pairs and their counts
    
    # emission and transmission parameter matrices
    emission_dict = get_emission2 (ptrain, k) # dictionary with keys as b_xj(u) and values as emission probabilities
    trans_dict = transition_dict2 (train, YY) # 2nd order transition dictionary a_((v0, v1), u)
        
    return trans_dict, emission_dict
    
    