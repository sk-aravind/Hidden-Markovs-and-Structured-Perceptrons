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
    for sent in range(len(Y.keys())):
        if (list(Y.keys())[sent] != 'start0') and (list(Y.keys())[sent] != 'start1') and (list(Y.keys())[sent] != 'stop0') and (list(Y.keys())[sent] != 'stop1'):
            tags[list(Y.keys())[sent]] = sent+2
            len_tweet += 1
        else:
            pass
        
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
        tweet.insert(0, ('////_','start1'))
        tweet.insert(0, ('////','start0'))
        tweet.append(('\\\\', 'stop0'))
        tweet.append(('\\\\_', 'stop1'))
        
    return train


# function that adds start and stop words to validation/test data set (no labels)
    # ptest: unmodified testing data 
def mod_test2 (ptest):
    
    test = deepcopy(ptest)
    # inserting start and stop nodes
    for tweet in test:
        tweet.insert(0, '////_')
        tweet.insert(0, '////')
        tweet.append('\\\\')
        tweet.append('\\\\_')
        
    return test



# function that generates sentiment pairs with associated indices
    # tags: dictionary of sentiments
def get_tags2pairs (tags):
    
    # generating dictionary of tag pairs
    inv_tags = dict (zip(tags.values(), tags.keys())) # swapping values and keys
    
    tags2 = defaultdict(int) # initializing dictionary of double tags
    for tagi in range(len(inv_tags.keys())):
        for tagj in range(len(inv_tags.keys())):
            tag_1 = inv_tags[tagi] # first tag in tuple 
            tag_2 = inv_tags[tagj] # second tag in tuple
            tags2[(tag_1, tag_2)] = tagi*len(inv_tags.keys()) + tagj
    
    return tags2

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


# function that computes 2nd order transition matrix
    # train: modified processed training set of features and labels
    # YY: dictionary with sentiment pairs and counts
    # sents: dictionary with sentiments and associated indices
    # sents_pairs: dictionary with sentiment pairs and associated indices
def transition_params2(train, YY, sents, sent_pairs):
    
    q2_uv = np.zeros([len(sents.keys()), len(sent_pairs.keys())]) # 2D array transitions
    
    # counting (u_0, u_1),v transitions for all (u_0, u_1),v
    for tweet in range(len(train)):
        for y in range(2, len(train[tweet])):
            
            # comparing data labels with sentiment keys
            label_i = sents[train[tweet][y][1]] 
            label_im1im2 = sent_pairs[(train[tweet][y-2][1], train[tweet][y-1][1])]

            # filling up transition matrix
            q2_uv[label_i, label_im1im2] += 1/YY[(train[tweet][y-2][1], train[tweet][y-1][1])]

    return q2_uv

# function that modifies transition matrix so that zero entries are epsilon instead
    # trans_mat: unmodified transition matrix
def mod_trans2 (trans_mat):
    
    for row in range(len(trans_mat[:, 0])):
        for col in range(len(trans_mat[0, :])):
            if trans_mat[row, col] == 0.0:
                trans_mat[row, col] = 1e-15
            else:
                pass
    return trans_mat

# function that runs the 2nd order viterbi algorithm recursively 
# arguments:
    # emissions: matrix of emission parameters
    # transitions: matrix of transition parameters
    # word_dict: dictionary of words with associated indices
    # line: line of words (tweet)
    # prev_scores0: scores of j-2 column
    # prev_scores1: scores of j-1 column
    # loop_ind: current loop iteration
    # ind_lis: list that stores optimal sentiment indices
def viterbi_algo2 (em_mat, trans_mat2, 
                  word_dict, line, prev_scores0, prev_scores1, 
                  loop_ind=2, ind_list=[]):

    word_ind = word_dict[line[loop_ind][0]] # associated index of current word
    emissions = em_mat[:, word_ind].reshape((len(trans_mat2[:,0]),1)) # col of emission parameters for current word
    scores = emissions*trans_mat2*np.transpose(np.kron(prev_scores0, prev_scores1)) # matrix of all possible scores 
    
    current_scores = np.zeros([len(prev_scores1), 1]) # init score array of current word layer 
    # loop to fill current score column
    for row in range(len(prev_scores1)):   
        current_scores[row, 0] = np.amax(scores[row,:])
    scores = np.zeros([len(current_scores), len(current_scores)^2]) # resetting score matrix
    
    # check statements to terminate recursion
    if loop_ind < len(line)-2:
        
        loop_ind += 1 # setting next iterations index
        # ensures no storing of start states
        if (current_scores.any() != 0.0):
            ind_list.append(np.argmax(current_scores)) # storing optimal path node indices  
        else:
            ind_list.append(2) # index for 'O'
            
        return viterbi_algo2(em_mat, trans_mat2, word_dict, line, prev_scores1, current_scores, loop_ind, ind_list)
    
    else:
        return ind_list
    
    
    