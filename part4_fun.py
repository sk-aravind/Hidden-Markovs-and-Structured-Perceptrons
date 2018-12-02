from part3_fun import *
from copy import deepcopy

# function that gets different sentiments (tags) from a data set
# this function adds the start and stop tags
def get_tags2(data):
    # data: unmodified training data
    
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
    tags['stop0'] = len_tweet + 2
    tags['stop1'] = len_tweet + 3
    tags['start0'] = 0
    tags['start1'] = 1
    
    return tags


# function that adds start and stop nodes to training set
def mod_train2 (ptrain):
    
    train = deepcopy(ptrain)
    # inserting start and stop nodes
    for tweet in train:
        tweet.insert(0, ('////_','start1'))
        tweet.insert(0, ('////','start0'))
        tweet.append(('\\\\', 'stop0'))
        tweet.append(('\\\\_', 'stop1'))
        
    return train


# function that adds start and stop nodes to validation set
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
def get_counts2(train, tags2):
    # count occurence of y,y
    count_yy = defaultdict(int)
    
    # getting the y_i, y_{i+1} counts
    for line in train:
        for obs_labeli in range(len(line)-1):
            count_yy[(line[obs_labeli][1], line[obs_labeli+1][1])] += 1
    
    # ensuring count_yy has count info for all possible pairs
    for pairs in range(len(tags2.keys())):
        if list(tags2.keys())[pairs] not in list(count_yy.keys()):
            count_yy[list(tags2.keys())[pairs]] = 0
        else: 
            pass
        
    return count_yy


# function that computes 2nd order transition parameter matrix
def transition_params2(train, YY, sents, sent_pairs):
    # train: modified processed training set of features and labels
    # YY: dictionary with sentiment pairs and counts
    # sents: dictionary with sentiments and associated indices
    # sents_pairs: dictionary with sentiment pairs and associated indices
    
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



# function that runs the viterbi algorithm recursively 
def viterbi_algo2 (em_mat, trans_mat2, 
                  word_dict, line, prev_scores0, prev_scores1, 
                  loop_ind=2, ind_list=[]):
    # emissions: matrix of emission parameters
    # transitions: matrix of transition parameters
    # word_dict: dictionary of words with associated indices
    # line: line of words (tweet)
    # prev_scores0: scores of j-2 column
    # prev_scores1: scores of j-1 column
    # loop_ind: current loop iteration
    # ind_lis: list that stores optimal sentiment indices

    word_ind = word_dict[line[loop_ind][0]] # associated index of current word
    emissions = em_mat[:, word_ind].reshape((len(trans_mat2[:,0]),1)) # col of emission parameters for current word
    scores = emissions*trans_mat2*np.transpose(np.kron(prev_scores0, prev_scores1)) # matrix of all possible scores 
    
    current_scores = np.zeros([len(prev_scores1), 1]) # init score array of current word layer 
    # loop to fill current score column
    for row in range(len(prev_scores1)):   
        current_scores[row, 0] = np.amax(scores[row,:])

    # check statements to terminate recursion
    if loop_ind < len(line)-2:
        loop_ind += 1 # setting next iterations index
        ind_list.append(np.argmax(current_scores)) # storing optimal path node indices  
        return viterbi_algo2(em_mat, trans_mat2, word_dict, line, prev_scores1, current_scores, loop_ind, ind_list)
    
    else:
        return ind_list
    
    
    