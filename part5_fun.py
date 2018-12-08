# functions for improved second order Viterbi algorithm (ML project part 5)
from part4_fun import *


# function that gets counts of tag triplets
    # train: modified processed training set 
def get_counts3(train): # tags2
    # count occurence of y,y
    count_yyy = defaultdict(int)
    
    # getting the (y_i, y_{i+1}) counts
    for line in train:
        for obs_labeli in range(len(line)-2):
            count_yyy[(line[obs_labeli][1], line[obs_labeli+1][1], line[obs_labeli+2][1])] += 1
        
    return count_yyy


# function that computes 2nd order emission parameters
    # train: training data
    # k: smoothing parameter for unseen words
def emissions_dict2 (train, k):
    
    # initializing count information dictionaries
    count_yy = defaultdict(int)
    count_x = defaultdict(int)
    count_xyy = defaultdict(int)
    emission = defaultdict(float)
    
    # counting 
    for line in train:
        for xy_i in range(1,len(line)):
            count_yy[(line[xy_i-1][1], line[xy_i][1])] += 1 # counting tag pairs
            count_x[line[xy_i][0]] += 1 # counting words 
            count_xyy[(line[xy_i][0], line[xy_i-1][1], line[xy_i][1])] += 1 # counting tag pairs to words 
    
    # getting numerator of emission parameters
    for xyy, count in count_xyy.items():
        if count_x[xyy[0]] <= k: # smoothing for unseen words
            emission[('#UNK#', xyy[1], xyy[2])] += count 
        else:
            emission[xyy] += count
    
    # getting emission parameters
    for xyy, count in emission.items():  
        emission[xyy] = count / count_yy[(xyy[1], xyy[2])]

    return emission



# function that computes second order transition parameters with modifications 
    # train: processed training set of features and labels
    # Y: dictionary with tags and counts
    # YY: dictionary with tag pairs and counts
def transition_dict2a (train, Y, YY, YYY):
    
    a_v0v1u = defaultdict(float) # init dictionary 
    
    # counting (v0, v1), u transitions 
    for tweet in train:
        for y_i in range(2, len(tweet)):
            
            # deleted interpolation
            k3 = (math.log1p(YYY[(tweet[y_i-2][1], tweet[y_i - 1][1], tweet[y_i][1])])+1) \
                    / (math.log1p(YYY[(tweet[y_i-2][1], tweet[y_i - 1][1], tweet[y_i][1])])+2)
            k2 = (math.log1p(YY[(tweet[y_i-2][1], tweet[y_i - 1][1])])+1) \
                    / (math.log1p(YY[(tweet[y_i-2][1], tweet[y_i - 1][1])])+2)
            
            # trigram, bigram and unigram parameters
            L = [k3, (1-k3)*k2, (1-k3)*(1-k2)] 
            
            # trigram, bigram and unigram to deal with sparsity
            trigram = L[0] / YY[(tweet[y_i-2][1], tweet[y_i - 1][1])]
            bigram = L[1] / Y[tweet[y_i-1][1]]
            unigram = L[2] / (Y[tweet[y_i][1]] ** 2)
            
            # filling up transition matrix
            a_v0v1u[((tweet[y_i - 2][1], tweet[y_i - 1][1]), tweet[y_i][1])] += trigram + bigram + unigram

    return a_v0v1u    





# function that generates 2nd order emission and transmission dicts, tags and word dictionaries
    # lang: language string (e.g. 'EN')
    # k: regulator for unseen words
def train_phase_2nd_order (lang, k):
    
    # reading tweets for particular language
    ptrain = data_from_file(lang + '/train') # unmodified
    train = mod_train2 (ptrain) # modified w/ start and stop states 
    
    Y = get_count_y (train) # dictionary of tags and their counts
    YY = get_counts2 (train) # dictionary of tag pairs and their counts
    YYY = get_counts3 (train) # dictionary of tag triplets and their counts

    # emission and transmission parameter matrices
    emission_dict = emissions_dict2 (ptrain, k) # dictionary with keys as b_xj(u) and values as emission probabilities 
    trans_dict = transition_dict2 (train, Y, YY, YYY) # 2nd order transition dictionary a_((v0, v1), u)
    
    return trans_dict, emission_dict