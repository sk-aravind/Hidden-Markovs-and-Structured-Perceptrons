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


# function that collates a sorted list of different words, 
# and a dictionary of words as keys and associated indices as values 
def vocab_count (train):
    # train: training data 

    all_words = [] # list of all words in train data
    
    # populating all_words list
    for tweet in train:
        for y in tweet:
            all_words.append(y[0]) # list of all words in data

    diff_words = np.unique(all_words).tolist() # list of unique words

    return len(diff_words)


# function that gets emission parameters with Laplace smoothing
    # train: training data
def emissions_Laplace (train):
     
    V = vocab_count (train) # size of vocabulary
    
    count_y = defaultdict(int) # tag counts
    count_xy = defaultdict(int) # (word, tag) counts
    emission = defaultdict(lambda:1/V) # emission parameters
    
    # count information
    for line in train:
        for xy in line:
            count_y[xy[1]] += 1
            count_xy[xy] += 1
    
    # numerator of emission params
    for xy, count in count_xy.items():
        emission[xy] += count
        
    # emissions with Laplace smoothing
    for xy, count in emission.items():
        emission[xy] = (count + 1) / (count_y[xy[1]] + V)

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
            norm = sum(L) # sum of lambda parameters 
            L = [L_j/norm for L_j in L] # ensuring parameters sum to 1 
            
            # trigram, bigram and unigram to deal with sparsity
            trigram = L[0] / YY[(tweet[y_i-2][1], tweet[y_i - 1][1])]
            bigram = L[1] / Y[tweet[y_i-1][1]]
            unigram = L[2] / (Y[tweet[y_i][1]] ** 2)
            
            # filling up transition matrix
            a_v0v1u[((tweet[y_i - 2][1], tweet[y_i - 1][1]), tweet[y_i][1])] += trigram + bigram + unigram

    return a_v0v1u    


# function that runs the viterbi algorithm for each tweet
    # a: 2nd order transition dictionary
    # b: emission dictionary
    # tags: dictionary of tags and indices
    # words: dictionary of words
    # tweet: tweet from data
def Viterbi_2order2 (a, b, tags, words, tweet):
    
    pi = defaultdict(float) # init score dictionary 
    
    for j in range(len(tweet)): # loop over words in tweet, O(n) complexity
        x_j = tweet[j] if tweet[j] in words else '#UNK#' # word in tweet
        
        for u in tags: # loop over possible tags, O(T) complexity
            if j == 0:
                pi[(j, u)] = a[(('start0', 'start1'), u)] * b[(x_j, u)] # base case

            elif j > 0: 
                pi[(j, u)] = max([pi[(j-1, v1)] * b[(x_j, u)] * \
                              max([a[((v0, v1), u)] for v0 in tags]) \
                              for v1 in tags]) # finding max score for u, O(T^2) complexity         
    
    # stop state scores
    n = len(tweet) # length of tweet
    pi[(n, 'stop0')] = max([pi[(n-1, v1)] * max([a[((v0, v1), 'stop0')] for v0 in tags]) for v1 in tags]) 
    pi[(n+1, 'stop1')] = pi[(n, 'stop0')] * max([a[((v0, 'stop0'), 'stop1')] for v0 in tags])
    
    # function that runs each 2nd order backtracking iteration
    def backtrack (j, v1, u):
        scores = {v0: pi[(j-2, v0)] * a[((v0, v1), u)] for v0 in tags} # scores for word j
        best_tag = max(scores, key=lambda key: scores[key]) if max(scores.values()) > 0 else 'O' # tag with the highest score
        return best_tag 

    reverse_tags = [] # init list 
    v1, u = 'stop0', 'stop1' # setting stop states for back tracking
    
    # bactracking for 2nd order Viterbi algorithm
    for j in range(len(tweet)+1, 1, -1):
        v0 = backtrack(j, v1, u)
        u, v1 = v1, v0 # moving to previous word
        reverse_tags.append(v0)

    return reverse_tags[::-1]



# function that gets F scores 
    # observed: gold data set
    # predicted: generated labels with Viterbi
def F_scores (observed, predicted):
    # Compare between gold data and prediction data.
    total_observed = 0
    total_predicted = 0
    correct = {'entities': 0, 'entity types': 0}

    for example in observed:
        observed_instance = observed[example]
        predicted_instance = predicted[example]
        total_observed += len(observed_instance)
        total_predicted += len(predicted_instance)

        for span in predicted_instance:
            span_sent = span[0]
            span_ne = (span[1], len(span) - 1)

            for observed_span in observed_instance:
                sent = observed_span[0]
                ne = (observed_span[1], len(observed_span) - 1)

                if span_ne == ne:
                    correct['entities'] += 1
                    if span_sent == sent:
                        correct['entity types'] += 1
    
    F = []
    for t in ('entities', 'entity types'):
        prec = correct[t] / total_predicted
        recl = correct[t] / total_observed
        try:
            F.append(2 * prec * recl / (prec + recl))
        except ZeroDivisionError:
            F.append(0)

    return F




# function that generates 2nd order emission and transmission dicts, tags and word dictionaries
    # lang: language string (e.g. 'EN')
    # k: regulator for unseen words
def train_phase_2order2 (lang, k):
    
    # reading tweets for particular language
    ptrain = data_from_file(lang + '/train') # unmodified
    train = mod_train2 (ptrain) # modified w/ start and stop states 
    
    Y = get_count_y (train) # dictionary of tags and their counts
    YY = get_counts2 (train) # dictionary of tag pairs and their counts
    YYY = get_counts3 (train) # dictionary of tag triplets and their counts

    # emission and transmission parameter matrices
    # emission_dict = get_emission2 (ptrain, k) # dictionary with keys as b_xj(u) and values as emission probabilities 
    emission_dict = emissions_Laplace (ptrain) # Laplace smoothed emission dictionary
    trans_dict = transition_dict2a (train, Y, YY, YYY) # 2nd order transition dictionary a_((v0, v1), u)
    
    return trans_dict, emission_dict


# function that runs tuning of hyperparameter k for F-scoring
    # test: test data set
    # lang: language string e.g. 'EN', 'FR', etc...
    # outfile: name of out file e.g. '/dev.p5_ktune.out'
    # k_min: minimum value of k
    # k_max: maximum value of k
    # num_k: number of k values to run over
def k_tuning (test, lang, outfile, k_min, k_max, num_k):
    
    k_vals = np.linspace(k_min, k_max, num_k).tolist() 
    F0 = [] # F score for entities
    F1 = [] # F score for entity types
    
    for k in k_vals:
        # ======================================== training ========================================
        a, b = train_phase_2order2 (lang, k) # getting 2nd order trained model parameters
        # ========================================================================================== 

        # ============================================ getting predictions ============================================
        predictions = []

        for tweet in test:
            words, tags = list(set(i) for i in zip(*b.keys()))
            prediction = list(zip(tweet, Viterbi_2order2 (a, b, tags, words, tweet)))
            predictions.append(prediction)

        write_predictions(predictions, lang, outfile) # writing predictions to outfile
        # =============================================================================================================       

        pred = get_entities(open(lang+outfile, encoding='utf-8'))
        gold = get_entities(open(lang+'/dev.out', encoding='utf-8'))    
        F0.append(F_scores(gold, pred)[0]) # F scores for entities
        F1.append(F_scores(gold, pred)[1]) # F scores for entity types
        
    return k_vals, F0, F1



