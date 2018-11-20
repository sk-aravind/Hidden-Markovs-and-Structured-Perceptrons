from collections import defaultdict
from preprocess import *

# qn 1

def get_counts(train):
    # count occurence of y & yx
    count_y = defaultdict(int)
    count_yx = defaultdict(int)

    for line in train:
        for obs_label in line:
            count_y[obs_label[1]] += 1
            count_yx[obs_label] += 1

    return count_y,count_yx

def get_emission(train):
    # Calculate emission parameters
    count_y,count_yx = get_counts(train)
    emission = defaultdict(float)
    for obs_label, count in count_yx.items():
        emission[obs_label] += count / count_y[obs_label[1]]

    return emission


# qn 2

def get_emission2(train, k):

    count_y = defaultdict(int)
    count_x = defaultdict(int)
    count_yx = defaultdict(int)
    emission = defaultdict(float)

    # Count y,x,yx
    for line in train:
        for obs_label in line:
            count_y[obs_label[1]] += 1
            count_x[obs_label[0]] += 1
            count_yx[obs_label] += 1

    # Replace x with unk if it appears less than k times
    for obs_label, count in count_yx.items():
        if count_x[obs_label[0]] < k:
            emission[('#UNK#', obs_label[1])] += count
        else:
            emission[obs_label] += count

    # Calculate emission
    for obs_label, count in emission.items():
        emission[obs_label] = count / count_y[obs_label[1]]

    return emission


# qn 3

def max_em_label(word, possible_labels, emissions, words):
    if word not in words:
        word = '#UNK#'
    return max((emissions.get((word, label), 0), label) for label in possible_labels)[1]

def predict_max_em(test, emissions):
    words, labels = tuple(set(i) for i in zip(*emissions.keys()))
    return [[(word, max_em_label(word, labels, emissions, words)) for word in line] for line in test]


if __name__ == '__main__':
    outfile = '/dev.p2.out'
    for lang in languages:

        # qn 1
        # A list of list of tuples. Each list in train is a sentence, each tuple in a sentence is a (word, tag).
        train = data_from_file(lang + '/train')
        emission = get_emission(train)

        # qn 2
        new_emission = get_emission2(train, 3)

        # qn 3
        count_y, count_yx = get_counts(train)
        labels = list(count_y)

        # A list of list of tuples of size 1. Each list in test is a sentence.
        test = data_from_file(lang + '/dev.in')
        # test is a list of list. Each sublist is an array of words, 1 tweet
        test = [[word[0] for word in line] for line in test]

        prediction = predict_max_em(test, new_emission)
        write_predictions(prediction, lang, outfile)

        pred = get_entities(open(lang+outfile, encoding='utf-8'))
        gold = get_entities(open(lang+'/dev.out', encoding='utf-8'))
        print(lang)
        compare_result(gold, pred)
        print()



'''
entities
 - Correct   : 2479
 - Precision : 0.1518
 - Recall    : 0.6058
 - F score   : 0.2427

sentiment
 - Correct   : 1136
 - Precision : 0.0695
 - Recall    : 0.2776
 - F score   : 0.1112

'''