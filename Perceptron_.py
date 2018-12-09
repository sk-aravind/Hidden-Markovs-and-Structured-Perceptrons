#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
lemao
"""
import argparse

class Model:

    def __init__(self, filePath, k=1):
        self.filePath = filePath
        self.k = k              # Observation replacement parameter
        self.obs = {}           # Dict of Observations
        self.obsList = []       # List of Observations/words
        self.tags = []          # Tag
        self.eParams = {}       # Emission Parameters
        self.tParams = {}       # Transition Parameters
        self.train_count =0

    def readData(self):
        # Process the data
        for line in open(self.filePath, 'r'):
            lineAsList = line.rstrip()
            if lineAsList:  # if its not just an empty string
                lineAsList = lineAsList.rsplit(' ', 1)

                # observation = casefold(segmentedLine[0])  # More Aggressive,but takes longer
                word = lineAsList[0].lower()  # WORD
                tag = lineAsList[1]  # TAG

                if word not in self.obs:  # if this observation has never been seen before
                    self.obs[word] = 1
                else:  # if this observation has been seen before
                    self.obs[word] += 1

                if tag not in self.tags:  # if this tag has never been seen before
                    self.tags.append(tag)

    def replaceWord(self):
        # Replace observations which appear for less than k times with ##UNK##
        self.obsList = list(self.obs)

        for word in self.obs:
            if self.obs[word] < self.k:
                self.obsList.remove(word)

        self.obsList.append('##UNK##')

    def createEParams(self):
        for tag in self.tags:
            self.eParams[tag] = {}
            for ob in self.obsList:
                self.eParams[tag][ob] = [0.0, 0.0]  # first term is weight, second term is average weight

    def createTParams(self):
        p_tagsList, c_tagsList = list(self.tags), list(self.tags)
        p_tagsList.append('##START##')
        c_tagsList.append('##STOP##')

        for p_tag in p_tagsList:
            self.tParams[p_tag] = {}
            for c_tag in c_tagsList:
                self.tParams[p_tag][c_tag] = [0.0, 0.0]

    def trainPerceptron(self, epochs=3, lr=1.0):

        self.train_count = 0
        for t in list(range(epochs)):
            # print('Epoch', t, '...')
            obsSequence = []
            validTagSequence = []

            for line in open(self.filePath, 'r'):
                lineAsList = line.rstrip()
                if lineAsList:
                    lineAsList = lineAsList.rsplit(' ', 1)
                    word, tag = lineAsList[0], lineAsList[1]  # X
                    obsSequence.append(word.lower())
                    validTagSequence.append(tag)
                else:
                    # Run Viterbi with given Params, self.obsList= train_data
                    predictions = self.viterbi(obsSequence, self.obsList, self.eParams, self.tParams)
                    # Update Weights based on viterbi predictions
                    self.eParams, self.tParams = self.updateWeights(obsSequence, validTagSequence,
                                                predictions,self.obsList, self.eParams, self.tParams, lr=lr)
                    self.train_count += 1
                    obsSequence = []
                    validTagSequence = []

    def averageWeights(self):
        # Calculate the average weights for emission features
        for tag in list(self.eParams):
            for observation in self.eParams[tag]:
                self.eParams[tag][observation][0] /= (self.train_count + 1)

        # Calculate the average weights for transition features
        for tag_a in list(self.tParams):
            for tag_b in self.tParams[tag_a]:
                self.tParams[tag_a][tag_b][0] /= (self.train_count + 1)

    def addStartStop(self,tags, pad=False):
        if pad:
            tags.insert(0, '')
            tags.append('')
        else:
            tags.insert(0, '##START##')
            tags.append('##STOP##')
        return tags

    def updateWeights(self,obs, val, pred, train_data, eParams, tParams, lr=1.0):
        # Update perceptron weights based on viterbi predictions
        # obs = Observations Sequence , val = Validatio Tags , pred = Prediction Tags
        val = self.addStartStop(val)
        pred = self.addStartStop(pred)
        obs = self.addStartStop(obs, pad=True)

        for i in range(len(val)):
            if val[i] != pred[i]:

                # Update weights for emission params
                if obs[i] in train_data:
                    eParams[val[i]][obs[i]][0] += lr
                    eParams[val[i]][obs[i]][1] += eParams[val[i]][obs[i]][0]

                    eParams[pred[i]][obs[i]][0] -= lr
                    eParams[pred[i]][obs[i]][1] += eParams[pred[i]][obs[i]][0]
                else:  # if this word is ##UNK##
                    eParams[val[i]]['##UNK##'][0] += lr
                    eParams[val[i]]['##UNK##'][1] += eParams[val[i]]['##UNK##'][0]

                    eParams[pred[i]]['##UNK##'][0] -= lr
                    eParams[pred[i]]['##UNK##'][1] += eParams[pred[i]]['##UNK##'][0]

                # Update weights for transition params
                tParams[val[i - 1]][val[i]][0] += lr
                tParams[val[i - 1]][val[i]][1] += tParams[val[i - 1]][val[i]][0]
                tParams[val[i]][val[i + 1]][0] += lr
                tParams[val[i]][val[i + 1]][1] += tParams[val[i]][val[i + 1]][0]

                tParams[pred[i - 1]][pred[i]][0] -= lr
                tParams[pred[i - 1]][pred[i]][1] += tParams[pred[i - 1]][pred[i]][0]
                tParams[pred[i]][pred[i + 1]][0] -= lr
                tParams[pred[i]][pred[i + 1]][1] += tParams[pred[i]][pred[i + 1]][0]

        return eParams, tParams


    def viterbi(self,obsSequence, train_data, eParams, tParams):

        probs = [{tag: [None, ''] for tag in eParams} for o in obsSequence]  # probabilities

        # Initialization
        for tag in eParams:
            score = 0.0
            score += tParams['##START##'][tag][0] # for start transitions
            if obsSequence[0] in train_data:  # check if word is in train
                score += eParams[tag][obsSequence[0]][0]
            else:
                score += eParams[tag]['##UNK##'][0]
            probs[0][tag] = [score, '##START##']

        # Recurse through sequence
        for word_index in list(range(1, len(obsSequence))):  # probs[k][tag_a] = max(a(tag_b, tag_a)...)
            for tag_a in eParams:
                for tag_b in eParams:
                    score = probs[word_index - 1][tag_b][0]
                    score += tParams[tag_b][tag_a][0]
                    if probs[word_index][tag_a][0] is None:
                        probs[word_index][tag_a] = [score, tag_b]
                    elif score > probs[word_index][tag_a][0]:
                        probs[word_index][tag_a] = [score, tag_b]

                if obsSequence[word_index] in train_data:
                    probs[word_index][tag_a][0] += eParams[tag_a][obsSequence[word_index]][0]
                else:
                    probs[word_index][tag_a][0] += eParams[tag_a]['##UNK##'][0]

        # Find highest prob for last observation
        prob_last_obs = [None, '']
        for tag in eParams:
            score = probs[-1][tag][0] + tParams[tag]['##STOP##'][0]  # account for final transition to '##STOP##'

            if prob_last_obs[0] is None:
                prob_last_obs = [score, tag]
            elif score > prob_last_obs[0]:
                prob_last_obs = [score, tag]

        # Backtracking
        prediction = [prob_last_obs[1]]
        for k in reversed(list(range(len(obsSequence)))):
            if k == 0: break  # skips ##START## tag
            prediction.insert(0, probs[k][prediction[0]][1])

        return prediction


    def predict(self,inputPath, outputPath):
        """ splits test file into separate observation sequences and feeds them into Viterbi algorithm """
        f = open(outputPath, 'w')
        # print('Model is predicting.....')
        observationSequence = []
        lower_observationSequence = []
        for line in open(inputPath, 'r'):
            observation = line.rstrip()
            if observation:
                observationSequence.append(observation)
                lower_observationSequence.append(observation.lower())
            else:
                predictionSequence = self.viterbi(lower_observationSequence, self.obsList, self.eParams, self.tParams)
                for i in list(range(len(observationSequence))):
                    f.write('%s %s\n' % (observationSequence[i], predictionSequence[i]))
                f.write('\n')
                observationSequence = []
                lower_observationSequence = []

        # print ('Model prediction has successfully concluded')
        # print ('Prediction Results have been stored in %s' % (outputPath))

        return f.close()

def averageWeights(eParams,tParams,train_count):
    # Calculate the average weights for emission features
    for tag in list(eParams):
        for observation in eParams[tag]:
            eParams[tag][observation][0] /= (train_count + 1)

    # Calculate the average weights for transition features
    for tag_a in list(tParams):
        for tag_b in tParams[tag_a]:
            tParams[tag_a][tag_b][0] /= (train_count + 1)

    return eParams,tParams


def predict(inputPath, outputPath, eParams, tParams,obsList):
    f = open(outputPath, 'w')
    # print('Model is predicting.....')
    observationSequence = []
    lower_observationSequence = []
    for line in open(inputPath, 'r'):
        observation = line.rstrip()
        if observation:
            observationSequence.append(observation)
            lower_observationSequence.append(observation.lower())
        else:
            predictionSequence = viterbi(lower_observationSequence, obsList, eParams, tParams)
            for i in list(range(len(observationSequence))):
                f.write('%s %s\n' % (observationSequence[i], predictionSequence[i]))
            f.write('\n')
            observationSequence = []
            lower_observationSequence = []

    # print ('Model prediction has successfully concluded')
    # print ('Prediction Results have been stored in %s' % (outputPath))

    return f.close()    
  
  
def viterbi(obsSequence, train_data, eParams, tParams):

    probs = [{tag: [None, ''] for tag in eParams} for o in obsSequence]  # probabilities

    # Initialization
    for tag in eParams:
        score = 0.0
        score += tParams['##START##'][tag][0] # for start transitions
        if obsSequence[0] in train_data:  # check if word is in train
            score += eParams[tag][obsSequence[0]][0]
        else:
            score += eParams[tag]['##UNK##'][0]
        probs[0][tag] = [score, '##START##']

    # Recurse through sequence
    for word_index in list(range(1, len(obsSequence))):  # probs[k][tag_a] = max(a(tag_b, tag_a)...)
        for tag_a in eParams:
            for tag_b in eParams:
                score = probs[word_index - 1][tag_b][0]
                score += tParams[tag_b][tag_a][0]
                if probs[word_index][tag_a][0] is None:
                    probs[word_index][tag_a] = [score, tag_b]
                elif score > probs[word_index][tag_a][0]:
                    probs[word_index][tag_a] = [score, tag_b]

            if obsSequence[word_index] in train_data:
                probs[word_index][tag_a][0] += eParams[tag_a][obsSequence[word_index]][0]
            else:
                probs[word_index][tag_a][0] += eParams[tag_a]['##UNK##'][0]

    # Find highest prob for last observation
    prob_last_obs = [None, '']
    for tag in eParams:
        score = probs[-1][tag][0] + tParams[tag]['##STOP##'][0]  # account for final transition to '##STOP##'

        if prob_last_obs[0] is None:
            prob_last_obs = [score, tag]
        elif score > prob_last_obs[0]:
            prob_last_obs = [score, tag]

    # Backtracking
    prediction = [prob_last_obs[1]]
    for k in reversed(list(range(len(obsSequence)))):
        if k == 0: break  # skips ##START## tag
        prediction.insert(0, probs[k][prediction[0]][1])

    return prediction



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, dest='dataset', help='Dataset to run script over', required=False, default='EN')
    parser.add_argument('-k', type=int, dest='k',
                        help='Smoothing variable', default=1,
                        required=False)
    parser.add_argument('-epochs', type=int, dest='i',
                        help='Number of epochs to train perceptron', default=1,
                        required=False)

    args = parser.parse_args()

    training_data_path = '../%s/train' % (args.dataset)
    validation_data_path = '../%s/dev.in' % (args.dataset)
    output_data_path = '../%s/dev.p5.out' % (args.dataset)
    test_data_path = '../%s/test.in' % (args.dataset)
    test_output_data_path = '../%s/test.p5.out' % (args.dataset)

    model = Model(training_data_path, k=args.k)
    model.readData()
    model.replaceWord()
    model.createEParams()
    model.createTParams()
    model.trainPerceptron(epochs=args.i)
    model.averageWeights()
    model.predict(validation_data_path,output_data_path)

    # For test data
    model.predict(test_data_path, test_output_data_path)


