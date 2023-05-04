# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import collections 
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.label_count = None
    self.labels = None 
    self.data_map = None
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    
  def check(self, out):
    probability = dict(collections.Counter(out))
    for k in probability.keys():
      probability[k] = probability[k]/float(len(out))
    return probability
  
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    # might be useful in your code later...
    # this is a list of all features in the training set.
    self.features = list(set([f for datum in trainingData for f in datum.keys()]))
    
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """

    "*** YOUR CODE HERE ***"
    # get priors of training labels
    label_count = dict(collections.Counter(trainingLabels)) # total count
    # normalizing label counts (i.e. calculating priors)
    for l in label_count.keys():
      label_count[l] = label_count[l]/float(len(trainingLabels)) 

    # initialize a list for every unique training label as key in a dictionary
    data_map = dict() 
    for label, prior in label_count.items():
      data_map[label] = collections.defaultdict(list) 

    # iterate through training labels and record the chronological order in which the label is arranged
    for label, prior in label_count.items():
      train_label = list()
      for i, l in enumerate(trainingLabels):
        if label == l:
          train_label.append(i)  

      # map correct label order to training data
      train_data = list() 
      for i in train_label: 
        train_data.append(trainingData[i])

      # populate dictionary data_map with correct data-label pair
      for j in range(len(train_data)): 
        for k, ptr in train_data[j].items():
          data_map[label][k].append(ptr)  

    # get total count
    labels = [l for l in label_count]

    # get probabilities for NB classifier
    for l in labels:
      for k, ptr in train_data[l].items():
        data_map[l][k] = self.check(data_map[l][k])

    # update variables 
    self.label_count = label_count
    self.data_map = data_map
    self.labels = labels 
    
    # util.raiseNotDefined()
        
  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    
    # for faces: 0 --> not face and 1 --> face
    print(f'Predicted label: {guesses[0]}') 
    
    return guesses
      
  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    logJoint = util.Counter()
    
    "*** YOUR CODE HERE ***"
    # get probability
    for x in self.labels:
      proba = self.label_count[x]

      # get data and calculate probability
      for k, ptr in datum.items():
        d = self.data_map[x][k]
        proba = proba + math.log(d.get(datum[k], 0.01)) 

      # update log Joint list with calculated probability 
      logJoint[x] = proba

    # print("In log prob")
    # util.raiseNotDefined()
    
    return logJoint
  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2) 
    
    Note: you may find 'self.features' a useful way to loop through all possible features
    """
    featuresOdds = []
       
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

    return featuresOdds
    

    
      
