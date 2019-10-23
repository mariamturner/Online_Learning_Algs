import json
import matplotlib.pylab as plt
import math
import os
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import random

def calculate_f1(y_gold, y_model):
    """
    Computes the F1 of the model predictions using the
    gold labels. Each of y_gold and y_model are lists with
    labels 1 or -1. The function should return the F1
    score as a number between 0 and 1.
    """
    actualEntity = 0
    modelPredEntity = 0
    correctEntity = 0
    for i in range(len(y_gold)):
        if y_gold[i] == 1:
            actualEntity += 1
        if y_model[i] == 1:
            if y_gold[i] == 1:
                correctEntity += 1
            modelPredEntity += 1
    precision = correctEntity/modelPredEntity
    recall = correctEntity/actualEntity
    f1 = 2 * ((precision * recall)/(precision + recall))
    return f1

class Classifier(object):
    """
    The Classifier class is the base class for all of the Perceptron-based
    algorithms. Your class should override the "process_example" and
    "predict_single" functions. Further, the averaged models should
    override the "finalize" method, where the final parameter values
    should be calculated. You should not need to edit this class any further.
    """
    def train(self, X, y):
        iterations = 10
        for iteration in range(iterations):
            for x_i, y_i in zip(X, y):
                self.process_example(x_i, y_i)
        self.finalize()

    def process_example(self, x, y):
        """
        Makes a predicting using the current parameter values for
        the features x and potentially updates the parameters based
        on the gradient. "x" is a dictionary which maps from the feature
        name to the feature value and y is either 1 or -1.
        """
        raise NotImplementedError

    def finalize(self):
        """Calculates the final parameter values for the averaged models."""
        pass

    def predict(self, X):
        """
        Predicts labels for all of the input examples. You should not need
        to override this method.
        """
        y = []
        for x in X:
            y.append(self.predict_single(x))
        return y

    def predict_single(self, x):
        """
        Predicts a label, 1 or -1, for the input example. "x" is a dictionary
        which maps from the feature name to the feature value.
        """
        raise NotImplementedError


class Perceptron(Classifier):
    def __init__(self, features):
        """
        Initializes the parameters for the Perceptron model. "features"
        is a list of all of the features of the model where each is
        represented by a string.
        """
        # Do not change the names of these 3 data members because
        # they are used in the unit tests
        self.eta = 1
        self.w = {feature: 0.0 for feature in features}
        self.theta = 0

    def process_example(self, x, y):
        y_pred = self.predict_single(x)
        if y != y_pred:
            for feature, value in x.items():
                self.w[feature] += self.eta * y * value
            self.theta += self.eta * y

    def predict_single(self, x):
        score = 0
        for feature, value in x.items():
            score += self.w[feature] * value
        score += self.theta
        if score <= 0:
            return -1
        return 1


class Winnow(Classifier):
    def __init__(self, alpha, features):
        # Do not change the names of these 3 data members because
        # they are used in the unit tests
        self.alpha = alpha
        self.w = {feature: 1.0 for feature in features}
        self.theta = -len(features)
        
    def process_example(self, x, y):
        y_pred = self.predict_single(x)
        if y != y_pred:
            for feature, value in x.items():
                self.w[feature] = self.w[feature] * (self.alpha ** (y * value))

    def predict_single(self, x):
        score = 0
        for feature, value in x.items():
            score += self.w[feature] * value
        score += self.theta
        if score <= 0:
            return -1
        return 1

class AdaGrad(Classifier):
    def __init__(self, eta, features):
        # Do not change the names of these 3 data members because
        # they are used in the unit tests
        self.eta = eta
        self.w = {feature: 0.0 for feature in features}
        self.theta = 0
        self.G = {feature: 1e-5 for feature in features}  # 1e-5 prevents divide by 0 problems
        self.H = 0
        
    def process_example(self, x, y):
        wtx = sum([self.w[feature] * value for feature, value in x.items()])
        if y*(wtx + self.theta) <= 1:
            for feature, value in x.items():
                G_accum = -y * value
                self.G[feature] = self.G[feature] + pow(G_accum,2)
                self.w[feature] = self.w[feature] - self.eta * G_accum / math.sqrt(self.G[feature])
            self.H += pow(y, 2)
            self.theta += (self.eta * (y/math.sqrt(self.H)))

    def predict_single(self, x):
        score = 0
        for feature, value in x.items():
            score += self.w[feature] * value
        score += self.theta
        if score <= 0:
            return -1
        return 1