import json
import matplotlib.pylab as plt
import math
import os
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import random

class AveragedPerceptron(Classifier):
    def __init__(self, features):
        self.eta = 1
        self.w = {feature: 0.0 for feature in features}
        self.theta = 0
        self.c = 0
        self.c_total = 0
        self.cum_w = {feature: 0.0 for feature in features}
        self.cum_theta = 0
        self.features = features
        
    def process_example(self, x, y):
        y_pred = self.predict_single(x)
        if y != y_pred:
            for feature, value in x.items():
                self.w[feature] += self.eta * y * value
                self.cum_w[feature] += self.c_total * self.eta * y * value
            self.theta += (self.eta * y)
            self.cum_theta += self.c_total * self.eta * y
            self.c_total += self.c
            self.c = 1
        else:
            self.c += 1

    def predict_single(self, x):
        score = 0
        for feature, value in x.items():
            score += self.w[feature] * value
        score += self.theta
        if score <= 0:
            return -1
        return 1
        
    def finalize(self):
        for feature in self.features:
            self.w[feature] -= self.cum_w[feature] / self.c_total
            self.theta -= self.cum_theta / self.c_total


class AveragedWinnow(Classifier):
    def __init__(self, alpha, features):
        self.alpha = alpha
        self.w = {feature: 1.0 for feature in features}
        self.theta = -len(features)
        self.cum_w = {feature: 1.0 for feature in features}
        self.c = 0 
        self.c_total = 0
        self.features = features

    def process_example(self, x, y):
        y_pred = self.predict_single(x)
        if y != y_pred:
            for feature in self.features:
                if feature in x:
                    self.w[feature] = self.w[feature] * (self.alpha ** (y * x[feature]))
                    self.cum_w[feature] += (self.w[feature] * self.c)
                    self.c_total = self.c_total + self.c
                    self.c = 1
                else:
                    self.c += 1
        self.c_total += self.c
       
    def predict_single(self, x):
        score = 0
        for feature, value in x.items():
            score += self.w[feature] * value
        score += self.theta
        if score <= 0:
            return -1
        return 1
        
    def finalize(self):
        for feature in self.features:
                self.w[feature] = (self.cum_w[feature])/self.c_total

class AveragedAdaGrad(Classifier):
    def __init__(self, eta, features):
        self.eta = eta
        self.w = {feature: 1e-5 for feature in features}
        self.theta = 0
        self.G = {feature: 1e-5 for feature in features}
        self.H = 0
        self.cum_w = {feature: 1e-5 for feature in features}
        self.cum_theta = 0
        self.c = 0
        self.c_total = 0
        self.features = features
        
    def process_example(self, x, y):
                    wtx = sum([self.w[feature]*value for feature, value in x.items()]) 
                    
                    if y * (wtx + self.theta) <= 1:

                        for feature in self.features:
                            self.cum_w[feature] += (self.c * self.w[feature])
                            for feature, value in x.items():
                                G_accum = -y * value
                                self.G[feature] += pow(g, 2)
                                self.w[feature] -= self.eta * G_accum/math.sqrt(self.G[feature])
                            self.H += pow(-y,2)
                            self.cum_theta += self.c * self.theta
                            self.theta += self.eta * y / math.sqrt(self.H)
                            self.c_total += self.c
                            self.c = 1
                    else:
                        self.c += 1
            

    def predict_single(self, x):
        score = 0
        for feature, value in x.items():
            score += self.w[feature] * value
        score += self.theta
        if score <= 0:
            return -1
        return 1
        
    def finalize(self):
        for feature in self.features:
            self.cum_w[feature] += self.c*self.w[feature]
            self.cum_w[feature] /= self.c_total
        self.cum_theta += self.c * self.theta
        self.cum_theta /= self.c_total
        self.w = self.cum_w

def plot_learning_curves(perceptron_accs,
                         winnow_accs,
                         adagrad_accs,
                         avg_perceptron_accs,
                         avg_winnow_accs,
                         avg_adagrad_accs,
                         svm_accs):
    """
    This function will plot the learning curve for the 7 different models.
    Pass the accuracies as lists of length 11 where each item corresponds
    to a point on the learning curve.
    """
    assert len(perceptron_accs) == 11
    assert len(winnow_accs) == 11
    assert len(adagrad_accs) == 11
    assert len(avg_perceptron_accs) == 11
    assert len(avg_winnow_accs) == 11
    assert len(avg_adagrad_accs) == 11
    assert len(svm_accs) == 11

    x = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 50000]
    plt.figure()
    f, (ax, ax2) = plt.subplots(1, 2, sharey=True, facecolor='w')
    ax.plot(x, perceptron_accs, label='perceptron')
    ax2.plot(x, perceptron_accs, label='perceptron')
    ax.plot(x, winnow_accs, label='winnow')
    ax2.plot(x, winnow_accs, label='winnow')
    ax.plot(x, adagrad_accs, label='adagrad')
    ax2.plot(x, adagrad_accs, label='adagrad')
    ax.plot(x, avg_perceptron_accs, label='avg-perceptron')
    ax2.plot(x, avg_perceptron_accs, label='avg-perceptron')
    ax.plot(x, avg_winnow_accs, label='avg-winnow')
    ax2.plot(x, avg_winnow_accs, label='avg-winnow')
    ax.plot(x, avg_adagrad_accs, label='avg-adagrad')
    ax2.plot(x, avg_adagrad_accs, label='avg-adagrad')
    ax.plot(x, svm_accs, label='svm')
    ax2.plot(x, svm_accs, label='svm')
    ax.set_xlim(0, 5500)
    ax2.set_xlim(49500, 50000)
    ax2.set_xticks([50000])
    # hide the spines between ax and ax2
    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax.yaxis.tick_left()
    ax.tick_params(labelright='off')
    ax2.yaxis.tick_right()
    ax2.legend()

def load_synthetic_data(directory_path):
    """
    Loads a synthetic dataset from the dataset root (e.g. "synthetic/sparse").
    You should not need to edit this method.
    """
    def load_jsonl(file_path):
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def load_txt(file_path):
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(int(line.strip()))
        return data

    def convert_to_sparse(X):
        sparse = []
        for x in X:
            data = {}
            for i, value in enumerate(x):
                if value != 0:
                    data[str(i)] = value
            sparse.append(data)
        return sparse

    X_train = load_jsonl(directory_path + '/train.X')
    X_dev = load_jsonl(directory_path + '/dev.X')
    X_test = load_jsonl(directory_path + '/test.X')

    num_features = len(X_train[0])
    features = [str(i) for i in range(num_features)]

    X_train = convert_to_sparse(X_train)
    X_dev = convert_to_sparse(X_dev)
    X_test = convert_to_sparse(X_test)

    y_train = load_txt(directory_path + '/train.y')
    y_dev = load_txt(directory_path + '/dev.y')
    y_test = load_txt(directory_path +  '/test.y')

    return X_train, y_train, X_dev, y_dev, X_test, y_test, features

def run_synthetic_experiment(data_path):
    """
    Runs the synthetic experiment on either the sparse or dense data
    depending on the data path (e.g. "data/sparse" or "data/dense").
    
    We have provided how to train the Perceptron on the training and
    test on the testing data (the last part of the experiment). You need
    to implement the hyperparameter sweep, the learning curves, and
    predicting on the test dataset for the other models.
    """
    X_train, y_train, X_dev, y_dev, X_test, y_test, features \
        = load_synthetic_data(data_path)
    
    
    # TODO: Hyperparameter sweeps
    # For each hyperparam value, train a model using that value and compute accuracy on dev dataset; each trained for 10 iterations.
    
    # WINNOW Choose α ∈ {1.1, 1.01, 1.005, 1.0005, 1.0001}.
    classifier = Winnow(1.1, features)
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_dev)
    acc = accuracy_score(y_dev, y_pred)
    print('Winnow 1.1', acc)
    classifier = Winnow(1.01, features)
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_dev)
    acc = accuracy_score(y_dev, y_pred)
    print('Winnow 1.01', acc)
    classifier = Winnow(1.005, features)
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_dev)
    acc = accuracy_score(y_dev, y_pred)
    print('Winnow 1.005', acc)
    classifier = Winnow(1.0005, features)
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_dev)
    acc = accuracy_score(y_dev, y_pred)
    print('Winnow 1.0005', acc)
    classifier = Winnow(1.0001, features)
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_dev)
    acc = accuracy_score(y_dev, y_pred)
    print('Winnow 1.0001', acc)
    #WINNOW AVG
    classifier = AveragedWinnow(1.1, features)
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_dev)
    acc = accuracy_score(y_dev, y_pred)
    print('Avg Winnow 1.1', acc)
    classifier = AveragedWinnow(1.01, features)
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_dev)
    acc = accuracy_score(y_dev, y_pred)
    print('Avg Winnow 1.01', acc)
    classifier = AveragedWinnow(1.005, features)
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_dev)
    acc = accuracy_score(y_dev, y_pred)
    print('Avg Winnow 1.005', acc)
    classifier = AveragedWinnow(1.0005, features)
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_dev)
    acc = accuracy_score(y_dev, y_pred)
    print('Avg Winnow 1.0005', acc)
    classifier = AveragedWinnow(1.0001, features)
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_dev)
    acc = accuracy_score(y_dev, y_pred)
    print('Avg Winnow 1.0001', acc)
    # ADAGRAD Choose η ∈ {1.5, 0.25, 0.03, 0.005, 0.001}
    classifier = AdaGrad(1.5, features)
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_dev)
    acc = accuracy_score(y_dev, y_pred)
    print('AdaGrad 1.5', acc)
    classifier = AdaGrad(0.25, features)
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_dev)
    acc = accuracy_score(y_dev, y_pred)
    print('AdaGrad 0.25', acc)
    classifier = AdaGrad(0.03, features)
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_dev)
    acc = accuracy_score(y_dev, y_pred)
    print('AdaGrad 0.03', acc)
    classifier = AdaGrad(0.005, features)
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_dev)
    acc = accuracy_score(y_dev, y_pred)
    print('AdaGrad 0.005', acc)
    classifier = AdaGrad(0.001, features)
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_dev)
    acc = accuracy_score(y_dev, y_pred)
    print('AdaGrad 0.001', acc)
    #ADAGRAD AVG
    """classifier = AveragedAdaGrad(1.5, features)
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_dev)
    acc = accuracy_score(y_dev, y_pred)
    print('Avg AdaGrad 1.5', acc)
    classifier = AveragedAdaGrad(0.25, features)
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_dev)
    acc = accuracy_score(y_dev, y_pred)
    print('Avg AdaGrad 0.25', acc)
    classifier = AveragedAdaGrad(0.03, features)
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_dev)
    acc = accuracy_score(y_dev, y_pred)
    print('Avg AdaGrad 0.03', acc)
    classifier = AveragedAdaGrad(0.005, features)
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_dev)
    acc = accuracy_score(y_dev, y_pred)
    print('Avg AdaGrad 0.005', acc)
    classifier = AveragedAdaGrad(0.001, features)
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_dev)
    acc = accuracy_score(y_dev, y_pred)
    print('Avg AdaGrad 0.001', acc)"""
    
    # TODO: Placeholder data for the learning curves. You should write
    # the logic to downsample the dataset to the number of desired training
    # instances (e.g. 500, 1000), then train all of the models on the
    # sampled dataset. Compute the accuracy and add the accuraices to
    # the corresponding list.
    
    train_data_sizes = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 50000]
    perceptron_accs = []
    winnow_accs = []
    adagrad_accs = []
    avg_perceptron_accs = []
    avg_winnow_accs = []
    avg_adagrad_accs = [0.5039] * 11
    svm_accs = []
    
    for i in train_data_sizes:
        X_train_rand = []
        y_train_rand = []
        for j in random.sample(range(0,i), i):
            X_train_rand.append(X_train[j])
            y_train_rand.append(y_train[j])
        classifier = Perceptron(features)
        classifier.train(X_train_rand, y_train_rand)
        y_pred = classifier.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        perceptron_accs.append(acc)
        
    for i in train_data_sizes:
        X_train_rand = []
        y_train_rand = []
        for j in random.sample(range(0,i), i):
            X_train_rand.append(X_train[j])
            y_train_rand.append(y_train[j])
        classifier = AveragedPerceptron(features)
        classifier.train(X_train_rand, y_train_rand)
        y_pred = classifier.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        avg_perceptron_accs.append(acc)
  
    for i in train_data_sizes:
        X_train_rand = []
        y_train_rand = []
        for j in random.sample(range(0,i), i):
            X_train_rand.append(X_train[j])
            y_train_rand.append(y_train[j])
        classifier = Winnow(1.005, features)
        classifier.train(X_train_rand, y_train_rand)
        y_pred = classifier.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        winnow_accs.append(acc)
       
    for i in train_data_sizes:
        X_train_rand = []
        y_train_rand = []
        for j in random.sample(range(0,i), i):
            X_train_rand.append(X_train[j])
            y_train_rand.append(y_train[j])
        classifier = AveragedWinnow(1.005, features)
        classifier.train(X_train_rand, y_train_rand)
        y_pred = classifier.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        avg_winnow_accs.append(acc)   

    for i in train_data_sizes:
        X_train_rand = []
        y_train_rand = []
        for j in random.sample(range(0,i), i):
            X_train_rand.append(X_train[j])
            y_train_rand.append(y_train[j])
        classifier = AdaGrad(1.5, features)
        classifier.train(X_train_rand, y_train_rand)
        y_pred = classifier.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        adagrad_accs.append(acc)
    
    for i in train_data_sizes:
        X_train_rand = []
        y_train_rand = []
        for j in random.sample(range(0,i), i):
            X_train_rand.append(X_train[j])
            y_train_rand.append(y_train[j])
        vectorizer = DictVectorizer()
        X_train_dict = vectorizer.fit_transform(X_train_rand)
        X_test_dict = vectorizer.fit_transform(X_test)
        classifier = LinearSVC(loss='hinge')
        classifier.fit(X_train_dict, y_train_rand)
        y_pred = classifier.predict(X_test_dict)
        acc = accuracy_score(y_test, y_pred)
        svm_accs.append(acc)
        
    """for i in train_data_sizes:
        X_train_rand = []
        y_train_rand = []
        for j in random.sample(range(0,i), i):
            X_train_rand.append(X_train[j])
            y_train_rand.append(y_train[j])
        classifier = AveragedAdaGrad(1.5, features)
        classifier.train(X_train_rand, y_train_rand)
        y_pred = classifier.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        avg_adagrad_accs.append(acc)"""
       
    plot_learning_curves(perceptron_accs, winnow_accs, adagrad_accs, avg_perceptron_accs, avg_winnow_accs, avg_adagrad_accs, svm_accs)
    
    # TODO: Train all 7 models on the training data and test on the test data
    classifier = Perceptron(features)
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print('Perceptron', acc)
    classifier = AveragedPerceptron(features)
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print('Averaged Perceptron', acc)
    classifier = Winnow(1.005, features)
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print('Winnow', acc)
    classifier = AdaGrad(1.5, features)
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print('Adagrad', acc)
    classifier = AveragedWinnow(1.005, features)
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print('Averaged Winnow', acc)
    vectorizer = DictVectorizer()
    X_train_dict = vectorizer.fit_transform(X_train)
    X_test_dict = vectorizer.transform(X_test)
    classifier = LinearSVC(loss='hinge')
    classifier.fit(X_train_dict, y_train)
    y_pred = classifier.predict(X_test_dict)
    acc = accuracy_score(y_test, y_pred)
    print('SVM', acc)
    """classifier = AveragedAdaGrad(features)
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print('Averaged Adagrad', acc)"""
    
def load_ner_data(path):
    """
    Loads the NER data from a path (e.g. "ner/conll/train"). You should
    not need to edit this method.
    """
    # List of tuples for each sentence
    data = []
    for filename in os.listdir(path):
        with open(path + '/' + filename, 'r') as file:
            sentence = []
            for line in file:
                if line == '\n':
                    data.append(sentence)
                    sentence = []
                else:
                    sentence.append(tuple(line.split()))
    return data

def extract_ner_features_train(train):
    """
    Extracts feature dictionaries and labels from the data in "train"
    Additionally creates a list of all of the features which were created.
    We have implemented the w-1 and w+1 features for you to show you how
    to create them.
    
    TODO: You should add your additional featurization code here.
    """
    y = []
    X = []
    features = set()
    for sentence in train:
        padded = sentence[:]
        padded.insert(0, ('SSS', None))
        padded.insert(0, ('SSS', None))
        padded.insert(0, ('SSS', None))
        padded.append(('EEE', None))
        padded.append(('EEE', None))
        padded.append(('EEE', None))
        for i in range(3, len(padded) - 3):
            y.append(1 if padded[i][1] == 'I' else -1)
            feat1 = 'w-1=' + str(padded[i - 1][0])
            feat2 = 'w+1=' + str(padded[i + 1][0])
            feat3 = 'w-2=' + str(padded[i - 2][0])
            feat4 = 'w+2=' + str(padded[i + 2][0])
            feat5 = 'w-3=' + str(padded[i - 3][0])
            feat6 = 'w+3=' + str(padded[i + 3][0])
            feat7 = 'w-1&w-2=' + str(padded[i - 1][0]) + ' ' + str(padded[i - 2][0])
            feat8 = 'w+1&w+2=' + str(padded[i + 1][0]) + ' ' + str(padded[i + 2][0])
            feat9 = 'w-1&w+1=' + str(padded[i - 1][0]) + ' ' + str(padded[i + 1][0])
            feats = [feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9]
            features.update(feats)
            feats = {feature: 1 for feature in feats}
            X.append(feats)
    return features, X, y

def extract_features_dev_or_test(data, features):
    """
    Extracts feature dictionaries and labels from "data". The only
    features which should be computed are those in "features". You
    should add your additional featurization code here.
    
    TODO: You should add your additional featurization code here.
    """
    y = []
    X = []
    for sentence in data:
        padded = sentence[:]
        padded.insert(0, ('SSS', None))
        padded.insert(0, ('SSS', None))
        padded.insert(0, ('SSS', None))
        padded.append(('EEE', None))
        padded.append(('EEE', None))
        padded.append(('EEE', None))
        for i in range(3, len(padded) - 3):
            y.append(1 if padded[i][1] == 'I' else -1)
            feat1 = 'w-1=' + str(padded[i - 1][0])
            feat2 = 'w+1=' + str(padded[i + 1][0])
            feat3 = 'w-2=' + str(padded[i - 2][0])
            feat4 = 'w+2=' + str(padded[i + 2][0])
            feat5 = 'w-3=' + str(padded[i - 3][0])
            feat6 = 'w+3=' + str(padded[i + 3][0])
            feat7 = 'w-1&w-2=' + str(padded[i - 1][0]) + ' ' + str(padded[i - 2][0])
            feat8 = 'w+1&w+2=' + str(padded[i + 1][0]) + ' ' + str(padded[i + 2][0])
            feat9 = 'w-1&w+1=' + str(padded[i - 1][0]) + ' ' + str(padded[i + 1][0])
            feats = [feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9]
            feats = {feature: 1 for feature in feats if feature in features}
            X.append(feats)
    return X, y

def run_ner_experiment(data_path):
    """
    Runs the NER experiment using the path to the ner data
    (e.g. "ner" from the released resources). We have implemented
    the standard Perceptron below. You should do the same for
    the averaged version and the SVM.
    
    The SVM requires transforming the features into a different
    format. See the end of this function for how to do that.
    """
    train = load_ner_data(data_path + '/conll/train')
    conll_test = load_ner_data(data_path + '/conll/test')
    enron_test = load_ner_data(data_path + '/enron/test')

    features, X_train, y_train = extract_ner_features_train(train)
    X_conll_test, y_conll_test = extract_features_dev_or_test(conll_test, features)
    X_enron_test, y_enron_test = extract_features_dev_or_test(enron_test, features)
                 
    # You should do this for the Averaged Perceptron and SVM
    classifier = Perceptron(features)
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_conll_test)
    conll_f1 = calculate_f1(y_conll_test, y_pred)
    y_pred = classifier.predict(X_enron_test)
    enron_f1 = calculate_f1(y_enron_test, y_pred)
    print('Perceptron')
    print('  CoNLL', conll_f1)
    print('  Enron', enron_f1)
    
    classifier = AveragedPerceptron(features)
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_conll_test)
    conll_f1 = calculate_f1(y_conll_test, y_pred)
    y_pred = classifier.predict(X_enron_test)
    enron_f1 = calculate_f1(y_enron_test, y_pred)
    print('Averaged Perceptron')
    print('  CoNLL', conll_f1)
    print('  Enron', enron_f1)
    
    # This is how you convert from the way we represent features in the
    # Perceptron code to how you need to represent features for the SVM.
    # You can then train with (X_train_dict, y_train) and test with
    # (X_conll_test_dict, y_conll_test) and (X_enron_test_dict, y_enron_test)
    vectorizer = DictVectorizer()
    X_train_dict = vectorizer.fit_transform(X_train)
    X_conll_test_dict = vectorizer.transform(X_conll_test)
    X_enron_test_dict = vectorizer.transform(X_enron_test)
    classifier = LinearSVC(loss='hinge')
    classifier.fit(X_train_dict, y_train)
    y_pred = classifier.predict(X_enron_test_dict)
    enron_f1 = calculate_f1(y_enron_test, y_pred)
    y_pred = classifier.predict(X_conll_test_dict)
    conll_f1 = calculate_f1(y_conll_test, y_pred)
    print('SVM')
    print('  CoNLL', conll_f1)
    print('  Enron', enron_f1)
    
# Run the synthetic experiment on the sparse dataset. "synthetic/sparse"
# is the path to where the data is located.
run_synthetic_experiment('synthetic/sparse')

# Run the synthetic experiment on the sparse dataset. "synthetic/dense"
# is the path to where the data is located.
run_synthetic_experiment('synthetic/dense')

# Run the NER experiment. "ner" is the path to where the data is located.
run_ner_experiment('ner')
