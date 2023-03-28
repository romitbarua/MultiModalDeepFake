from heapq import merge
import sklearn
from sklearn.metrics import accuracy_score, log_loss, roc_curve
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

VALID_MODELS = ['svn', 'logreg', 'kmeans']

class ModelManager:

    def __init__(self, model_name, data, merge_train_dev: bool = False):

        self.model_name = model_name
        self._splitDataframe(data, merge_train_dev=merge_train_dev)

    def init_model(self, params=None):

        assert self.model_name.lower() in VALID_MODELS, f'{self.model_name} is not valid. Valid models include {VALID_MODELS}'

        if self.model_name == 'svm':
            if params is None:
                self.model = SVC()
            else:
                self.model = SVC(**params)
        elif self.model_name == 'logreg':
            if params is None:
                self.model = LogisticRegression()
            else:
                self.model = LogisticRegression(**params)
        elif self.model_name == 'kmeans':
            if params is None:
                self.model = KMeans()
            else:
                self.model = KMeans(**params)

    def _splitDataframe(self, merge_train_dev: bool ):

        if merge_train_dev:
            self.train = self.data[(self.data.type == 'train') | (self.data.type == 'dev')]
            self.dev = None
        else:    
            self.train = self.data[(self.data.type == 'train')]
            self.dev = self.data[(self.data.type == 'dev')]

        self.test = self.data[(self.data.type == 'test')]         

    def trainModel(self, label_col: str):
        # Train the model using the training data
        y_train = self.train[label_col]
        X_train = self.train.drop(columns=label_col).copy()
        self.model.fit(X_train, y_train)

    def predict(self, label_col: str):
        # Make predictions on the test data
        y_test = self.test[label_col]
        X_test = self.test.drop(columns=label_col).copy()
        self.y_pred = self.model.predict(X_test)
        self.y_prob = self.model.predict_proba(X_test)[:, 1]

        # Calculate accuracy and log loss
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.log_loss_value = log_loss(self.y_test, self.y_prob)

        return self.accuracy, self.log_loss_value

    def trainPredict(self):
        self.trainModel()
        acc, log_loss = self.predict()
        return acc, log_loss

    def plotRocCurve(self,):
        # Create a ROC curve plot
        fpr, tpr, _ = roc_curve(self.y_test, self.y_prob)
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()

    def plotProbaDistribution(self):
        # Create a histogram of test set probability scores
        plt.hist(self.y_prob)
        plt.xlabel('Probability Score')
        plt.ylabel('Frequency')
        plt.title('Test Set Probability Score Distribution')
        plt.show()
