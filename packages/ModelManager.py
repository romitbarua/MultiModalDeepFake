from heapq import merge
import sklearn
from sklearn.metrics import accuracy_score, log_loss, roc_curve
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

VALID_MODELS = ['svm', 'logreg', 'knn']

class ModelManager:

    def __init__(self, model_name, data, feature_cols, merge_train_dev: bool = False):

        self.model_name = model_name
        self.data = data
        self._splitDataframe(merge_train_dev=merge_train_dev)
        self.init_model()
        self.feature_cols = feature_cols

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
        elif self.model_name == 'knn':
            if params is None:
                self.model = KNeighborsClassifier()
            else:
                self.model = KNeighborsClassifier(**params)

    def _splitDataframe(self, merge_train_dev: bool):

        if merge_train_dev:
            self.train = self.data[(self.data.type == 'train') | (self.data.type == 'dev')]
            self.dev = None
        else:    
            self.train = self.data[(self.data.type == 'train')]
            self.dev = self.data[(self.data.type == 'dev')]

        self.test = self.data[(self.data.type == 'test')]         
        
        ## TEMP SOLUTION
        self.train = self.train.dropna()
        self.test = self.test.dropna()

    def trainModel(self, label_col: str):
        # Train the model using the training data
        self.y_train = self.train[label_col]
        self.X_train = self.train[self.feature_cols].copy()

        self.model.fit(self.X_train, self.y_train)

    def predict(self, label_col: str):
        # Make predictions on the test data
        self.y_test = self.test[label_col]
        self.X_test = self.test[self.feature_cols].copy()
        
        self.y_pred = self.model.predict(self.X_test)
        

        # Calculate accuracy and log loss
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        
        if self.model_name not in ['svm']:
            self.y_prob = self.model.predict_proba(self.X_test)
            self.log_loss_value = log_loss(self.y_test, self.y_prob)

            return self.accuracy, self.log_loss_value
        
        return self.accuracy, None

    def trainPredict(self, label_col: str):
        self.trainModel(label_col=label_col)
        acc, log_loss = self.predict(label_col=label_col)
        return acc, log_loss

    def plotRocCurve(self):
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
