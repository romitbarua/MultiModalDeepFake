import pandas as pd
import numpy as np
import os
import random
from tqdm import tqdm
from pathlib import Path
import opensmile
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
import matplotlib.pyplot as plt
import sklearn.feature_selection as fs
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_regression, SequentialFeatureSelector, GenericUnivariateSelect
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn import svm

#base_path
base_path = "/home/ubuntu"

############################################################################################
# Initialization ###########################################################################
############################################################################################

class smileFeatureSelectorBase:

    def __init__(self, df, 
                 metadata=['id', 'path', 'split', 'type', 'label', 'multiclass_label', 'duration(seconds)'], 
                 standardize: bool = True) -> None:

        print('Initializing data...')
        
        self.data = df
        self.metadata = metadata
        self.all_features = self.data.drop(columns=self.metadata).columns
        self.architectures = list(self.data.type.unique())

        #do we need the test data?

        self.train_df = self.data[self.data['split'] == 'train'].copy()
        self.dev_df = self.data[self.data['split'] == 'dev'].copy()
        self.test_df = self.data[self.data['split'] == 'test'].copy()

        ## standardize the features inside the train, dev, and test sets for the cols_to_scale columns

        if standardize:
            print('Standardizing features...')
            cols_to_scale = list(self.all_features)
            scaler = StandardScaler()
            scaler.fit(self.train_df[cols_to_scale])
            self.train_df.loc[:, cols_to_scale] = scaler.transform(self.train_df.loc[:, cols_to_scale])
            self.dev_df.loc[:, cols_to_scale] = scaler.transform(self.dev_df.loc[:, cols_to_scale])
            self.test_df.loc[:, cols_to_scale] = scaler.transform(self.test_df.loc[:, cols_to_scale])
            self.scaler = scaler
        else:
            self.scaler = None
        
        #print('smileFeatureSelector object initialized.\n')

############################################################################################
# Brute Force Feature Selection ############################################################
############################################################################################
class smileFeatureSelectorBruteForce(smileFeatureSelectorBase):

    def __init__(self, df, 
                 metadata=['id', 'path', 'split', 'type', 'label', 'multiclass_label', 'duration(seconds)'], 
                 standardize: bool = True,
                 model=LogisticRegression()):
        """
        Initialize the smileFeatureSelectorBruteForce class.
        """
        #initialize the base class
        super().__init__(df, metadata, standardize)
        #load the model to use for brute force feature selection
        self.bffs_model = model
        print('smileFeatureSelectorBruteForce object initialized.\n')

    # ... (rest of the methods in smileFeatureSelectorBruteForce)
    def generate_bffs_data(self, archs = ['ElevenLabs', 'Waveglow', 'Parallel_WaveGan', 'Multi_Band_MelGan', 
                                    'MelGanLarge', 'MelGan', 'HifiGan', 'Full_Band_MelGan']):
        """
        Generate data for brute force feature selection
        """
        #store the architectures to run bffs on
        self.bffs_archs = archs

        #create a bffs dataframe to store the results
        self.bffs_data = pd.DataFrame(self.all_features, columns=['features'])

        #iterate through each architecture and run bffs
        for arch in self.bffs_archs:
            print("\nRunning for {} architecture\n".format(arch))
            self.bffs_data[arch] = self._run_bffs(arch)

        #run bffs for all architectures
        print("\nRunning for all architectures\n")
        self.bffs_data['all_archs'] = self._run_bffs(self.bffs_archs)

        print("\nBrute force feature selection data generated.\nAll data stored in self.bffs_data.\n")

    def _run_bffs(self, arch):
        """
        Runs brute force feature selection
        """
        #checks to see if arch is a list of architectures
        if isinstance(arch, list):
            trdf = self.train_df[self.train_df.type.isin(arch)]
            dvdf = self.dev_df[self.dev_df.type.isin(arch)]
        #for each individual architecture
        else:
            trdf = self.train_df[self.train_df.type==arch]
            dvdf = self.dev_df[self.dev_df.type==arch]
        
        #split train data into X and y
        X_train = trdf.drop(columns=self.metadata).copy()
        y_train = trdf['label'].copy()
        
        #split dev data into X and y
        X_dev = dvdf.drop(columns=self.metadata).copy()
        y_dev = dvdf['label'].copy()

        #store the dev accuracies for each feature
        dev_accuracies = []
        
        #run bffs
        for i in tqdm(range(len(self.all_features))):
            
            model = self.bffs_model
            model.fit(X_train.iloc[:,i].to_numpy().reshape(-1, 1), y_train)
            y_hat_train = model.predict(X_train.iloc[:,i].to_numpy().reshape(-1, 1))
            y_hat_dev = model.predict(X_dev.iloc[:,i].to_numpy().reshape(-1, 1))        
            dev_accuracy = accuracy_score(y_dev, y_hat_dev)
            dev_accuracies.append(dev_accuracy)
        
        return dev_accuracies

    def select_bffs_features(self, num_features=10, arch='all_archs', return_df=False):
        """
        Selects the top num_features features from the bffs_data dataframe
        """
        #sort the bffs_data dataframe by the specified architecture
        sorted_df = self.bffs_data.sort_values(by=arch, ascending=False)
        #select the top num_features
        self.bffs_features = list(sorted_df.features[:num_features])
        print("\nTop {} features selected from {} architecture.\n".format(num_features, arch))
        for count, item in enumerate(self.bffs_features):
            print("{}. {}".format(count+1, item))

        if return_df:
            return self.data[self.data.columns.intersection(self.metadata + self.bffs_features)]
    
############################################################################################
# Feature Selection From Model #############################################################
############################################################################################
class smileFeatureSelectFromModel(smileFeatureSelectorBase):

    def __init__(self, df, 
                 metadata=['id', 'path', 'split', 'type', 'label', 'multiclass_label', 'duration(seconds)'], 
                 standardize: bool = True,
                 model=RandomForestClassifier()):
        """
        Initialize the smileFeatureSelectorBruteForce class.
        """
        #initialize the base class
        super().__init__(df, metadata, standardize)
        #load the model to use for brute force feature selection
        self.model = model
        print('smileFeatureSelectFromModel object initialized.\n')

    # ... (rest of the methods in smileFeatureSelectFromModel)
    def select_features(self, max_features=10, arch='all_archs', return_df=False):
        """
        Selects the top num_features features based on the model specified
        """
        #for architectures
        if not arch=='all_archs':
            #checks to see if arch is a list of architectures
            if isinstance(arch, list):
                trdf = self.train_df[self.train_df.type.isin(arch)]
                dvdf = self.dev_df[self.dev_df.type.isin(arch)]
            #for each individual architecture
            else:
                trdf = self.train_df[self.train_df.type==arch]
                dvdf = self.dev_df[self.dev_df.type==arch]
        else:
            trdf = self.train_df
            dvdf = self.dev_df
            
        #split train data into X and y
        X_train = trdf.drop(columns=self.metadata).copy()
        y_train = trdf['label'].copy()
        
        #split dev data into X and y
        X_dev = dvdf.drop(columns=self.metadata).copy()
        y_dev = dvdf['label'].copy()
        
        #instantiating the model and fitting it
        self.sfm_model = SelectFromModel(self.model, max_features=max_features)
        self.sfm_model.fit(X_train, y_train)

        #getting the selected features
        self.sfm_features = list(X_train.columns[self.sfm_model.get_support()])
        print("\nTop {} features selected from {} architecture.\n".format(max_features, arch))
        for count, item in enumerate(self.sfm_features):
            print("{}. {}".format(count+1, item))

        if return_df:
            return self.data[self.data.columns.intersection(self.metadata + self.sfm_features)]

############################################################################################
# Univariate Feature Selection #############################################################
############################################################################################
class smileUnivariateFeatureSelector(smileFeatureSelectorBase):

    def __init__(self, df, 
                 metadata=['id', 'path', 'split', 'type', 'label', 'multiclass_label', 'duration(seconds)'], 
                 standardize: bool = True):
        """
        Initialize the smileUnivariateFeatureSelector class.
        """
        #initialize the base class
        super().__init__(df, metadata, standardize)
        
        print('smileUnivariateFeatureSelector object initialized.\n')

    # ... (rest of the methods in smileUnivariateFeatureSelector)
    def select_features(self, score_func=fs.mutual_info_classif, mode='k_best',
                        num_features=10, arch='all_archs', return_df=False):
        """ 
        Selects the top num_features features based on the score_func specified
        using univariate feature selection
        """
        
        #for architectures
        if not arch=='all_archs':
            #checks to see if arch is a list of architectures
            if isinstance(arch, list):
                trdf = self.train_df[self.train_df.type.isin(arch)]
                dvdf = self.dev_df[self.dev_df.type.isin(arch)]
            #for each individual architecture
            else:
                trdf = self.train_df[self.train_df.type==arch]
                dvdf = self.dev_df[self.dev_df.type==arch]
        else:
            trdf = self.train_df
            dvdf = self.dev_df
                
        #split train data into X and y
        X_train = trdf.drop(columns=self.metadata).copy()
        y_train = trdf['label'].copy()
        
        #split dev data into X and y
        X_dev = dvdf.drop(columns=self.metadata).copy()
        y_dev = dvdf['label'].copy()
        
        #instantiating the model and fitting it
        self.univariate_model = GenericUnivariateSelect(score_func=score_func, mode=mode, 
                                                        param=num_features)
        self.univariate_model.fit(X_train, y_train)

        #getting the selected features
        self.univariate_features = list(X_train.columns[self.univariate_model.get_support()])
        print("\nTop {} features selected from {} architecture.\n".format(num_features, arch))
        for count, item in enumerate(self.univariate_features):
            print("{}. {}".format(count+1, item))

        if return_df:
            return self.data[self.data.columns.intersection(self.metadata + self.univariate_features)]