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
import time
import random

#base_path
base_path = "/home/ubuntu/"

############################################################################################
# Functions ################################################################################
############################################################################################


def process_data_for_all_archs(df, archs):
    
    # create example list
    id_list = list(df.id.unique())
    
    num_archs = len(archs)
    
    subsets = {}
    
    for i, arch in enumerate(archs[:-1]):
        
        subset=random.sample(id_list, k=len(id_list)//(num_archs-i))
        subsets[arch]=subset
        id_list = [x for x in id_list if x not in subsets[arch]]
    
    subsets[archs[-1]] = id_list
        
    return subsets


############################################################################################
# Initialization ###########################################################################
############################################################################################

class smileFeatureSelectorBase:

    def __init__(self, df, 
                 metadata=['type', 'id', 'architecture', 'arch_filter', 'path', 'label', 'multiclass_label', 'duration(seconds)'], 
                 standardize: bool = True) -> None:

        print('Initializing data...')
        
        self.data = df
        self.metadata = metadata
        self.all_features = self.data.drop(columns=self.metadata).columns

        #do we need the test data?

        self.train_df = self.data[self.data['type'] == 'train'].copy()
        self.dev_df = self.data[self.data['type'] == 'dev'].copy()
        self.test_df = self.data[self.data['type'] == 'test'].copy()

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
                 metadata=['type', 'id', 'architecture', 'arch_filter', 'path', 'label', 'multiclass_label', 'duration(seconds)'],
                 real_col='Real',
                 fake_cols=['ElevenLabs', 'UberDuck', 'RandWaveFake'], 
                 standardize: bool = True,
                 model=LogisticRegression()):
        """
        Initialize the smileFeatureSelectorBruteForce class.
        """
        #initialize the base class
        super().__init__(df, metadata, standardize)
        #load the model to use for brute force feature selection
        self.bffs_model = model
        self.real_col = real_col
        self.fake_cols = fake_cols
        print('smileFeatureSelectorBruteForce object initialized.\n')

    # ... (rest of the methods in smileFeatureSelectorBruteForce)
    def generate_data(self):
        """
        Generate data for brute force feature selection
        """
        print('Generating brute force feature selection data...')
        #create a bffs dataframe to store the results
        self.bffs_data = pd.DataFrame(self.all_features, columns=['features'])
        
        for col in self.fake_cols:
            
            train_df = pd.concat((self.train_df[self.train_df.architecture==self.real_col], self.train_df[self.train_df.architecture==col]))
            dev_df = pd.concat((self.dev_df[self.dev_df.architecture==self.real_col], self.dev_df[self.dev_df.architecture==col]))

            #run bffs
            self.bffs_data[col] = self._run_bffs(train_df, dev_df, mode='binary')

        self.bffs_data['average_binary'] = self.bffs_data[self.fake_cols].mean(axis=1)

        #for all archs
        #this is based on the approach in the notebook where we filter out archs and assign NA to remaining rows during merge
        train_df = self.train_df[self.train_df.arch_filter.notna()]
        dev_df = self.dev_df[self.dev_df.arch_filter.notna()]
        #run bffs
        self.bffs_data['all_archs'] = self._run_bffs(train_df, dev_df, mode='binary')

        self.bffs_data['multiclass'] = self._run_bffs(self.train_df, self.dev_df, mode='multiclass')

        print("\nBrute force feature selection data generated.\nAll data stored in self.bffs_data.\n")

    def _run_bffs(self, train_df, dev_df, mode='binary'):
        """
        Runs brute force feature selection
        """
        #split train data into X and y
        X_train = train_df.drop(columns=self.metadata).copy()
        if mode=='binary':
            y_train = train_df['label'].copy()
        elif mode=='multiclass':
            y_train = train_df['multiclass_label'].copy()
        
        #split dev data into X and y
        X_dev = dev_df.drop(columns=self.metadata).copy()
        if mode=='binary':
            y_dev = dev_df['label'].copy()
        elif mode=='multiclass':
            y_dev = dev_df['multiclass_label'].copy()

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

    def FS_from_col(self, num_features=10, sort_col='average_binary', return_df=False, print_features=True, return_features=False):
        """
        Selects the top num_features features from the bffs_data dataframe
        """
        #sort the bffs_data dataframe by the specified architecture
        sorted_df = self.bffs_data.sort_values(by=sort_col, ascending=False)
        #select the top num_features
        bffs_features = list(sorted_df.features.iloc[:num_features])

        if print_features:
            print("\nTop {} features selected from {} architecture.\n".format(num_features, sort_col))
            for count, item in enumerate(bffs_features):
                print("{}. {}".format(count+1, item))

        if return_df:
            return self.data[self.data.columns.intersection(self.metadata + bffs_features)]
        
        if return_features:
            return bffs_features
    
    def FS_top_n_from_all_archs(self, n=10, return_df=False, return_features=False, print_features=True):
        """
        Selects the top num_features features from the bffs_data dataframe
        """

        feature_set = set()
        for col in self.fake_cols:
            feature_set.update(set(self.FS_from_col(num_features=n, sort_col=col, return_features=True, print_features=False)))
        
        if print_features:
            print("\nTop features selected from all architectures.\n")
            for count, item in enumerate(feature_set):
                print("{}. {}".format(count+1, item))
        
        if return_df:
            return self.data[self.data.columns.intersection(self.metadata + list(feature_set))]
        
        if return_features:
            return list(feature_set)

    
############################################################################################
# Feature Selection From Model #############################################################
############################################################################################
class smileFeatureSelectFromModel(smileFeatureSelectorBase):

    def __init__(self, df, 
                 metadata=['type', 'id', 'architecture', 'arch_filter', 'path', 'label', 'multiclass_label', 'duration(seconds)'], 
                 real_col='Real',
                 fake_cols=['ElevenLabs', 'UberDuck', 'RandWaveFake'], 
                 standardize: bool = True,
                 model=RandomForestClassifier()):
        """
        Initialize the smileFeatureSelectorBruteForce class.
        """
        #initialize the base class
        super().__init__(df, metadata, standardize)

        self.real_col=real_col
        self.fake_cols=fake_cols

        #load the model to use for brute force feature selection
        self.model = model
        print('smileFeatureSelectFromModel object initialized.\n')

    # ... (rest of the methods in smileFeatureSelectFromModel)
    def select_features_binary(self, max_features=10, archs='all_archs', return_df=False, print_features=True, return_features=False):
        """
        Selects the top num_features features based on the model specified
        """
        #for architectures
        if not archs=='all_archs':
            #checks to see if archs is a list of architectures
            if isinstance(archs, list):
                
                #alternate approach that treats each architecture separately while selecting features
                #self.binary_feature_set = set()
                #for arch in archs:
                    #trdf = pd.concat((self.train_df[self.train_df.arch  ==self.real_col], self.train_df[self.train_df.architecture==arch]))
                    #dvdf = pd.concat((self.dev_df[self.dev_df.architecture==self.real_col], self.dev_df[self.dev_df.architecture==arch]))
                    #sfm_features = self._run_sfm(trdf, dvdf, max_features_per_arch)
                    #self.binary_feature_set.update(set(sfm_features))
                
                #approach that treats all architectures together while selecting features
                trdf = self.train_df[self.train_df.arch_filter.isin(archs)]
                dvdf = self.dev_df[self.dev_df.arch_filter.isin(archs)]
                sfm_features = self._run_sfm(trdf, dvdf, max_features)
                self.binary_feature_set = set(sfm_features)

            #for each individual architecture
            else:
                trdf = pd.concat((self.train_df[self.train_df.architecture==self.real_col], self.train_df[self.train_df.architecture==archs]))
                dvdf = pd.concat((self.dev_df[self.dev_df.architecture==self.real_col], self.dev_df[self.dev_df.architecture==archs]))

                sfm_features = self._run_sfm(trdf, dvdf, max_features)
                self.binary_feature_set = set(sfm_features)
        
        #for all architectures
        else:
            
            #approach that treats all architectures together while selecting features
            trdf = self.train_df[self.train_df.arch_filter.isin(self.fake_cols)]
            dvdf = self.dev_df[self.dev_df.arch_filter.isin(self.fake_cols)]
            sfm_features = self._run_sfm(trdf, dvdf, max_features)
            self.binary_feature_set = set(sfm_features)
            
        if print_features:
            print("\nSelected features:.\n")
            for count, item in enumerate(self.binary_feature_set):
                print("{}. {}".format(count+1, item))

        if return_features:
            return list(self.binary_feature_set)

        if return_df:
            return self.data[self.data.columns.intersection(self.metadata + list(self.binary_feature_set))]
        
    def select_features_multiclass(self, max_features=10, archs='all_archs', return_df=False, print_features=True, return_features=False):

        #for architectures
        if not archs=='all_archs':
            #checks to see if archs is a list of architectures
            if isinstance(archs, list):

                #approach that treats all architectures together while selecting features
                trdf = self.train_df[self.train_df.arch_filter.isin(archs)]
                dvdf = self.dev_df[self.dev_df.arch_filter.isin(archs)]
                sfm_features = self._run_sfm(trdf, dvdf, max_features, multiclass=True)
                self.multiclass_feature_set = set(sfm_features)   

                sfm_features = self._run_sfm(self.train_df, self.dev_df, max_features, multiclass=True)
                self.multiclass_feature_set = set(sfm_features)
            
            else:
                print('Please specify a list of architectures...')

        #for all architectures
        else:

            sfm_features = self._run_sfm(self.train_df, self.dev_df, max_features, multiclass=True)
            self.multiclass_feature_set = set(sfm_features)
            
        if print_features:
            print("\nSelected features:.\n")
            for count, item in enumerate(self.multiclass_feature_set):
                print("{}. {}".format(count+1, item))

        if return_features:
            return list(self.multiclass_feature_set)

        if return_df:
            return self.data[self.data.columns.intersection(self.metadata + list(self.multiclass_feature_set))]


    
    def _run_sfm(self, trdf, dvdf, max_features, multiclass=False):

        #split train data into X and y
        X_train = trdf.drop(columns=self.metadata).copy()
        if multiclass:
            y_train = trdf['multiclass_label'].copy()
        else:
            y_train = trdf['label'].copy()
        
        #split dev data into X and y
        #X_dev = dvdf.drop(columns=self.metadata).copy()
        #if multiclass:
            #y_dev = dvdf['multiclass_label'].copy()
        #else:
            #y_dev = dvdf['label'].copy()
        
        #instantiating the model and fitting it
        sfm_model = SelectFromModel(self.model, max_features=max_features)
        sfm_model.fit(X_train, y_train)

        #getting the selected features
        sfm_features = list(X_train.columns[sfm_model.get_support()])
        return sfm_features
        

############################################################################################
# Univariate Feature Selection #############################################################
############################################################################################

#running univariate statistical tests based on mutual information
#https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.GenericUnivariateSelect.html#sklearn.feature_selection.GenericUnivariateSelect

class smileUnivariateFeatureSelector(smileFeatureSelectorBase):

    def __init__(self, df, 
                 metadata=['type', 'id', 'architecture', 'arch_filter', 'path', 'label', 'multiclass_label', 'duration(seconds)'],
                 real_col='Real',
                 fake_cols=['ElevenLabs', 'UberDuck', 'RandWaveFake'], 
                 standardize: bool = True):
        """
        Initialize the smileUnivariateFeatureSelector class.
        """
        #initialize the base class
        super().__init__(df, metadata, standardize)

        self.real_col=real_col
        self.fake_cols=fake_cols
        
        print('smileUnivariateFeatureSelector object initialized.\n')

    # ... (rest of the methods in smileUnivariateFeatureSelector)
    def select_features(self, score_func=fs.mutual_info_classif, mode='k_best',
                        num_features=10, archs='all_archs', return_df=False, print_features=True, return_features=False):
        """ 
        Selects the top num_features features based on the score_func specified
        using univariate feature selection
        """
        
        #for architectures
        if not archs=='all_archs':
            #checks to see if arch is a list of architectures
            if isinstance(archs, list):

                #approach that treats all architectures together while selecting features
                trdf = self.train_df[self.train_df.arch_filter.isin(archs)]
                dvdf = self.dev_df[self.dev_df.arch_filter.isin(archs)]
                uvfs_features = self._run_uvfs(trdf, dvdf, num_features, score_func=score_func, mode=mode)
                self.univariate_features = set(uvfs_features)
            #for each individual architecture
            #for each individual architecture
            else:
                trdf = pd.concat((self.train_df[self.train_df.architecture==self.real_col], self.train_df[self.train_df.architecture==archs]))
                dvdf = pd.concat((self.dev_df[self.dev_df.architecture==self.real_col], self.dev_df[self.dev_df.architecture==archs]))

                uvfs_features = self._run_uvfs(trdf, dvdf, num_features, score_func=score_func, mode=mode)
                self.univariate_features = set(uvfs_features)
        
        #for all architectures
        else:
            
            #approach that treats all architectures together while selecting features
            trdf = self.train_df[self.train_df.arch_filter.isin(self.fake_cols)]
            dvdf = self.dev_df[self.dev_df.arch_filter.isin(self.fake_cols)]
            uvfs_features = self._run_uvfs(trdf, dvdf, num_features, score_func=score_func, mode=mode)
            self.univariate_features = set(uvfs_features)
        
        if print_features:
            print("\nSelected features:.\n")
            for count, item in enumerate(self.univariate_features):
                print("{}. {}".format(count+1, item))
        
        if return_features:
            return list(self.univariate_features)

        if return_df:
            return self.data[self.data.columns.intersection(self.metadata + list(self.univariate_feature))]
        
    def _run_uvfs(self, trdf, dvdf, num_features, score_func, mode):

        #split train data into X and y
        X_train = trdf.drop(columns=self.metadata).copy()
        y_train = trdf['label'].copy()
        
        #split dev data into X and y
        #X_dev = dvdf.drop(columns=self.metadata).copy()
        #y_dev = dvdf['label'].copy()
        
        #instantiating the model and fitting it
        uvfs_model = GenericUnivariateSelect(score_func=score_func, mode=mode, 
                                                        param=num_features)
        uvfs_model.fit(X_train, y_train)

        #getting the selected features
        univariate_features = list(X_train.columns[uvfs_model.get_support()])
        return univariate_features
