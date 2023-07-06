
from packages.SavedFeatureLoader import loadFeatures
from packages.SmileFeatureSelector import *
from sklearn.ensemble import RandomForestClassifier

VALID_FEATURE_SELECTORS = ['random_forest']

class SmileFeatureManager:

    def __init__(self, data) -> None:
        self.data = data
        self.metadata_cols = data.columns
        self.loadSavedFeatures()

    def loadSavedFeatures(self):

        self.feature_df = loadFeatures(self.data.copy(), 'openSmile',  feature_filepath_col='file')
        #GK edit 06/02/2023 to drop duration column
        self.feature_df = self.feature_df.drop(columns=['duration(seconds)'])

    def generateFeatureDf(self, feature_selector_type, label_type, feature_count=10):
        
        assert feature_selector_type in VALID_FEATURE_SELECTORS, f'{feature_selector_type} not valid. Valid types include {VALID_FEATURE_SELECTORS}'
        assert label_type in ['binary', 'multiclass'], 'Label type must be either binary or multiclass'

        if feature_selector_type == 'random_forest':
            #GK edit 06/29/23 to set random state
            selector = smileFeatureSelectFrmModel(self.feature_df, metadata=list(self.metadata_cols), model=RandomForestClassifier(random_state=12))

            if label_type == 'binary':
                df =  selector.select_features_binary(max_features=feature_count, return_df=True)
            else:
                df = selector.select_features_multiclass(max_features=feature_count, return_df=True) 
                
            return df


    

        





from packages.SavedFeatureLoader import loadFeatures
from packages.SmileFeatureSelector import *
from sklearn.ensemble import RandomForestClassifier

VALID_FEATURE_SELECTORS = ['random_forest']

class SmileFeatureManager:

    def __init__(self, data) -> None:
        self.data = data
        self.metadata_cols = data.columns
        self.loadSavedFeatures()

    def loadSavedFeatures(self):

        self.feature_df = loadFeatures(self.data.copy(), 'openSmile',  feature_filepath_col='file')
        #GK edit 06/02/2023 to drop duration column
        self.feature_df = self.feature_df.drop(columns=['duration(seconds)'])

    def generateFeatureDf(self, feature_selector_type, label_type, feature_count=10):
        
        assert feature_selector_type in VALID_FEATURE_SELECTORS, f'{feature_selector_type} not valid. Valid types include {VALID_FEATURE_SELECTORS}'
        assert label_type in ['binary', 'multiclass'], 'Label type must be either binary or multiclass'

        if feature_selector_type == 'random_forest':
            #GK edit 06/29/23 to set random state
            selector = smileFeatureSelectFromModel(self.feature_df, metadata=list(self.metadata_cols), model=RandomForestClassifier(random_state=12))

            if label_type == 'binary':
                df =  selector.select_features_binary(max_features=feature_count, return_df=True)
            else:
                df = selector.select_features_multiclass(max_features=feature_count, return_df=True) 
                
            return df


    

        




