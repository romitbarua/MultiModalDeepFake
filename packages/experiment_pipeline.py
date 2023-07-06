import sys
import os
sys.path.append("/home/ubuntu/MultiModalDeepFake")
import nemo.collections.asr as nemo_asr 
import pandas as pd
import mlflow

from packages.LJDataLoader import LJDataLoader
from packages.AudioEmbeddingsManager import AudioEmbeddingsManager
from packages.ModelManager import ModelManager
from packages.CadenceModelManager import CadenceModelManager
import packages.AnalysisManager as am
from packages.SmileFeatureManager import SmileFeatureManager

class ExperimentPipeline:

    #################################################################################
    ################################# Initialization ################################
    #################################################################################

    def __init__(self, fake_cols, metadata_path, data_df=None) -> None:
        
        #intialize the class and generate the data for experiment pipeline if data is not provided
        self.fake_cols = fake_cols
        self.metadata_path = metadata_path
        if data_df is None:
            self.data_df = self._generate_split(self.fake_cols, self.metadata_path)
        else:
            self.data_df = data_df
        self.feature_store = {} 

    def _generate_split(self, fake_cols, metadata_path):
        
        #filter data used in training of elevenlabs and initialize the data loader
        loader = LJDataLoader(data_path=self.metadata_path, filter_cols=['ElevenLabsCloneClip'])
        
        loader.splitData()

        ## GK Comment: Review Code with Romit to see if this can be cleaned up and made more efficient
        source_architectures = ['Full_Band_MelGan', 'HifiGan', 'MelGan', 'MelGanLarge', 'Multi_Band_MelGan', 'Parallel_WaveGan', 'Waveglow']
        new_col_name = 'RandWaveFake'
        loader.selectRandomArchitecture(target_col=new_col_name, source_cols=source_architectures)
        
        source_architectures = ['ElevenLabs', 'UberDuck']
        new_col_name = 'EL_UD_Fake'
        loader.selectRandomArchitecture(target_col=new_col_name, source_cols=source_architectures)
        
        source_architectures = ['RandWaveFake', 'ElevenLabs', 'UberDuck']
        new_col_name = 'Fake'
        loader.selectRandomArchitecture(target_col=new_col_name, source_cols=source_architectures)
        
        data_df = loader.generateFinalDataFrame(real_col='Real', fake_cols=fake_cols)
        
        return data_df
    
    #################################################################################
    ################################# Feature Generation ############################
    #################################################################################
    
    def generate_features(self, feature_method='all', openSmile_feature_count=10):

        
        #### titanet features ####
        if feature_method == 'titanet':
            self.feature_store['titanet'] = self._generate_titanet_features()

        #### openSmile features ####
        if feature_method == 'openSmile_binary':
            self.feature_store['openSmile_binary'] = self._generate_openSmile_features(feature_selector_type='random_forest', 
                                                                                       label_type='binary', 
                                                                                       feature_count=openSmile_feature_count)

        if feature_method == 'openSmile_multiclass':
            self.feature_store['openSmile_multiclass'] = self._generate_openSmile_features(feature_selector_type='random_forest', 
                                                                                            label_type='multiclass', 
                                                                                            feature_count=openSmile_feature_count)
        #### cadence features ####ß
        if feature_method == 'cadence':
            self.feature_store['cadence'] = self._generate_cadence_features()

        #### all features ####ß
        if feature_method == 'all':
            self.feature_store['titanet'] = self._generate_titanet_features()
            self.feature_store['openSmile_binary'] = self._generate_openSmile_features(feature_selector_type='random_forest', 
                                                                                       label_type='binary', 
                                                                                       feature_count=openSmile_feature_count)
            self.feature_store['openSmile_multiclass'] = self._generate_openSmile_features(feature_selector_type='random_forest', 
                                                                                            label_type='multiclass', 
                                                                                            feature_count=openSmile_feature_count)
            self.feature_store['cadence'] = self._generate_cadence_features()

    #### private methods for feature generation ####
    
    def _generate_titanet_features(self):

        speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large')
        embedding_manager = AudioEmbeddingsManager(model=speaker_model, data=self.data_df) 

        return embedding_manager.generateFeatureDf()

    def _generate_openSmile_features(self, feature_selector_type, label_type, feature_count):

        smile_manager = SmileFeatureManager(self.data_df)
        
        if label_type == 'binary':
            return smile_manager.generateFeatureDf(feature_selector_type, label_type, feature_count)

        if label_type == 'multiclass':
            return smile_manager.generateFeatureDf(feature_selector_type, label_type, feature_count)

    def _generate_cadence_features(self):
        
        cadence_manager = CadenceModelManager(self.data_df)
        cad_feature_df, cad_feature_cols, scalar = cadence_manager.run_cadence_feature_extraction_pipeline(fill_na=-1) # Add param for load features or not

        return cad_feature_df, cad_feature_cols
    
    #################################################################################
    ################################# Train Predict #################################
    #################################################################################
    
    def train_predict_using_models(self, models = ['logreg', 'random_forest'], run_tags=None, run_name_prefix=None, create_df_artifact=False):

        #run train eval loop
        for model_type in models:
            for label_type in ['label', 'multiclass_label']:
                for feature_method, feature_data in self.feature_store.items():

                    #condition to skip certain feature methods for certain label types
                    if feature_method == 'openSmile_binary' and label_type == 'multiclass_label':
                        break
                    if feature_method == 'openSmile_multiclass' and label_type == 'label':
                        break

                    #generate mlflow run details
                    run_tags, run_name = self._generate_mlflow_run_details(run_tags, run_name_prefix, model_type, label_type, feature_method)

                    #start mlflow run
                    with mlflow.start_run(tags=run_tags, run_name=run_name) as run:

                        #instantiate model and perform trainPredict
                        model = ModelManager(model_name=model_type, 
                                             data=feature_data[0], 
                                             feature_cols=feature_data[1], 
                                             merge_train_dev=True)
                        
                        model.trainPredict(label_col=label_type)

                        #mlflow logging
                        self._log_mlflow_run_details(run, model, create_df_artifact)

                        #end mlflow run
                        mlflow.end_run()

                        print('Finished run: ' + run.info.run_name)
            

    #### private methods for train predict ####
    
    def _generate_mlflow_run_details(self, run_tags, run_name_prefix, model_type, label_type, feature_method):

        #tag details
        _run_tags = {'feature_method': feature_method, 'label_type': label_type, 'selected_architectures':self.fake_cols}
        if run_tags is not None:
            _run_tags.update(run_tags)

        #run name
        if (run_tags is not None) and ('laundered'in run_tags.keys()) and (run_tags['laundered'] == 1):
            _run_name = run_name_prefix + '_' + feature_method + '_' + label_type + '_' + model_type + '_laundered'
        else:
            _run_name = run_name_prefix + '_' + feature_method + '_' + label_type + '_' + model_type

        return _run_tags, _run_name
    
    def _log_mlflow_run_details(self, run, model, create_df_artifact) -> None:

        ##### update tags #####
        mlflow.set_tag('estimator_name', type(model.model).__name__)
        
        ##### 1) mlflow log model with schema i.e. signature #####
        signature = mlflow.models.signature.infer_signature(model.X_train, model.y_train)
        mlflow.sklearn.log_model(model.model, "model_" + run.info.run_name, signature=signature)
        
        ##### 2) mlflow log model params #####
        mlflow.log_params(model.model.get_params())

        ##### 3) mlflow log model artifacts #####  
        ## train_dev_test data
        if create_df_artifact:
            data_path = '/home/ubuntu/mlflow-test/data.csv'
            model.data.to_csv(data_path, index=False)   
            mlflow.log_artifact(data_path)
            os.remove(data_path)

        ##### 4) mlflow log model metrics #####
        #save class accuracies independently
        for key, value in model.class_accuracy.items():
            mlflow.log_metric(key + '_accuracy', value) 

        #save aggregate accuracy
        agg_accuracy = 0
        for key, value in model.class_accuracy.items():
            if key in self.fake_cols:
                agg_accuracy += value
            #compute average accuracy for fake classes  
            agg_accuracy = agg_accuracy / len(self.fake_cols) 
            mlflow.log_metric("fake_accuracy", agg_accuracy)
        
        #save aggregate accuracy 
        mlflow.log_metric("accuracy", model.accuracy)

        #save log loss  
        mlflow.log_metric("log_loss", model.log_loss_value)

        #save eer score 
        if model.eer_score is not None:
            mlflow.log_metric("eer_score", model.eer_score)

        #save eer threshold
        if model.eer_threshold is not None:
            mlflow.log_metric("eer_threshold", model.eer_threshold)             

