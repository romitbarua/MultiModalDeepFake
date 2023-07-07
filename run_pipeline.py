import sys
sys.path.append("/home/ubuntu/MultiModalDeepFake")
import pandas as pd
import mlflow
import packages.experiment_pipeline as ep
import time
import argparse

def main(experiment_name, open_smile_feature_count, create_df_artifact):

    #start timing
    start_time = time.time()

    print("\nRunning pipeline for experiment: \n", experiment_name)
    mlflow.set_experiment(experiment_name)

    print("\nopen_smile_feature_count: \n", open_smile_feature_count)
    print("\ncreate_df_artifact: \n", create_df_artifact)

    #set the models to run
    models = ['logreg', 'random_forest']

    ####################################
    ##### run for unlaundered data #####
    ####################################

    #mlflow tag setting
    run_tags={'laundered':0}
    #metadata path
    metadata_path_unlaundered = '/home/ubuntu/data/wavefake_data/LJ_metadata_16000KHz.csv'

    #pipeline params
    run_params = {}
    run_params['EL'] = ['ElevenLabs']
    run_params['UD'] = ['UberDuck']
    run_params['WF'] = ['RandWaveFake']
    run_params['EL_UD'] = ['ElevenLabs', 'UberDuck']
    run_params['EL_UD_WF'] = ['ElevenLabs', 'UberDuck', 'RandWaveFake']
    run_params['EL_UD_Fake'] = ['EL_UD_Fake']
    run_params['Fake'] = ['Fake']

    #run the pipeline for unlaundered data
    for run_name_prefix, fake_cols in run_params.items():
        print('Running pipeline for unlaundered data with fake_cols: ', fake_cols)

        #create and run pipeline object
        exp = ep.ExperimentPipeline(fake_cols, metadata_path_unlaundered) 
        exp.generate_features(feature_method='all', open_smile_feature_count=open_smile_feature_count)
        if len(fake_cols) == 1:
            exp.train_predict_using_models(run_name_prefix=run_name_prefix, run_tags=run_tags, 
                                            models=models, create_df_artifact=create_df_artifact, label_type='label')
        else:
            exp.train_predict_using_models(run_name_prefix=run_name_prefix, run_tags=run_tags, 
                                            models=models, create_df_artifact=create_df_artifact, label_type='multiclass_label')
        
    ####################################
    ##### run for laundered data #####
    ####################################

    #mlflow tag setting
    run_tags={'laundered':1}
    #metadata path
    metadata_path_laundered = '/home/ubuntu/data/wavefake_data/LJ_metadata_16KHz_Laundered.csv'

    #pipeline params
    run_params = {}
    run_params['EL'] = ['ElevenLabs']
    run_params['UD'] = ['UberDuck']
    run_params['WF'] = ['RandWaveFake']
    run_params['EL_UD'] = ['ElevenLabs', 'UberDuck']
    run_params['EL_UD_WF'] = ['ElevenLabs', 'UberDuck', 'RandWaveFake']
    run_params['EL_UD_Fake'] = ['EL_UD_Fake']
    run_params['Fake'] = ['Fake']

    #run the pipeline for unlaundered data
    for run_name_prefix, fake_cols in run_params.items():
        print('Running pipeline for unlaundered data with fake_cols: ', fake_cols)

        #create and run pipeline object
        exp = ep.ExperimentPipeline(fake_cols, metadata_path_laundered) 
        exp.generate_features(feature_method='all', open_smile_feature_count=open_smile_feature_count)
        if len(fake_cols) == 1:
            exp.train_predict_using_models(run_name_prefix=run_name_prefix, run_tags=run_tags, 
                                            models=models, create_df_artifact=create_df_artifact, label_type='label')
        else:
            exp.train_predict_using_models(run_name_prefix=run_name_prefix, run_tags=run_tags, 
                                            models=models, create_df_artifact=create_df_artifact, label_type='multiclass_label')
        
    print('\nDone running pipeline for all data successfully!\n')


if __name__ == "__main__":

    # Create an argument parser
    parser = argparse.ArgumentParser(description="Run pipeline")

    # Add the command-line arguments
    parser.add_argument("experiment_name", type=str, help="Name of the experiment")
    parser.add_argument("--create_df_artifact", action="store_true", help="Flag to enable creating df artifact")
    parser.add_argument("--open_smile_feature_count", type=int, default=10, help="Value for open smile feature count")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Check if the experiment name is provided
    if not args.experiment_name:
        parser.error("Experiment name is required.")

    # Extract the arguments
    experiment_name = args.experiment_name
    create_df_artifact = args.create_df_artifact
    open_smile_feature_count = args.open_smile_feature_count

    # Call the main function with the arguments
    main(experiment_name, open_smile_feature_count, create_df_artifact)