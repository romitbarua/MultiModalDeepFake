import sys
sys.path.append("/home/ubuntu/MultiModalDeepFake")
import pandas as pd
import mlflow
import packages.experiment_pipeline as ep
import time

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
    metadata_path = '/home/ubuntu/data/wavefake_data/LJ_metadata_16000KHz.csv'

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
        exp = ep.ExperimentPipeline(fake_cols, metadata_path) 
        exp.generate_features(feature_method='all', openSmile_feature_count=openSmile_feature_count)
        exp.train_predict_using_models(run_name_prefix=run_name_prefix, run_tags=run_tags, 
                                        models=models, create_df_artifact=create_df_artifact)
        
    ####################################
    ##### run for laundered data #####
    ####################################

    #mlflow tag setting
    run_tags={'laundered':1}
    #metadata path
    metadata_path = '/home/ubuntu/data/wavefake_data/LJ_metadata_16KHz_Laundered.csv'

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
        exp = ep.ExperimentPipeline(fake_cols, metadata_path) 
        exp.generate_features(feature_method='all', openSmile_feature_count=openSmile_feature_count)
        exp.train_predict_using_models(run_name_prefix=run_name_prefix, run_tags=run_tags, 
                                        models=models, create_df_artifact=create_df_artifact)
        
    print('\nDone running pipeline for all data successfully!\n')


if __name__ == "__main__":

    # Check if the correct number of arguments are provided
    if len(sys.argv) < 2 or len(sys.argv) > 5:
        print("Usage: python3 run_pipeline.py <experiment_name> [create_df_artifact] [open_smile_feature_count=<value>]")
        sys.exit(1)

    # Retrieve the command-line arguments
    experiment_name = sys.argv[1]
    arg2 = sys.argv[2] if len(sys.argv) >= 3 else None

    #boolean for logging df artifact
    create_df_artifact = True if arg2 == "create_df_artifact" else False
    
    #set default value for open_smile_feature_count
    open_smile_feature_count = 10
    # Extract open_smile_feature_count if provided
    for arg in sys.argv[3:]:
        if arg.startswith("open_smile_feature_count="):
            open_smile_feature_count = int(arg.split("=")[1])

    # Call the main function with the arguments
    main(experiment_name, open_smile_feature_count, create_df_artifact)