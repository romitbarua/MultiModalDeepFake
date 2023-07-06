import sys
sys.path.append("/home/ubuntu/MultiModalDeepFake")
import pandas as pd
import mlflow
import experiment_pipeline as ep

def main(arg1, arg2):

    #set the experiment name from arg1
    mlflow.set_experiment(arg1)
    openSmile_feature_count=20
    ##### set the parameters for running all models #####
    #these could be an optional arguments from command line
    models = ['logreg', 'random_forest']

    #boolean for logging df artifact
    create_df_artifact = True if arg2 == "create_df_artifact" else False

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
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python3 run_pipeline.py <experiment_name> [create_df_artifact]")
        sys.exit(1)

    # Retrieve the command-line arguments
    arg1 = sys.argv[1]
    arg2 = sys.argv[2] if len(sys.argv) == 3 else None

    # Call the main function with the arguments
    main(arg1, arg2)