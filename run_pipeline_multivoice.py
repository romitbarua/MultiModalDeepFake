import sys
sys.path.append("/home/ubuntu/MultiModalDeepFake")
import pandas as pd
import mlflow
import packages.experiment_pipeline as ep
import time
import argparse

"""
#seeding the random number generator
#GK Comment: This code is not needed since we are now sorting the list
import random
random_seed = 10
random_generator = random.Random(random_seed)
"""

from packages.TIMITDataLoader import TIMITDataLoader
from packages.LJDataLoader import LJDataLoader
from packages.AudioEmbeddingsManager import AudioEmbeddingsManager
from packages.ModelManager import ModelManager
from packages.CadenceModelManager import CadenceModelManager
import packages.AnalysisManager as am
from packages.SmileFeatureManager import SmileFeatureManager


#fixed values
timit_data_path = '/home/ubuntu/data/TIMIT_and_ElevenLabs/TIMIT and ElevenLabs'
fake_voices = ['Adam', 'Antoni', 'Arnold', 'Bella', 'Biden', 'Domi', 'Elli', 'Josh', 'Obama', 'Rachel', 'Sam']
#set the models to run
models = ['logreg', 'random_forest']

#helper functions
def chunks(lst, n):
    #sort the list
    lst.sort()
    #random_generator.shuffle(lst)
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main(experiment_name, open_smile_feature_count, create_df_artifact):

    #load the timit data
    timit_data_loader = TIMITDataLoader(timit_data_path)
    #generate the split
    df = timit_data_loader.generate_split()
    #get speakers
    df['speaker'] = [item.split('/')[-1].split('_')[0] for item in df['path']]

    #create partitions
    real_speakers = list(set([item for item in df['speaker'] if not item.startswith(tuple(fake_voices))]))
    fake_speakers = list(set([item for item in df['speaker'] if item.startswith(tuple(fake_voices))]))

    real_speaker_partitions = list(chunks(real_speakers, 20))
    fake_speaker_partitions = list(chunks(fake_speakers, 2))

    #debugging
    #print("Length of real + fake speakers {}\n".format(len(real_speakers + fake_speakers)))
    #print(df[(df['type']=='train') | (df['type']=='dev')].value_counts('multiclass_label'))
    #print(real_speaker_partitions)
    #print(fake_speaker_partitions)

    #set the mlflow experiment
    mlflow.set_experiment(experiment_name)

    # run the experiment
    counter = 1 
    for fake_speaker_chunk in fake_speaker_partitions:
        for real_speaker_chunk in real_speaker_partitions:
            
            #debugging code
            #print(fake_speaker_chunk)
            #print(real_speaker_chunk)
            #progress bar
            print(f'\nprogress: {counter}/{len(fake_speaker_partitions)*len(real_speaker_partitions)}\n')

            #voices to remove
            voices_to_remove = fake_speaker_chunk+real_speaker_chunk

            #generating split speaker test from the 
            data_df = timit_data_loader.generate_split_speaker_test(voices_to_remove)

            #instantiate the pipeline class by providing the df
            exp = ep.ExperimentPipeline(fake_cols=['ElevenLabs'], metadata_path=None, data_df=data_df)

            #generate features
            exp.generate_features(feature_method='all', open_smile_feature_count=open_smile_feature_count)

            run_name_prefix = f'multivoice_run_{counter}'
            run_tags = {'voices_to_remove': voices_to_remove}

            exp.train_predict_using_models(run_name_prefix=run_name_prefix, run_tags=run_tags, 
                                           models=models, create_df_artifact=create_df_artifact, label_type='label')


            counter+=1

            

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
