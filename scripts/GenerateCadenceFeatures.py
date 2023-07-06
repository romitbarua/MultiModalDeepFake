import sys, os
sys.path.insert(0, '.')
import pandas as pd
import json
from packages.CadenceModelManager import CadenceModelManager


params_path = '/home/ubuntu/data/wavefake_data/Cadence_features/16khz_Laundered/params.json'
metadata_path = '/home/ubuntu/data/wavefake_data/LJ_metadata_16KHz_Laundered.csv'
output_dir = '/home/ubuntu/data/TIMIT_and_ElevenLabs/cadence'

with open(params_path) as file:
    params = json.load(file)
metadata = pd.read_csv(metadata_path)

archs = ['Full_Band_MelGan', 'HifiGan', 'MelGan', 'MelGanLarge', 'Multi_Band_MelGan', 'Parallel_WaveGan', 'Waveglow', 'ElevenLabs', 'UberDuck', 'Real']

arch = 'Real'
file_paths = metadata[[arch]].dropna()
file_paths.columns = ['path']

output = None

path_dir = '/home/ubuntu/data/TIMIT_and_ElevenLabs/16KHz'
paths = os.listdir(path_dir)
paths = [os.path.join(path_dir, path) for path in paths]
file_paths = pd.DataFrame({'path':paths})


model_manager = CadenceModelManager(file_paths)

print(f'Running for {arch}')

'''
for key in params.keys():
    window_size = params[key]['window_size']
    silence_threshold = params[key]['silence_threshold']
    
    key_output = model_manager.generate_features(window_size, silence_threshold, list(file_paths['path']))
    key_output.columns  = [f'{key}_{col}' for col in key_output.columns]
    key_output['path'] = list(file_paths['path'])
    
    if output is None:
        output = key_output
    else:
        output = output.merge(key_output, on='path')
'''

for key in ['Mix']:
    window_size = params[key]['window_size']
    silence_threshold = params[key]['silence_threshold']
    
    key_output = model_manager.generate_features(window_size, silence_threshold, list(file_paths['path']))
    key_output.columns  = [f'{key}_{col}' for col in key_output.columns]
    key_output['path'] = list(file_paths['path'])
    
    if output is None:
        output = key_output
    else:
        output = output.merge(key_output, on='path')
        
output.to_csv(f'{output_dir}/{arch}.csv', index=False)
    




