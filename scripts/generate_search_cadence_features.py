import sys
sys.path.append('/home/ubuntu/MultiModalDeepFake/packages')
from CadenceModelManager import CadenceModelManager
import pandas as pd
import os
import numpy as np

input_dirs = []
output_dir = '/home/ubuntu/data/wavefake_data/cadence_search_features'

metadata_path = '/home/ubuntu/data/wavefake_data/LJ_metadata_16000KHz.csv'
metadata = pd.read_csv(metadata_path)
#metadata = metadata.sample(frac=0.005)

metadata_name = os.path.splitext(os.path.basename(metadata_path))[0]

#architectures = ['Real', 'Full_Band_MelGan', 'HifiGan', 'MelGan', 'MelGanLarge', 'Mutli_Band_MelGan', 'Parallel_WaveGan', 'Waveglow', 'ElevenLabs', 'UberDuck']
#architectures = ['Real', 'Full_Band_MelGan']
#architectures = ['HifiGan', 'MelGan'] 
#architectures = ['MelGanLarge', 'Mutli_Band_MelGan']
#architectures = ['Parallel_WaveGan', 'Waveglow'] 
architectures = ['ElevenLabs', 'UberDuck']

excluded_window_size = []
excluded_silence_threshold = []

for architecture in architectures:
    
    print()
    print(f'Generating Features for {architecture}')
    
    arch_df = pd.DataFrame({'path':metadata[architecture]}).reset_index(drop=True).dropna()
    window_size_options = [50, 100, 150, 250, 500, 750, 1000, 2000]
    silence_threshold_options = np.linspace(0.005, 0.1, 20)
    
    for window_size in window_size_options:
        
        if window_size in excluded_window_size:
            continue
        
        for silence_threshold in silence_threshold_options:
            
            if silence_threshold in excluded_silence_threshold:
                continue
            
            
            cad_model = CadenceModelManager(arch_df)
            features = cad_model.generate_features(window_size, silence_threshold)
            print(arch_df.shape, features.shape)
            df = pd.concat((arch_df, features), axis=1)
            print(df.isna().sum())
            df.to_csv(os.path.join(output_dir, f'{metadata_name}_{architecture}_ws{str(window_size)}_st{str(int(silence_threshold*1000))}.csv'), index=False)
            
            
            
    
    
    


