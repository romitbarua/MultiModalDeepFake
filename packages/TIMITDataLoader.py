import os
import glob
from random import random, sample, seed, shuffle
import pandas as pd
import numpy as np

class TIMITDataLoader:

    def __init__(self, data_path: str, id_col: str = 'id') -> None:
        self.file_path = data_path
        
    def flatten(self, l):
        return [item for sublist in l for item in sublist]
    
    def get_all_files(self):
        files = []
        for r, d, f in os.walk(self.file_path):
            for file in f:
                if '.wav' in file.lower():
                    files.append(os.path.join(r, file))
        
        cleaned_files = [item for item in files if not '_processed' in item]
        
        return(cleaned_files)
        
    def generate_split(self):

        data_df = self.generateFinalDataFrame()
        
        indices = list(data_df.index)
        
        shuffle(indices)
        total_samples = len(indices)
        train_size = int(0.6 * total_samples)
        dev_size = int(0.2 * total_samples)

        train_indices = indices[:train_size]
        dev_indices = indices[train_size:train_size + dev_size]
        test_indices = indices[train_size + dev_size:]
        
        data_df.loc[train_indices, 'type']= 'train'
        data_df.loc[dev_indices, 'type'] = 'dev'
        data_df.loc[test_indices, 'type'] = 'test'

        train_count = data_df[data_df['type'] == 'train'].shape[0]
        dev_count = data_df[data_df['type'] == 'dev'].shape[0]
        test_count = data_df[data_df['type'] == 'test'].shape[0]

        print(f'# of Train instances: {train_count}')
        print(f'# of Dev instances: {dev_count}')
        print(f'# of Test instances: {test_count}')

        return data_df
    
    def generate_split_speaker_test(self, speakers_to_remove=['Biden', 'MJAE0']):

        data_df = self.generateFinalDataFrame()
        
        data_df['speaker'] = [item.split('/')[-1].split('_')[0] for item in data_df['path']]
        data_df['remove'] = [1 if item in speakers_to_remove else 0 for item in data_df['speaker']]
        
        data_df_without_test_speakers = data_df[data_df['remove'] == 0]
        data_df_with_test_speakers = data_df[data_df['remove'] == 1]

        cleaned_indices = list(data_df_without_test_speakers.index)
        removed_indices = list(data_df_with_test_speakers.index)
        
        shuffle(cleaned_indices)
        total_samples = len(cleaned_indices)
        train_size = int(0.6 * len(cleaned_indices))
        dev_size = int(0.2 * len(cleaned_indices))

        train_indices = cleaned_indices[:train_size]
        dev_indices = cleaned_indices[train_size:train_size + dev_size]
        test_indices = cleaned_indices[train_size + dev_size:]
        
        data_df.loc[train_indices, 'type']= 'train'
        data_df.loc[dev_indices, 'type'] = 'dev'
        data_df.loc[test_indices, 'type'] = 'test'
        
        # Drop the original 'test' indices
        data_df = data_df[data_df.type != 'test']
        
        # Set the left out speakers to be the only 'test' indices
        data_df.loc[removed_indices, 'type'] = 'test'
        
        # Clean up dataframe
        data_df.drop(['remove'], axis=1, inplace=True)

        train_count = data_df[data_df['type'] == 'train'].shape[0]
        dev_count = data_df[data_df['type'] == 'dev'].shape[0]
        test_count = data_df[data_df['type'] == 'test'].shape[0]

        print(f'# of Train instances: {train_count}')
        print(f'# of Dev instances: {dev_count}')
        print(f'# of Test instances: {test_count}')

        return data_df.reset_index(drop=True)
        
    
    def generateFinalDataFrame(self, balanced: bool = True):
        
        phrases = [item for item in os.listdir(self.file_path) if not item.startswith('.')]
        
        # Split into real and fake files with lots of checks 
        fake_voices = ['Adam', 'Antoni', 'Arnold', 'Bella', 'Biden', 'Domi', 'Elli', 'Josh', 'Obama', 'Rachel', 'Sam']
        all_files = [os.listdir(self.file_path+f'/{phrase}/real') for phrase in phrases] + [os.listdir(self.file_path+f'/{phrase}/fake/') for phrase in phrases]
        all_files = self.flatten(all_files)
        
        real_timit_files = [item for item in all_files if not item.startswith(tuple(fake_voices))]
        fake_timit_files = [item for item in all_files if item.startswith(tuple(fake_voices))]
        
        all_paths = self.get_all_files()
        real_timit_paths = [item for item in all_paths if not any(filter_item in item for filter_item in fake_voices)]
        fake_timit_paths = [item for item in all_paths if any(filter_item in item for filter_item in fake_voices)]
            
        # Remove any phrases that aren't present in both real and fake files 
        real_unique_phrases = list(set([item.split('_')[1].split('.')[0] for item in real_timit_files]))
        fake_unique_phrases = list(set([item.split('_')[1].split('.')[0] for item in fake_timit_files]))
        
        non_overlapping_phrases = set(real_unique_phrases) ^ set(fake_unique_phrases)
        overlapping_phrases = set.intersection(set(real_unique_phrases), set(fake_unique_phrases))
        
        for item in non_overlapping_phrases:
            if item not in real_unique_phrases:
                fake_unique_phrases.remove(item)
            if item not in fake_unique_phrases:
                real_unique_phrases.remove(item)
            
        print(f'N real and fake phrases: {len(real_unique_phrases)}, {len(fake_unique_phrases)}')
        if len(real_unique_phrases) != len(fake_unique_phrases):
            print('ERROR: unequal lengths of real and fake phrases')
            return None
        
        if balanced:
            seed(12)
            
            # Real = 0
            balanced_real_paths = []
            balanced_fake_paths = []
            
            real_paths_flattened = []
            fake_paths_flattened = []
            
            
            for phrase in overlapping_phrases: 
                real_files = [item for item in real_timit_files if f'_{phrase}.' in item]
                fake_files = [item for item in fake_timit_files if f'_{phrase}.' in item]
                
                real_paths = [item for item in real_timit_paths if phrase in item] # Needs to be if real_file[i] in full paths 
                fake_paths = [item for item in fake_timit_paths if phrase in item]
    
            
                if isinstance(real_paths, list):
                    real_paths_flattened.append(real_paths[0])
                else:
                    real_paths_flattened.append(real_paths)
                    
                if isinstance(fake_paths, list):
                    fake_paths_flattened.append(fake_paths[0])
                else:
                    fake_paths_flattened.append(fake_paths)
                
                real_len = len(set(real_paths_flattened))
                fake_len = len(set(fake_paths_flattened))
                
                if real_len > fake_len:
                    balanced_fake_paths.append(list(set(fake_paths_flattened)))
                    balanced_real_paths.append(sample(set(real_paths_flattened), fake_len))
                    
                if fake_len > real_len: 
                    balanced_real_paths.append(list(set(real_paths_flattened)))
                    balanced_fake_paths.append(sample(set(fake_paths_flattened), real_len))
        
        balanced_real_paths = list(set(self.flatten(balanced_real_paths)))
        balanced_fake_paths = list(set(self.flatten(balanced_fake_paths)))
        
        print(len(balanced_fake_paths), len(balanced_real_paths))
        
        # Debug
        print(balanced_real_paths[0])
        
        df = pd.DataFrame({'type':['tbc' for i in range(len(balanced_real_paths)+len(balanced_fake_paths))],
                          'id':[i for i in range(len(balanced_real_paths)+len(balanced_fake_paths))], 
                          'architecture':[0 for item in balanced_real_paths] + [1 for item in balanced_fake_paths],
                           'orig_path': balanced_real_paths + balanced_fake_paths,
                           'label':[0 for item in balanced_real_paths] + [1 for item in balanced_fake_paths],
                           'multiclass_label':[0 for item in balanced_real_paths] + [1 for item in balanced_fake_paths]
                          })
        
    
        downsampled_src = '/home/ubuntu/data/TIMIT_and_ElevenLabs/16KHz'
        orig_paths = df['orig_path'].tolist()
        downsampled_paths = [os.path.join(downsampled_src, os.path.basename(path)) for path in orig_paths]
        
        df['path'] = downsampled_paths
        
        return df
    
'''
import os
import glob
from random import random, sample, seed, shuffle
import pandas as pd
import numpy as np

class TIMITDataLoader:

    def __init__(self, data_path: str, id_col: str = 'id') -> None:
        self.file_path = data_path
        
    def flatten(self, l):
        return [item for sublist in l for item in sublist]
    
    def get_all_files(self):
        files = []
        for r, d, f in os.walk(self.file_path):
            for file in f:
                if '.wav' in file.lower():
                    files.append(os.path.join(r, file))
        
        cleaned_files = [item for item in files if not '_processed' in item]
        
        return(cleaned_files)
        
    def generate_split(self):

        data_df = self.generateFinalDataFrame()
        
        indices = list(data_df.index)
        
        shuffle(indices)
        total_samples = len(indices)
        train_size = int(0.6 * total_samples)
        dev_size = int(0.2 * total_samples)

        train_indices = indices[:train_size]
        dev_indices = indices[train_size:train_size + dev_size]
        test_indices = indices[train_size + dev_size:]
        
        data_df.loc[train_indices, 'type']= 'train'
        data_df.loc[dev_indices, 'type'] = 'dev'
        data_df.loc[test_indices, 'type'] = 'test'

        train_count = data_df[data_df['type'] == 'train'].shape[0]
        dev_count = data_df[data_df['type'] == 'dev'].shape[0]
        test_count = data_df[data_df['type'] == 'test'].shape[0]

        print(f'# of Train instances: {train_count}')
        print(f'# of Dev instances: {dev_count}')
        print(f'# of Test instances: {test_count}')

        return data_df
    
    def generate_split_speaker_test(self, speakers_to_remove=['Biden', 'MJAE0']):

        data_df = self.generateFinalDataFrame()
        
        data_df['speaker'] = [item.split('/')[-1].split('_')[0] for item in data_df['path']]
        data_df['remove'] = [1 if item in speakers_to_remove else 0 for item in data_df['speaker']]
        
        data_df_without_test_speakers = data_df[data_df['remove'] == 0]
        data_df_with_test_speakers = data_df[data_df['remove'] == 1]

        cleaned_indices = list(data_df_without_test_speakers.index)
        removed_indices = list(data_df_with_test_speakers.index)
        
        shuffle(cleaned_indices)
        total_samples = len(cleaned_indices)
        train_size = int(0.6 * len(cleaned_indices))
        dev_size = int(0.2 * len(cleaned_indices))

        train_indices = cleaned_indices[:train_size]
        dev_indices = cleaned_indices[train_size:train_size + dev_size]
        test_indices = cleaned_indices[train_size + dev_size:]
        
        data_df.loc[train_indices, 'type']= 'train'
        data_df.loc[dev_indices, 'type'] = 'dev'
        data_df.loc[test_indices, 'type'] = 'test'
        
        # Drop the original 'test' indices
        data_df = data_df[data_df.type != 'test']
        
        # Set the left out speakers to be the only 'test' indices
        data_df.loc[removed_indices, 'type'] = 'test'
        
        # Clean up dataframe
        data_df.drop(['remove'], axis=1, inplace=True)

        train_count = data_df[data_df['type'] == 'train'].shape[0]
        dev_count = data_df[data_df['type'] == 'dev'].shape[0]
        test_count = data_df[data_df['type'] == 'test'].shape[0]

        print(f'# of Train instances: {train_count}')
        print(f'# of Dev instances: {dev_count}')
        print(f'# of Test instances: {test_count}')

        return data_df.reset_index(drop=True)
        
    
    def generateFinalDataFrame(self, balanced: bool = True):
        
        phrases = [item for item in os.listdir(self.file_path) if not item.startswith('.')]
        
        # Split into real and fake files with lots of checks 
        fake_voices = ['Adam', 'Antoni', 'Arnold', 'Bella', 'Biden', 'Domi', 'Elli', 'Josh', 'Obama', 'Rachel', 'Sam']
        all_files = [os.listdir(self.file_path+f'/{phrase}/real') for phrase in phrases] + [os.listdir(self.file_path+f'/{phrase}/fake/') for phrase in phrases]
        all_files = self.flatten(all_files)
        
        real_timit_files = [item for item in all_files if not item.startswith(tuple(fake_voices))]
        fake_timit_files = [item for item in all_files if item.startswith(tuple(fake_voices))]
        
        all_paths = self.get_all_files()
        real_timit_paths = [item for item in all_paths if not any(filter_item in item for filter_item in fake_voices)]
        fake_timit_paths = [item for item in all_paths if any(filter_item in item for filter_item in fake_voices)]
            
        # Remove any phrases that aren't present in both real and fake files 
        real_unique_phrases = list(set([item.split('_')[1].split('.')[0] for item in real_timit_files]))
        fake_unique_phrases = list(set([item.split('_')[1].split('.')[0] for item in fake_timit_files]))
        
        non_overlapping_phrases = set(real_unique_phrases) ^ set(fake_unique_phrases)
        overlapping_phrases = set.intersection(set(real_unique_phrases), set(fake_unique_phrases))
        
        for item in non_overlapping_phrases:
            if item not in real_unique_phrases:
                fake_unique_phrases.remove(item)
            if item not in fake_unique_phrases:
                real_unique_phrases.remove(item)
            
        print(f'N real and fake phrases: {len(real_unique_phrases)}, {len(fake_unique_phrases)}')
        if len(real_unique_phrases) != len(fake_unique_phrases):
            print('ERROR: unequal lengths of real and fake phrases')
            return None
        
        if balanced:
            seed(4)
            
            # Real = 0
            balanced_real_paths = []
            balanced_fake_paths = []
            
            real_paths_flattened = []
            fake_paths_flattened = []
            
            
            for phrase in overlapping_phrases: 
                real_files = [item for item in real_timit_files if f'_{phrase}.' in item]
                fake_files = [item for item in fake_timit_files if f'_{phrase}.' in item]
                
                real_paths = [item for item in real_timit_paths if phrase in item] # Needs to be if real_file[i] in full paths 
                fake_paths = [item for item in fake_timit_paths if phrase in item]
    
            
                if isinstance(real_paths, list):
                    real_paths_flattened.append(real_paths[0])
                else:
                    real_paths_flattened.append(real_paths)
                    
                if isinstance(fake_paths, list):
                    fake_paths_flattened.append(fake_paths[0])
                else:
                    fake_paths_flattened.append(fake_paths)
                
                real_len = len(set(real_paths_flattened))
                fake_len = len(set(fake_paths_flattened))
                
                if real_len > fake_len:
                    balanced_fake_paths.append(list(set(fake_paths_flattened)))
                    balanced_real_paths.append(sample(set(real_paths_flattened), fake_len))
                    
                if fake_len > real_len: 
                    balanced_real_paths.append(list(set(real_paths_flattened)))
                    balanced_fake_paths.append(sample(set(fake_paths_flattened), real_len))
        
        balanced_real_paths = list(set(self.flatten(balanced_real_paths)))
        balanced_fake_paths = list(set(self.flatten(balanced_fake_paths)))
        
        print(len(balanced_fake_paths), len(balanced_real_paths))
        
        # Debug
        print(balanced_real_paths[0])
        
        df = pd.DataFrame({'type':['tbc' for i in range(len(balanced_real_paths)+len(balanced_fake_paths))],
                          'id':[i for i in range(len(balanced_real_paths)+len(balanced_fake_paths))], 
                          'architecture':[0 for item in balanced_real_paths] + [1 for item in balanced_fake_paths],
                           'orig_path': balanced_real_paths + balanced_fake_paths,
                           'label':[0 for item in balanced_real_paths] + [1 for item in balanced_fake_paths],
                           'multiclass_label':[0 for item in balanced_real_paths] + [1 for item in balanced_fake_paths]
                          })
        
    
        downsampled_src = '/home/ubuntu/data/TIMIT_and_ElevenLabs/16KHz'
        orig_paths = df['orig_path'].tolist()
        downsampled_paths = [os.path.join(downsampled_src, os.path.basename(path)) for path in orig_paths]
        
        df['path'] = downsampled_paths
        
        return df
    '''