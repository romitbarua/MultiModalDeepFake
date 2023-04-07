import pandas as pd
import numpy as np
import os
import random
from tqdm import tqdm
from pathlib import Path
import json
import opensmile

#base_path
base_path = "/home/ubuntu/"

class smileFeatureLoader:
    def __init__(self, df, 
                 file_mappings= '/home/ubuntu/data/OpenSmile/04-2023-OSF-feature-csvs/file_mappings.json') -> None:

        self.df = df
        self.df['path_keys'] = df.path.apply(lambda x: '/'.join(x.split('/')[:-1]))
        
        with open(file_mappings) as f:
            self.file_mappings = json.load(f)
        
        self.merged_df = self._load_smile_features()

    ## TBD - need to handle NaNs if the merge did not find a match for smile features
    ## call openSMILE feature generator to generate features for the missing files OR ignore the files
        
    def _load_smile_features(self):

        merged_df = pd.DataFrame()
        
        #iterate through all the unique path keys in the provided df
        for path_key in tqdm(self.df.path_keys.unique()):
            
            #load the precomputed smile features for files in the path key
            csv_path = self.file_mappings[path_key]
            right_df = pd.read_csv(csv_path)
            right_df = right_df.rename(columns={'file':'path'})
            
            #create the left df by filtering the provided df by the path key
            left_df = self.df[self.df.path_keys == path_key]

            #merge the left and right dfs
            merged_df = pd.concat([merged_df, pd.merge(left_df, right_df, on='path' , how='left')], axis=0).reset_index(drop=True)

        return merged_df
    
    def return_merged_df(self):
        
        return self.merged_df.drop(columns=['path_keys']).dropna()
    

        
