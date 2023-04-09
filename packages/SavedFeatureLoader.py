import pandas as pd
import numpy as np
import os
import random
from tqdm import tqdm
from pathlib import Path
import json

FEATURE_MAP_PATH = '/home/ubuntu/data/FeatureMap.json'


def loadFeatures(metadata, feature_type, metadata_filepath_col='path', feature_filepath_col='path', feature_map_path=FEATURE_MAP_PATH):
    
    metadata['path_keys'] = metadata['path'].apply(os.path.dirname)
    present_paths = metadata['path_keys'].unique().tolist()
    with open(feature_map_path) as f:
        feature_map = json.load(f)
    
    merged_df = pd.DataFrame()
    for path in tqdm(present_paths):
        feature_df = pd.read_csv(feature_map[path][feature_type])
        filter_metadata = metadata[metadata['path_keys'] == path]
        merged_df = pd.concat([merged_df,pd.merge(filter_metadata, feature_df, how='left', left_on=metadata_filepath_col, right_on=feature_filepath_col)], axis=0).reset_index(drop=True)
    
    merged_df = merged_df.drop(columns=['path_keys']) 
    return merged_df
        
    

    

        
