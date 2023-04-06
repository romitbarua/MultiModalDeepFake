from random import random
import pandas as pd
import numpy as np

class LJDataLoader:

    def __init__(self, data_path: str, id_col: str = 'id') -> None:
        
        assert '.csv' in data_path, 'Data Path should be a csv file.'
        self.metadata = pd.read_csv(data_path)
        self._validateData()
        self.id_col = id_col
        
    def _validateData(self):
        # ADD MORE GRANULAR FUNCTIONALITY LATER
        self.metadata = self.metadata.dropna().reset_index()
        
    def sample(self, perc: float = 0.1):
        self.metadata = self.metadata.sample(frac=perc, ignore_index=True)
        

    def splitData(self, train_perc=0.6, dev_perc=0.2, test_perc=0.2, shuffle: bool=True):
        assert train_perc+dev_perc+test_perc == 1, ''
        
        if shuffle:
            self.metadata = self.metadata.sample(frac=1, ignore_index=True)

        self.metadata['type'] = None
        
        train_idx, dev_idx = int(self.metadata.shape[0]*train_perc), int(self.metadata.shape[0]*(train_perc+dev_perc))
        
        print(self.metadata.shape[0])
        print(train_idx, dev_idx)
        
        self.metadata.loc[:train_idx, 'type'] = 'train'
        self.metadata.loc[train_idx:dev_idx, 'type'] = 'dev'
        self.metadata.loc[dev_idx:, 'type'] = 'test'

    def selectRandomArchitecture(self, target_col: str, source_cols: list):

        def randomlySelectCols(rw):
            rand_idx = np.random.randint(0, len(source_cols))
            return rw[source_cols[rand_idx]]

        self.metadata[target_col] = self.metadata.apply(lambda row: randomlySelectCols(row), axis=1)

    def generateFinalDataFrame(self, real_col: str, fake_cols: list, single_id_entry: bool = False):

        agg_cols = [real_col] + fake_cols

        if single_id_entry:
            filter_df = self.metadata[agg_cols].copy()
            multiclass_labels = np.random.randint(0, len(agg_cols), filter_df.shape[0]).reshape(filter_df.shape[0], -1)
            chosen_data = np.take_along_axis(filter_df.to_numpy(), multiclass_labels, axis=1).squeeze()
            multiclass_labels = multiclass_labels.squeeze()
            labels = np.where(multiclass_labels == 0, 0, 1) #in the future, may need to double check that this works for varying column orders
            return pd.DataFrame({'path':chosen_data, 'label':labels, 'multiclass_label':multiclass_labels, 'type':self.metadata['type']})

        filter_df = self.metadata[agg_cols+['type']].copy()
        output = pd.melt(filter_df, id_vars=['type'], value_vars=agg_cols, value_name='path', var_name='architecture')
        output['label'] = np.where(output['architecture']==real_col, 0, 1)
        multiclass_map = {k: v for v, k in enumerate(agg_cols)}
        output['multiclass_label'] = output['architecture'].map(multiclass_map)
        output = output.drop(columns=['architecture'])
        return output
    

