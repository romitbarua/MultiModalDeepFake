import numpy as np
import pandas as pd
from packages.SavedFeatureLoader import loadFeatures

SAVED_EMBEDDINGS_DIR = '/home/ubuntu/data/wavefake_data/Embeddings/16000KHz'

def generateTitaNetEmbeddings(model, paths, normalize):
    
        embeddings = np.array([model.get_embedding(file_path).cpu().detach().numpy()[0] for file_path in paths])
        
        if normalize:
            raise NotImplementedError
            #embeddings = normalize(embeddings)
        
        return embeddings

class AudioEmbeddingsManager:

    def __init__(self, model, data) -> None:
        self.model = model
        self.data = data

    def generateFeatureDf(self, normalize: bool = False, regenerate_embeddings: bool = False):
        
        if regenerate_embeddings:
            embeddings_df = pd.DataFrame(self.generateEmbeddings(normalize))

            feature_cols = list(embeddings_df.columns)
            feature_df = pd.concat((self.data, embeddings_df), axis=1)
            
        else:
            feature_df = loadFeatures(self.data.copy(), 'titanet')
            feature_cols = set(feature_df.columns) ^ set(self.data.columns)
            
        return feature_df, feature_cols


    def generateEmbeddings(self, normalize):
        
        return generateTitaNetEmbeddings(self.model, self.data['path'], normalize)






    
