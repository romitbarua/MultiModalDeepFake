import numpy as np
import pandas as pd
from packages.SavedFeatureLoader import loadFeatures

SAVED_EMBEDDINGS_DIR = '/home/ubuntu/data/wavefake_data/Embeddings/16000KHz'

def generateTitaNetEmbeddings(model, paths, normalize):
    
        embeddings = np.array([model.get_embedding(file_path).cpu().detach().numpy()[0] for file_path in paths])
        
        if normalize: ## SB_COMMENT: Why do we no longer need to normalize ?
            raise NotImplementedError
            #embeddings = normalize(embeddings)
        
        return embeddings

class AudioEmbeddingsManager:

    def __init__(self, model, data) -> None:
        self.model = model
        self.data = data

    def generateFeatureDf(self, normalize: bool = False, regenerate_embeddings: bool = False):
        
        if regenerate_embeddings: ## SB_COMMENT: why would we ever need to regenerate? Is there an overwrite later?
            embeddings_df = pd.DataFrame(self.generateEmbeddings(normalize)) ## SB_COMMENT: can probably get rid of the normalize arg now

            feature_cols = list(embeddings_df.columns) ## SB_COMMENT: would there be any benefit in trying less than 192 embeddings?
            feature_df = pd.concat((self.data, embeddings_df), axis=1)
            
        else:
            feature_df = loadFeatures(self.data.copy(), 'titanet')
            feature_cols = list(set(feature_df.columns) ^ set(self.data.columns))
            
        return feature_df, feature_cols


    def generateEmbeddings(self, normalize):
        
        return generateTitaNetEmbeddings(self.model, self.data['path'], normalize)






    
