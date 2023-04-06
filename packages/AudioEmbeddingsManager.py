import numpy as np
import pandas as pd

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

    def generateFeatureDf(self, normalize: bool = False):
        
        embeddings_df = pd.DataFrame(self.generateEmbeddings(normalize))
        
        feature_cols = list(embeddings_df.columns)
        feature_df = pd.concat((self.data, embeddings_df), axis=1)
        return feature_df, feature_cols


    def generateEmbeddings(self, normalize):
        
        return generateTitaNetEmbeddings(self.model, self.data['path'], normalize)






    
