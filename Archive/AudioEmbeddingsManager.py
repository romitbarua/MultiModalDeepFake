import numpy as np

class AudioEmbeddingsManager:

    def __init__(self, model, data) -> None:
        self.model = model
        self.data = data

    def generateEmbeddings(self, normalize: bool = False):

        embeddings = np.array([self.model.get_embedding(file_path).cpu().detach().numpy()[0] for file_path in self.data['path']])
        
        if normalize:
            raise NotImplementedError
            #embeddings = normalize(embeddings)
        
        np.random.shuffle(embeddings)
        return embeddings



    
