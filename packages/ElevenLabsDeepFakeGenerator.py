import yaml
import requests
import os

from packages.BaseDeepFakeGenerator import BaseDeepFakeGenerator

class ElevenLabsDeepFakeGenerator(BaseDeepFakeGenerator):
    
    def __init__(self):

        super().__init__()
        self.api_key = self._load_API_key()
        
    
    def _load_API_key(self, config_path='./Configs/secret/config.yaml'):
        with open(config_path, 'r') as file:
            inputs = yaml.safe_load(file)
        xi_api_key = inputs['eleven_labs_api_key']
        return xi_api_key

    def generateDeepFakeFromDataFrame(self, dataframe_path: str, output_dir: str,
                                        source_col: str, transcript_col: str, 
                                        voice_id: str):
        
        file_names, transcripts = self.loadTextFromDataFrame(dataframe_path=dataframe_path,
                                                            source_col=source_col,
                                                            transcript_col=transcript_col)

        for idx, transcript in enumerate(transcripts):
            audio_clip = self.generateDeepfake(voice_id=voice_id,
                                                text=transcript)

            file_name = file_names[idx].replace(os.path.splitext(file_names[idx])[1], '.mpeg')
            with open(os.path.join(output_dir, file_name), 'wb') as f:
                f.write(audio_clip.content)
                f.close()
        
    def generateDeepfake(self, voice_id, text):
        headers = {
            'accept': 'audio/mpeg',
            'xi-api-key': self.api_key,
            'Content-Type': 'application/json'
        }

        data = f'{{"text": "{text}"}}'

        return requests.post(f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}', headers=headers, data=data)

        