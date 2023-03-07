
from packages.BaseDeepFakeGenerator import BaseDeepFakeGenerator

class ElevenLabsDeepFakeGenerator(BaseDeepFakeGenerator):
    
    def __init__(self):
        
        self.api_key = self._load_API_key()
        
    
    def _load_API_key(self, config_path='../config/config.yaml'):
        with open('../config/config.yaml', 'r') as file:
            inputs = yaml.safe_load(file)
        xi_api_key = inputs['eleven_labs_api_key']
        
    def genderate_deepfake(self, voice_id, text, full_output_path):
        headers = {
            'accept': 'audio/mpeg',
            'xi-api-key': api_key,
            'Content-Type': 'application/json'
        }

        #data = '{"text": "Eleven Labs has a made a voice cloning AI that will do a good job at modeling Biden"}'
        data = f'{{"text": "{text}"}}'

        r = requests.post(f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}', headers=headers, data=data)

        with open(f'/Users/romitbarua/Documents/Berkeley/Spring 2023/world_leaders/ElevenLabDeepFakes/{output_file}.mpeg', 'wb') as f:
            f.write(r.content)
            f.close()
        
        