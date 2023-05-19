import sys, os
sys.path.insert(0, '.')

import pandas as pd

from packages.AudioManager import AudioManager

def main():
    
    am = AudioManager()
    
    home_path = '/home/ubuntu/'
    
    folders_to_convert = [('data/wavefake_data/generated_audio/ljspeech_elevenlabs/16000KHz', 'data/wavefake_data/generated_audio/ljspeech_elevenlabs/16KHz_Laundered'), 
                          ('data/wavefake_data/generated_audio/ljspeech_full_band_melgan/16000KHz', 'data/wavefake_data/generated_audio/ljspeech_full_band_melgan/16KHz_Laundered'),
                          ('data/wavefake_data/generated_audio/ljspeech_hifiGAN/16000KHz', 'data/wavefake_data/generated_audio/ljspeech_hifiGAN/16KHz_Laundered'),
                          ('data/wavefake_data/generated_audio/ljspeech_melgan/16000KHz', 'data/wavefake_data/generated_audio/ljspeech_melgan/16KHz_Laundered'),
                          ('data/wavefake_data/generated_audio/ljspeech_melgan_large/16000KHz', 'data/wavefake_data/generated_audio/ljspeech_melgan_large/16KHz_Laundered'),
                          ('data/wavefake_data/generated_audio/ljspeech_multi_band_melgan/16000KHz','data/wavefake_data/generated_audio/ljspeech_multi_band_melgan/16KHz_Laundered'),
                          ('data/wavefake_data/generated_audio/ljspeech_parallel_wavegan/16000KHz','data/wavefake_data/generated_audio/ljspeech_parallel_wavegan/16KHz_Laundered'),
                          ('data/wavefake_data/generated_audio/ljspeech_uberduck/16000KHz','data/wavefake_data/generated_audio/ljspeech_uberduck/16KHz_Laundered'),
                          ('data/wavefake_data/generated_audio/ljspeech_waveglow/16000KHz','data/wavefake_data/generated_audio/ljspeech_waveglow/16KHz_Laundered'),
                          ('data/wavefake_data/LJSpeech_1.1/wavs/16000KHz','data/wavefake_data/LJSpeech_1.1/wavs/16KHz_Laundered')]
    
    
    results = []
    
    for folders in folders_to_convert:
        input_dir = os.path.join(home_path, folders[0])
        output_dir = os.path.join(home_path, folders[1])
        
        result = am.launderAudioDirectory(input_dir, output_dir)
        results.extend(result)
        
    output = pd.DataFrame(results, columns=['Path', 'isTranscode', 'BitRate', 'isNoise', 'SNR'])
    output.to_csv('/home/ubuntu/data.csv')
    
    
    
    
main()