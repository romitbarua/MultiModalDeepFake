import sys, os
sys.path.insert(0, '.')

from packages.AudioManager import AudioManager

def main():
    
    #input_folder = '/home/ubuntu/data/wavefake_data/generated_audio/ljspeech_uberduck/Original'
    #output_folder = '/home/ubuntu/data/wavefake_data/generated_audio/ljspeech_uberduck/16000KHz'
    target_sample_rate = 16000
    
    am = AudioManager()
    
    home_path = '/home/ubuntu/'
    
    folders_to_convert = [('data/TIMIT_and_ElevenLabs/Original','data/TIMIT_and_ElevenLabs/16KHz')]
    
    
    
    for folders in folders_to_convert:
        input_folder = os.path.join(home_path, folders[0])
        output_folder = os.path.join(home_path, folders[1])
        
        am.resampleAudioDirectory(input_directory=input_folder, output_directory=output_folder, target_sample_rate=target_sample_rate, replace_existing=True)
    
main()