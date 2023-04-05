import sys
sys.path.insert(0, '.')

from packages.AudioManager import AudioManager

def main():
    
    input_folder = '/home/ubuntu/data/wavefake_data/generated_audio/ljspeech_uberduck/Original'
    output_folder = '/home/ubuntu/data/wavefake_data/generated_audio/ljspeech_uberduck/16000KHz'
    target_sample_rate = 16000
    
    am = AudioManager()
    am.resampleAudioDirectory(input_directory=input_folder, output_directory=output_folder, target_sample_rate=target_sample_rate, replace_existing=False)
    
main()