import sys, os
sys.path.insert(0, '.')

from packages.AudioManager import AudioManager

def main():

    input_dir = '/home/ubuntu/data/wavefake_data/LJSpeech_1.1/wavs/16000KHz'
    output_dir = '/home/ubuntu/data/wavefake_data/LJSpeech_1.1/wavs/16000KHz_AAC_64K'
    bitrate='64k'

    am = AudioManager()
    am.convertAudioDirectory(input_dir, input_format='.wav', output_dir=output_dir,
                             output_format='.mp4', delete_original=False, bitrate=bitrate,
                             codec='aac'
                            )
    
    #am.convertAudioDirectory(output_dir, input_format='.mp4', 
    #                         output_format='.wav', delete_original=True,
    #                         codec='aac'
    #                        )

main()