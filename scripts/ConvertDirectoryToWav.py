import sys, os
sys.path.insert(0, '.')

from packages.AudioManager import AudioManager

def main():

    dir = '/home/ubuntu/data/wavefake_data/generated_audio/ljspeech_elevenlabs'

    am = AudioManager()
    am.convertAudioDirectory(dir, input_format='.mpeg')

main()