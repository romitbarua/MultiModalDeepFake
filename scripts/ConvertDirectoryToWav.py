import sys, os
sys.path.insert(0, '.')

from packages.AudioManager import AudioManager

def main():

    dir = '/Users/romitbarua/Documents/Berkeley/Spring 2023/MultiModalDeepFake/data/wavefake_data/ElevenLabsDeepFakes'

    am = AudioManager()
    am.convertAudioDirectory(dir, input_format='.mpeg')

main()