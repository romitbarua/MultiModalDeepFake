import sys
sys.path.insert(0, '.')

from packages.ElevenLabsDeepFakeGenerator import ElevenLabsDeepFakeGenerator

def main():

    file_path = '/home/ubuntu/sample.csv'
    output_dir = '/home/ubuntu/data/wavefake_data/generated_audio/11LabsDeepFakes/'
    source_col = 'Real'
    transcript_col = 'transcript_1'
    deepfake_machine = ElevenLabsDeepFakeGenerator()

    deepfake_machine.generateDeepFakeFromDataFrame(dataframe_path=file_path, output_dir=output_dir, source_col=source_col, transcript_col=transcript_col, voice_id='qeVSLLD6XfJjhEBj9qYp', convert_audio_to_format='.wav', punc_to_remove=['"', '£'])


main()
