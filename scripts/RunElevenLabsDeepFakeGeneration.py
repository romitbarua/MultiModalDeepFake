import sys
sys.path.insert(0, '.')

from packages.ElevenLabsDeepFakeGenerator import ElevenLabsDeepFakeGenerator

def main():

    file_path = '/home/ubuntu/sample.csv'
    output_dir = '/home/ubuntu/data/wavefake_data/generated_audio/11LabsDeepFakes/'
    source_col = 'Real'
    transcript_col = 'transcript_1'
    deepfake_machine = ElevenLabsDeepFakeGenerator()

    deepfake_machine.generateDeepFakeFromDataFrame(file_path, output_dir, source_col, transcript_col, 'qeVSLLD6XfJjhEBj9qYp', '.wav', ['"'])


main()
