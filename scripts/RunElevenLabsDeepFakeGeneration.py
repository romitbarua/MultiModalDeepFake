import sys
sys.path.insert(1, '/Users/romitbarua/Documents/Berkeley/Spring 2023/MultiModalDeepFake/')

from packages.ElevenLabsDeepFakeGenerator import ElevenLabsDeepFakeGenerator

def main():

    file_path = '/Users/romitbarua/Downloads/labs_sample.csv'
    output_dir = '/Users/romitbarua/Documents/Berkeley/Spring 2023/MultiModalDeepFake/data/wavefake_data/ElevenLabsDeepFakes/'
    source_col = 'Real'
    transcript_col = 'transcript_1'
    deepfake_machine = ElevenLabsDeepFakeGenerator()

    deepfake_machine.generateDeepFakeFromDataFrame(file_path, output_dir, source_col, transcript_col, 'qeVSLLD6XfJjhEBj9qYp')


main()
