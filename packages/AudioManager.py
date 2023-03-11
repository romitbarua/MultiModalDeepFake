from pydub import AudioSegment
import os
from packages.LibrosaManager import LibrosaManager 
import soundfile as sf

class AudioManager:

    def __init__(self) -> None:
        pass

    def convertAudioDirectory(self,
                                audio_dir: str,
                                input_format: str,
                                output_format: str = '.wav',
                                output_dir: str = None,
                                delete_original: bool = True):

        for file in os.listdir(audio_dir):

            if input_format in file:
                self.convertAudioFileTypes(os.path.join(audio_dir, file), output_format=output_format, 
                                            delete_original=delete_original, output_dir=output_dir)


    def convertAudioFileTypes(self, audio_path: str,
                                output_format: str = '.wav',
                                delete_original: bool = True,
                                output_dir: str = None,
                                output_file_name: str = None):
        
        assert output_format in ['.wav', '.mp4'], 'Please enter valid output type (.wav, .mp4)'
        
        try:
            import_audio = AudioSegment.from_file(audio_path)

            if isinstance(output_file_name, type(None)):
                output_file_name = os.path.basename(audio_path)
            output_file_name = output_file_name.replace(os.path.splitext(output_file_name)[1], output_format)

            if isinstance(output_dir, type(None)):
                output_dir = os.path.dirname(audio_path)

            import_audio.export(os.path.join(output_dir, output_file_name), format=output_format.replace('.', ''))

            if delete_original:
                os.remove(audio_path)
                
        except:
            print(f'Failed to Convert Audio File: {audio_path}')

    def resampleAudioDirectory(self, input_directory: str, output_directory: str, target_sample_rate):
        
        for file in os.listdir(input_directory):
            try:
                librosa_manager = LibrosaManager(os.path.join(input_directory, file))
                resampled_audio = librosa_manager(target_sample_rate)
                sf.write(os.path.join(output_directory, file), resampled_audio, target_sample_rate, subtype='PCM_24')
            except Exception as e:
                print(f'Failed to Resample: {file}')
                print(f'Error Msg: {e}')
                print()


