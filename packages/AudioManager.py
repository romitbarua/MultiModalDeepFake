from pydub import AudioSegment
import os
from packages.LibrosaManager import LibrosaManager 
import soundfile as sf
from random import randint

class AudioManager:

    def __init__(self) -> None:
        pass

    def convertAudioDirectory(self,
                                audio_dir: str,
                                input_format: str,
                                output_format: str = '.wav',
                                output_dir: str = None,
                                delete_original: bool = False,
                                bitrate: str = None,
                                codec: str = None):

        for file in os.listdir(audio_dir):

            if input_format in file:
                self.convertAudioFileTypes(os.path.join(audio_dir, file), output_format=output_format, 
                                            delete_original=delete_original, output_dir=output_dir,
                                            bitrate=bitrate, codec=codec)
                


    def convertAudioFileTypes(self, audio_path: str,
                                output_format: str = '.wav',
                                delete_original: bool = False,
                                output_dir: str = None,
                                output_file_name: str = None,
                                bitrate: str = None,
                                codec: str = None):
        
        assert output_format in ['.wav', '.mp4'], f'{output_format} is an invalid output format. Please enter types: (.wav, .mp4).'
        
        try:
            import_audio = AudioSegment.from_file(audio_path)

            if isinstance(output_file_name, type(None)):
                output_file_name = os.path.basename(audio_path)
            output_file_name = output_file_name.replace(os.path.splitext(output_file_name)[1], output_format)

            if not output_dir:
                output_dir = os.path.dirname(audio_path)

            import_audio.export(os.path.join(output_dir, output_file_name),
                                format=output_format.replace('.', ''),
                                codec=codec,
                                bitrate=bitrate)

            if delete_original:
                os.remove(audio_path)
                
        except Exception as e:
            print(f'Failed to Convert Audio File: {audio_path}')
            print('Error: ', e)

    def resampleAudioDirectory(self, input_directory: str, output_directory: str, target_sample_rate: int, replace_existing: bool = False):
        
        for file in os.listdir(input_directory):
            if not replace_existing:
                if os.path.isfile(os.path.join(output_directory, file)):
                    continue
            
            try:
                librosa_manager = LibrosaManager(os.path.join(input_directory, file))
                resampled_audio = librosa_manager.resample(target_sample_rate)
                sf.write(os.path.join(output_directory, file), resampled_audio, target_sample_rate, subtype='PCM_24')
            except Exception as e:
                print(f'Failed to Resample: {file}')
                print(f'Error Msg: {e}')
                print()


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
                                delete_original: bool = False,
                                bitrate: str = None,
                                codec: str = None):

        for file in os.listdir(audio_dir):

            if input_format in file:
                self.convertAudioFileTypes(os.path.join(audio_dir, file), output_format=output_format, 
                                            delete_original=delete_original, output_dir=output_dir,
                                            bitrate=bitrate, codec=codec)
                


    def convertAudioFileTypes(self, audio_path: str,
                                output_format: str = '.wav',
                                delete_original: bool = False,
                                output_dir: str = None,
                                output_file_name: str = None,
                                bitrate: str = None,
                                codec: str = None):
        
        assert output_format in ['.wav', '.mp4'], f'{output_format} is an invalid output format. Please enter types: (.wav, .mp4).'
        
        try:
            import_audio = AudioSegment.from_file(audio_path)

            if isinstance(output_file_name, type(None)):
                output_file_name = os.path.basename(audio_path)
            output_file_name = output_file_name.replace(os.path.splitext(output_file_name)[1], output_format)

            if not output_dir:
                output_dir = os.path.dirname(audio_path)

            import_audio.export(os.path.join(output_dir, output_file_name),
                                format=output_format.replace('.', ''),
                                codec=codec,
                                bitrate=bitrate)

            if delete_original:
                os.remove(audio_path)
                
        except Exception as e:
            print(f'Failed to Convert Audio File: {audio_path}')
            print('Error: ', e)

    def resampleAudioDirectory(self, input_directory: str, output_directory: str, target_sample_rate: int, replace_existing: bool = False):
        
        for file in os.listdir(input_directory):
            if not replace_existing:
                if os.path.isfile(os.path.join(output_directory, file)):
                    continue
            
            try:
                librosa_manager = LibrosaManager(os.path.join(input_directory, file))
                resampled_audio = librosa_manager.resample(target_sample_rate)
                sf.write(os.path.join(output_directory, file), resampled_audio, target_sample_rate, subtype='PCM_24')
            except Exception as e:
                print(f'Failed to Resample: {file}')
                print(f'Error Msg: {e}')
                print()
                
    
    def add_noise_with_snr(audio_path: str, snr_range: list = [10, 80]):
        audio, sr = librosa.load(audio_path)
        
        audio_power = np.mean(audio ** 2)
        
        noise_snr = randint(snr_range[0], snr_range[1])
        noise_power = audio_power / (10 ** (noise_snr / 10))
        noise = np.random.normal(scale=np.sqrt(noise_power), size=len(audio))
        noisy_audio = audio + noise

        return noisy_audio
    
    def launderAudioDirectory(self, input_directory: str, output_directory: str, noise_type: str = 'random_gaussian', replace_existing: bool = False):
        
        for file in os.listdir(input_directory):
            if not replace_existing:
                if os.path.isfile(os.path.join(output_directory, file)):
                    continue
            
            try:
                if noise_type == 'random_gaussian':
                    noisy_audio = add_noise_with_snr(audio_path=file)
                    
                sf.write(os.path.join(output_directory, file), noisy_audio, target_sample_rate, subtype='PCM_24')
            except Exception as e:
                print(f'Failed to add noise: {file}')
                print(f'Error Msg: {e}')
                print()


