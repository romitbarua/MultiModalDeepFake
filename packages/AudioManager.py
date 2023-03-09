from pydub import AudioSegment
import os

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

        import_audio = AudioSegment.from_file(audio_path)

        if isinstance(output_file_name, type(None)):
            output_file_name = os.path.basename(audio_path)
        output_file_name = output_file_name.replace(os.path.splitext(output_file_name)[1], output_format)

        if isinstance(output_dir, type(None)):
            output_dir = os.path.dirname(audio_path)
        
        import_audio.export(os.path.join(output_dir, output_file_name), format=output_format)

        if delete_original:
            os.remove(audio_path)

