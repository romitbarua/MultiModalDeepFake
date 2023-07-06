import sys
sys.path.append("/home/ubuntu/MultiModalDeepFake")
import time

from packages.CadenceModelManager import CadenceModelManager
from packages.LJDataLoader import LJDataLoader

arch = 'ElevenLabs'

file_path = '/home/ubuntu/data/wavefake_data/LJ_metadata_16000KHz.csv'
loader = LJDataLoader(data_path=file_path)

source_architectures = ['Full_Band_MelGan', 'HifiGan', 'MelGan', 'MelGanLarge', 'Multi_Band_MelGan', 'Parallel_WaveGan', 'Waveglow']
new_col_name = 'RandWaveFake'
loader.selectRandomArchitecture(target_col=new_col_name, source_cols=source_architectures)

loader.splitData()
data_df = loader.generateFinalDataFrame(real_col='Real', fake_cols=[arch])

cad_model = CadenceModelManager(data_df)


start_time = time.time()
output_dir = '/home/ubuntu/data/wavefake_data/Cadence_features/16khz/'
output_name = arch
cad_model.hyperparam_search_and_featues(output_dir=output_dir, output_name=output_name, n_iter=50, sample_size=500, init_ex_count=25)
print("--- %s seconds ---" % (time.time() - start_time))