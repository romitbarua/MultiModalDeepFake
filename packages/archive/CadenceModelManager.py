from math import trunc
import sys
import random
import pandas as pd
import os
import pathlib
import yaml
import disvoice
import librosa 
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import scipy.stats as stats
from numpy import diff
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class CadenceModelManager:

    def __init__(self, data, silence_threshold: float = 0.005, low_pass_filter_cutoff: int = 10, trunc_window_size: int = 100) -> None: # Silence threshold assumed 0.5% 

        self.data = data
        #self.silence_threshold = silence_threshold
        self.low_pass_filter_cutoff = low_pass_filter_cutoff
        #self.window_size = trunc_window_size
        self.sr = sr = librosa.load(self.data['path'][0])[1]

        ## RB COMMENT: FOR NOW I AM STORING THE NORMALIZED AND TRUNCATED AUDIO, BUT MAYBE WE WANT TO MOVE TO A SINGLE VAR
        ## FROM A SPACE EFFICIENCY PERSPECTIVE (ESPECIALLY AS OUR LJ DATASET GROWS, IT MAY BE A LOT TO SAVE IN MEMORY)
        self.normalized_audio = []
        self.truncated_audio = []
        
    def generate_features(self, window_size, silence_threshold):
        
        ## PREPROCESS
        
        ## RB COMMENT: COMMENTING OUT -> DONE IN MODULE 1
        # Extract input files
        #print('Extracting input files')
        #all_wav_files, all_flags = extract_input_files(data_input_path)
        
        ## RB COMMENT: MOVED INTO THE INNIT FUNCTION
        # Obtain sample rate
        #sr = librosa.load(self.data['path'][0])[1]
        
        ## RB COMMENT: COMMENTING OUT -> DONE IN MODULE 1
        # Balance data
        #print('Balancing data')
        # TO DO: RELABANCE WITH NEW FOLDERS PLAY.HT AND UBERDUCK
        #rebalanced_wav_files, rebalanced_flags = balance_data(all_wav_files)
        
        # Normalise amplitudes
        print('Normalizing amplitudes')
        self.normalize_audio_amplitudes()
        
        # Hyperparam tune 
        #window_size, silence_threshold = self.hyperparam_search()
        
        # Save down the best params and proceed 
        
        ## RB COMMENT: DO WE NEED THE START_IDS AND END_IDS? NOT SEEING THEM USED ANYWHERE ELSE
        # Truncate silences
        print('Truncating silences')
        start_ids, end_ids = self.truncate_silences(window_size, silence_threshold)
        
        ## FEATURE ENGINEERING
        # Extract pauses 
        print('Extracting pauses')
        pauses = self.run_all_files(self.get_silence, window_size, silence_threshold)

        # Extract pause spreads
        print('Extracting pause spreads')
        silence_spreads = self.run_all_files(self.get_silence_spread, window_size, silence_threshold)

        # Extract amplitude and derivative
        print('Extracting amplitude features')
        amps = self.run_all_files(self.get_amplitude, window_size, silence_threshold)
        
        ## FEATURE CONSOLIDATION
        # Create dataframe 
        print('Creating dataframe')
        features = pd.DataFrame({
                            'pause_ratio':[item['ratio_pause_voiced'] for item in pauses], 
                            'pause_mean':[item['mean_of_silences'] for item in silence_spreads], 
                            'pause_std':[item['spread_of_silences'] for item in silence_spreads],  
                            'n_pauses':[item['n_pauses'] for item in silence_spreads], 
                            'amp_deriv':[item['abs_deriv_amplitude'] for item in amps],
                            'amp_mean':[item['mean_amplitude'] for item in amps]
                            })
        
        print('Complete')
        
        return features
        

    def run_cadence_feature_extraction_pipeline(self, window_size, silence_threshold, scaler = None):
        

        features = self.generate_features(window_size, silence_threshold)
        

        full_df = pd.concat((self.data, features), axis=1)
        
        if scaler is None:
            scaler = MinMaxScaler()
            full_df.loc[full_df['type'] == 'train', list(features.columns)] = scaler.fit_transform(full_df.loc[full_df['type']=='train', list(features.columns)])
            full_df.loc[~(full_df['type'] == 'train'), list(features.columns)] = scaler.transform(full_df.loc[~(full_df['type']=='train'), list(features.columns)])
            return full_df, list(features.columns), scaler
        else:
            full_df.loc[:, list(features.columns)] = scaler.transform(full_df.loc[:, list(features.columns)])
            return full_df, list(features.columns), None
            
                                                                                   

        


    ################################################## MODEL TESTING SCRIPT ##################################################
    '''
    def run_cadence_test(data_input_path, flags = [1], silence_threshold = 0.005, low_pass_filter_cutoff = 10):
        
        ## PREPROCESS
        # Extract input files
        print('Extracting input files')
        all_wav_files, _ = extract_input_files(data_input_path)
        
        # Obtain sample rate
        sr = librosa.load(all_wav_files[0])[1]
        
        # Normalise amplitudes
        print('Normalizing amplitudes')
        normalized_audios = normalize_audio_amplitudes(all_wav_files)
        
        # Truncate silences
        print('Truncating silences')
        truncated_audios, start_ids, end_ids = truncate_silences(normalized_audios, silence_threshold, window_size=100)
        
        ## FEATURE ENGINEERING
        # Extract pauses 
        print('Extracting pauses')
        r_pauses, f_pauses = run_all_files(truncated_audios, flags, get_silence, silence_threshold)

        # Extract pause spreads
        print('Extracting pause spreads')
        r_silence_spreads, f_silence_spreads = run_all_files(truncated_audios, flags, get_silence_spread, silence_threshold)

        # Extract amplitude and derivative
        print('Extracting amplitude features')
        r_amps, f_amps = run_all_files(truncated_audios, flags, get_amplitude, silence_threshold, sample_rate=sr, cutoff_frequency=low_pass_filter_cutoff)
        
        ## FEATURE CONSOLIDATION
        # Create dataframe 
        print('Creating dataframe')
        features = pd.DataFrame({'file': all_wav_files, 
                            'pause_ratio':[item['ratio_pause_voiced'] for item in r_pauses + f_pauses], 
                            'pause_mean':[item['mean_of_silences'] for item in r_silence_spreads + f_silence_spreads], 
                            'pause_std':[item['spread_of_silences'] for item in r_silence_spreads + f_silence_spreads],  
                            'n_pauses':[item['n_pauses'] for item in r_silence_spreads + f_silence_spreads], 
                            'amp_deriv':[item['abs_deriv_amplitude'] for item in r_amps + f_amps],
                            'amp_mean':[item['mean_amplitude'] for item in r_amps + f_amps], 
                            'fake':flags})
        
        
        print('Complete')

        return features
    '''


    ################################################## FUNCTIONS ##################################################

    '''
    # Begin with list of files; here we use an example template while we await the full class ouput
    def extract_input_files(self, data_input_path):

        all_wav_files = pathlib.Path(data_input_path)
        all_wav_files = list(all_wav_files.rglob("*.wav")) + list(all_wav_files.rglob("*.WAV"))
        all_wav_files = [str(file) for file in all_wav_files]

        #real_resampled_wav_files = [file for file in all_wav_files if 'TIMIT converted' in file]
        #fake_resampled_wav_files = [file for file in all_wav_files if not 'TIMIT converted' in file]

        flags = [1 if 'TIMIT converted' in str(item) else 0 for item in all_wav_files]
        
        return all_wav_files, flags #real_resampled_wav_files, fake_resampled_wav_files, flags
    '''


    # NEED TO CHECK THIS - does it do it by architecture?
    '''
    def balance_data(self, all_wav_files):
        
        folders = set([all_wav_files[i].split('_')[-1].split('.')[0] for i in range(len(all_wav_files))])
        
        real_resampled_wav_files = [file for file in all_wav_files if 'TIMIT converted' in file]
        fake_resampled_wav_files = [file for file in all_wav_files if not 'TIMIT converted' in file]
        
        # Ensure we take the same number of each phrase for real and fake, downsample the fake files 
        balanced_real_resampled_wav_files = []
        balanced_fake_resampled_wav_files = []
        
        for folder in folders:
            real_examples = [file for file in real_resampled_wav_files if f'_{folder}.' in file]
            fake_examples = [file for file in fake_resampled_wav_files if f'_{folder}.' in file]

            if len(real_examples) > len(fake_examples):
                real_examples = random.sample(real_examples, len(fake_examples))
            else:
                fake_examples = random.sample(fake_examples, len(real_examples))

            [balanced_real_resampled_wav_files.append(file) for file in real_examples]
            [balanced_fake_resampled_wav_files.append(file) for file in fake_examples]
        
        rebalanced_wav_files = balanced_real_resampled_wav_files + balanced_fake_resampled_wav_files
        rebalanced_flags = [i for i in np.zeros(len(balanced_real_resampled_wav_files))] + [i for i in np.ones(len(balanced_fake_resampled_wav_files))] 
        
        return rebalanced_wav_files, rebalanced_flags
    '''

    def normalize_audio_amplitudes(self):
        
        for file in self.data['path']:
            sample = librosa.load(file)[0]
            max_abs = np.max(np.abs(sample))
            normalized_sample = sample/max_abs
            self.normalized_audio.append(normalized_sample)

    def truncate_silences(self, window_size, silence_threshold, counter=0):

        start_ids = []
        end_ids = []
        
        for audio in self.normalized_audio:
            counter += 1
            if counter % 100 == 0:
                print(f'Truncating audio {counter}/{len(self.normalized_audio)} ({round(counter*100/len(self.normalized_audio))}%)')

            for j in range(len(audio)):
                roll_average = np.mean(np.abs(audio[j:j+window_size]))
                if roll_average > silence_threshold:
                    truncation_id_start = j
                    break

            for j in reversed(range(len(audio))):
                roll_average = np.mean(np.abs(audio[j-window_size:j]))
                if roll_average > silence_threshold:
                    truncation_id_end = j-window_size
                    break
            self.truncated_audio.append(audio[truncation_id_start:truncation_id_end])
            start_ids.append(truncation_id_start)
            end_ids.append(truncation_id_end)
        
        return start_ids, end_ids

    def moving_average(self, x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    def get_silence(self, audio, window_size, silence_threshold):
        thresh = max(abs(audio))*silence_threshold
        
        moving_avg = self.moving_average(abs(audio), window_size) # Window size = 100 

        silent = np.where(abs(moving_avg) < thresh)
        voiced = np.where(abs(moving_avg) >= thresh)
        
        pct_pause = len(silent[0])*100/(len(silent[0])+len(voiced[0]))
        pct_voiced = len(voiced[0])*100/(len(silent[0])+len(voiced[0]))
        ratio_pause_voiced = len(silent[0])/len(voiced[0]) 

        return {'pct_pause':pct_pause, 'pct_voiced': pct_voiced, 'ratio_pause_voiced': ratio_pause_voiced}

    def get_silence_spread(self, audio, window_size, silence_threshold):

        thresh = max(abs(audio))*silence_threshold
        
        moving_avg = self.moving_average(abs(audio), window_size) # Window size = 100 

        silent_windows = np.where(moving_avg < thresh)
        moving_avg[silent_windows] = 0
        silence_count = 0
        silence_counts = []
        
        for i in range(len(moving_avg)-1):
            item = moving_avg[i]
            next_item = moving_avg[i+1]
            
            if item != 0 and next_item == 0:
                silence_count = 0
                
            elif item == 0 and next_item == 0:
                silence_count += 1
                
            elif item == 0 and next_item != 0:
                silence_counts.append(silence_count)
            
            else:
                continue  
        
        # Get spreads/means and normalise
        spread_of_silences = np.std(silence_counts)/len(moving_avg)
        mean_of_silences = np.mean(silence_counts)/len(moving_avg)
        n_pauses = len(silence_counts)
            
        return {'spread_of_silences':spread_of_silences, 'mean_of_silences':mean_of_silences, 'silence_counts':silence_counts, 'n_pauses':n_pauses}

    '''
    def run_all_files(self, truncated_audios, flags, function, percent, sample_rate=None, cutoff_frequency=None):
        # Instantiate results - r=real, f=fake
        r_results = []
        f_results = []

        real_indices = np.array([int(i) for i in range(len(flags)) if flags[i] == 0])
        fake_indices = np.array([int(i) for i in range(len(flags)) if flags[i] != 0])
        
        real_examples = [truncated_audios[i] for i in real_indices]
        fake_examples = [truncated_audios[i] for i in fake_indices]

        for item in real_examples:
            r_result = function(item, percent, sample_rate, cutoff_frequency)
            r_results.append(r_result)

        for item in fake_examples:
            f_result = function(item, percent, sample_rate, cutoff_frequency)
            f_results.append(f_result)
        
        return r_results, f_results
    '''

    def run_all_files(self, function, window_size, silence_threshold):
        results = []
        for item in self.truncated_audio:
            results.append(function(item, window_size, silence_threshold))
        return results

    def filter_signal(self, audio):
        t = np.arange(len(audio)) / self.sr
        w = self.low_pass_filter_cutoff / (self.sr / 2) 
        b, a = signal.butter(5, w, 'low')
        smoothed_signal = signal.filtfilt(b, a, audio)
        
        return smoothed_signal

    def get_amplitude(self, audio, window_size, silence_threshold):

        abs_audio = abs(audio)
        smoothed_signal = self.filter_signal(abs_audio)
        
        deriv_amplitude = np.mean(diff(smoothed_signal))
        mean_amplitude = np.mean(smoothed_signal)
        
            
        return {'abs_deriv_amplitude':abs(deriv_amplitude), 'mean_amplitude':mean_amplitude}
    
    '''
    def hyperparam_search(self, window_size_options = [50, 100, 150, 250, 500, 750, 1000, 2000] , silence_threshold_options = np.linspace(0.005, 0.1, 20), model = 'logistic_regression', scaler = None):
        
        window_sizes = []
        silence_thresholds = []
        performances = []
        
        input_audios = self.normalized_audio
        input_labels = self.data['label']
        
        for window_size in window_size_options:
            print(f'Window size: {window_size}')
            for silence_threshold in silence_threshold_options:
        
                ########### Generate features ###########
                start_ids, end_ids = self.truncate_silences(window_size, silence_threshold)

                #print('Tuning pauses')
                pauses = self.run_all_files(self.get_silence, window_size, silence_threshold)

                # Extract pause spreads
                #print('Tuning pause spreads')
                silence_spreads = self.run_all_files(self.get_silence_spread, window_size, silence_threshold)

                # Extract amplitude and derivative
                #print('Tuning amplitude features')
                amps = self.run_all_files(self.get_amplitude, window_size, silence_threshold)

                ## FEATURE CONSOLIDATION
                # Create dataframe 
                #print('Creating dataframe')
                features = pd.DataFrame({
                                    'pause_ratio':[item['ratio_pause_voiced'] for item in pauses], 
                                    'pause_mean':[item['mean_of_silences'] for item in silence_spreads], 
                                    'pause_std':[item['spread_of_silences'] for item in silence_spreads],  
                                    'n_pauses':[item['n_pauses'] for item in silence_spreads], 
                                    'amp_deriv':[item['abs_deriv_amplitude'] for item in amps],
                                    'amp_mean':[item['mean_amplitude'] for item in amps]
                                    })


                full_df = pd.concat((self.data, features), axis=1)

                if scaler is None:
                    scaler = MinMaxScaler()
                    full_df.loc[full_df['type'] == 'train', list(features.columns)] = scaler.fit_transform(full_df.loc[full_df['type']=='train', list(features.columns)])
                    full_df.loc[~(full_df['type'] == 'train'), list(features.columns)] = scaler.transform(full_df.loc[~(full_df['type']=='train'), list(features.columns)])
                else:
                    full_df.loc[:, list(features.columns)] = scaler.transform(full_df.loc[:, list(features.columns)])

                ######## END ########

                #print(full_df.columns)
                #print(full_df.head())

                X_train = full_df[['pause_ratio', 'pause_mean', 'pause_std', 'n_pauses', 'amp_deriv', 'amp_mean']][full_df['type']=='train']
                y_train = full_df['label'][full_df['type']=='train']
                X_dev = full_df[['pause_ratio', 'pause_mean', 'pause_std', 'n_pauses', 'amp_deriv', 'amp_mean']][full_df['type']=='dev']
                y_dev = full_df['label'][full_df['type']=='dev']
                
                #if nans in full_df:
                #    
                #    continue

                if model == 'logistic_regression':

                    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
                    score = clf.score(X_dev, y_dev)
                    
                    print(f'Windowsize {window_size}, silence_threshold {silence_threshold}: {score}')
                    
                window_sizes.append(window_size)
                silence_thresholds.append(silence_threshold)
                performances.append(score)

        
        optimal_window_size = window_sizes[np.argmax(performances)]
        optimal_silence_threshold = silence_thresholds[np.argmax(performances)]
        
        print(f'OPTIMAL PARAMS SELECTED: window size: {optimal_window_size}, silence threshold: {optimal_silence_threshold}')
        
        return optimal_window_size, optimal_silence_threshold
    '''
    
    def target_function(self, window_size, silence_threshold, label_col = 'label', model=LogisticRegression(random_state=12)):
        
        features, feature_cols = self.run_cadence_feature_extraction_pipeline(window_size, silence_threshold)
        
        X_train = features.loc[features['type'] == 'train', feature_cols]
        y_train = features.loc[features['type'] == 'train', label_col]
        X_dev = features.loc[features['type'] == 'dev', feature_cols]
        y_dev = features.loc[features['type'] == 'dev', label_col]
        
        model.fit(X_train, y_train)
        score = model.score(X_dev, y_dev)
        
        return score
    
    def run_target_function(self, x)
        
    
    def hyperparam_search(self, feature_dir):
            
            
        
        window_size_options = [50, 100, 150, 250, 500, 750, 1000, 2000]
        silence_threshold_options = np.linspace(0.005, 0.1, 20)
        
        best_acc = 0
        best_window_size = None
        best_silence_threshold = None
        
        for window_size in window_size_options:
            for silence_threshold in silence_threshold_options:
                
                #get the unique architectures
                archs = self.data['architecture'].unique()
                
                #load the data
                ## Note that we only have 200 data points, so we wont be joining to the main self.data
                ## this is just to see what window_size and silence threshold we want
                dfs = pd.DataFrame(columns=['path', 'pause_ratio', 'pause_mean', 'pause_std', 'n_pauses', 'amp_deriv', 'amp_mean'])
                
                
                for arch in archs:
                    file_name = f'{arch}_ws{window_size}_st{str(int(silence_threshold*10000))}.csv'
                    file_path = os.path.join(feature_dir, file_name)
                    dfs = dfs.append(pd.read_csv(file_path))
                    
                    
                
                    
                    
                    
                

                
                
                
                
        
