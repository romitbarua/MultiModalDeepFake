import time 
import sys, os, io

import os
import sys
sys.stderr = open(os.devnull, "w")  # silence stderr

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


sys.path.append("/home/ubuntu/MultiModalDeepFake")


AUDIO_PATH = '/home/ubuntu/data/demo/demo_hany_16KHz/short_clip_for_demo.wav'

def load_pkl_file(filepath):
    with open(filepath, 'rb') as file:
        return pickle.load(file)

def gen_embeddings(em_features):
    
    import nemo.collections.asr as nemo_asr 
    speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large')
    embeddings = speaker_model.get_embedding(AUDIO_PATH)
    embeddings_df = pd.DataFrame(embeddings)
    embeddings_df.columns = [str(col) for col in embeddings_df.columns]
    return embeddings_df[em_features]
    

def gen_openSmileFeatures(os_features):
    
    feature_extractor = opensmile.Smile(feature_set=opensmile.FeatureSet.ComParE_2016, feature_level=opensmile.FeatureLevel.Functionals)
    features = feature_extractor.process_file(AUDIO_PATH).reset_index()
    duration = features['end'] - features['start']
    duration = duration.astype('timedelta64[ms]')/1000
    features['duration(seconds)'] = duration
    return features[os_features]

def gen_cadenceFeatures(cadence_scalar):
    
    df = pd.DataFrame({'path':[AUDIO_PATH], 'type':['test']})
    cadence_manager = CadenceModelManager(df)
    cad_feature_df, cad_feature_cols, _ =  cadence_manager.run_cadence_feature_extraction_pipeline(cadence_scalar)
    return cad_feature_df[cad_feature_cols]

def predict(model, features, model_name):
    proba = model.predict_proba(features)[0]
    pred = np.argmax(proba)
    confidence = proba[pred]
    pred_type = 'Real' if pred == 0 else 'Fake'
    print(f'{model_name}: {pred_type} (Confidence: {confidence*100:.0f}%)')
    

proceed = input("Press Y to record audio: ")

proceed = 'y'

if proceed == 'y':

    print('Recording audio')
    for i in range(0, 7):
        print('*')
        time.sleep(1)
    print('Audio captured')

    #text_trap = io.StringIO()
    #sys.stdout = text_trap
    
    print('Debug 1')

    import pandas as pd
    import pickle
    import numpy as np
    from packages.LJDataLoader import LJDataLoader
    from packages.AudioEmbeddingsManager import AudioEmbeddingsManager
    from packages.ModelManager import ModelManager
    from packages.CadenceModelManager import CadenceModelManager
    import packages.AnalysisManager as am
    from packages.SmileFeatureManager import SmileFeatureManager
    import opensmile


    print('Generating Features...')

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    embeddings_model = load_pkl_file('/home/ubuntu/data/demo/models/titanet.pkl')
    openSmile_model = load_pkl_file('/home/ubuntu/data/demo/models/openSmile.pkl')
    cadence_model = load_pkl_file('/home/ubuntu/data/demo/models/cadence.pkl')
    openSmile_features = load_pkl_file('/home/ubuntu/data/demo/models/openSmile_features.pkl')
    em_features = load_pkl_file('/home/ubuntu/data/demo/models/em_features.pkl')
    cadence_scalar = load_pkl_file('/home/ubuntu/data/demo/models/cadence_scalar.pkl')

    emb_feat = gen_embeddings(em_features)
    cad_feat = gen_cadenceFeatures(cadence_scalar)
    os_feat = gen_openSmileFeatures(openSmile_features)

    for i in range(0, 5):
        print('*')
        time.sleep(1)

    predict(cadence_model, cad_feat, 'Cadence Features')
    predict(openSmile_model, os_feat, 'OpenSmile Features')
    predict(embeddings_model, emb_feat, 'TitaNet Embeddings')

    print('COMPLETED')
