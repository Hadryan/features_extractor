import os, sys
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from extractor import FeaturesExtractor

""" Automatic feature extraction from respiratory database.

    This script organizes respiratory database metadata into a pandas dataframe. The dataframe includes
    filenames and breath start/stop times as well as appropriate labels (wheeze/crackle). This is fed to a
    FeatureExtractor object. From there a matrix of features are extracted to the variable 'feature_data,' 
    which is in turn imported into the evaluation.py script for SVM classification experiments.
    
    Full feature extraction for this dataset takes upwards of 90 minutes with Gini filtering turned off.
    With Gini filtering on, it takes more than 10 hours. For this reason, a pre-loaded feature_data.pickle 
    is supplied and loaded by default. 
    
    If you would like to see how FeatureExtractor works in a short period of time, set do_demo to True. This
    will extract features from the first 10 breath excerpts.
    
    For the FeatureExtractor class itself, please see (the heavily annotated) extractor.py script.
    
    EXAMPLE USE from TERMINAL: 
    
    python3 resp_extraction.py 1 1          --- extract features from entire resp dataset, plot as matrix.
                                                saves features as demo_data.pickle
                                                
    python3 resp_extraction.py 1            --- short extract: plot from first 10 clips, plot result as matrix.
                                                saves features as short_demo_data.pickle

"""


# Defining helper functions.
def force_pickle_ext(save_as_filename):
    base, ext = os.path.splitext(save_as_filename)
    if ext is not ".pickle":
        save_as_filename = base + '.pickle'
        print('File name changed to ' + save_as_filename)
    return save_as_filename


def save_pickle(var_to_save, save_as_filename):
    save_as_filename = force_pickle_ext(save_as_filename)
    outfile = open(save_as_filename, 'wb')
    pickle.dump(var_to_save, outfile)
    outfile.close()


def get_precomputed_features():
    with open('feature_data.pickle', 'rb') as openfile:
        feature_data = pd.DataFrame()
        feature_data = pickle.load(openfile)
    return feature_data


def get_without_outliers(data, z_score=4):
    #   Assuming normal distribution, remove rows with outliers.
    rows_with_outliers = (np.abs(data) > z_score).any(1)
    return data[rows_with_outliers == False]


def get_filenames_labels(audio_dir):
    filenames_labels = []
    print('Building database from metadata...')

    for filename in tqdm(os.listdir(audio_dir)):
        file_root, file_extension = os.path.splitext(filename)
        if file_extension == ".txt":
            filepath = os.path.join(audio_dir, filename)
            adventitious = np.genfromtxt(filepath, delimiter='\t')

            wavfilename = file_root + ".wav"
            for i in range(adventitious.shape[0]):
                t_start = adventitious[i, 0]
                t_stop = adventitious[i, 1]
                has_crackle = adventitious[i, 2]
                has_wheeze = adventitious[i, 3]
                filenames_labels.append([wavfilename, t_start, t_stop,
                                         has_crackle, has_wheeze])

    filenames_labels = pd.DataFrame(filenames_labels,
                                    columns=["filename", "t_start", "t_stop",
                                             "has_crackle", "has_wheeze"])
    return filenames_labels


def extract_feature_data(filenames_labels, audio_dir, long_extract):
    if not long_extract:
        print('Short extraction -- using first 10 entries only.')
        fe = FeaturesExtractor(filenames_labels.iloc[:10, :], audio_dir,
                               do_gini_filter=True, resample_freq=16000)
        fe.extract()
        save_pickle(fe.feature_data, 'short_demo_data')
        feature_data = fe.feature_data
    else:
        fe = FeaturesExtractor(filenames_labels, audio_dir,
                               do_gini_filter=True, resample_freq=16000)
        fe.extract()
        save_pickle(fe.feature_data, 'demo_data')
        feature_data = fe.feature_data

    return feature_data


if __name__ == '__main__':
    if len(sys.argv) > 1:
        do_demo = int(sys.argv[1])
    else:
        do_demo = 0

    if len(sys.argv) > 2:
        do_long = int(sys.argv[2])
    else:
        do_long = 0

    if do_demo:
        audio_dir = "./Respiratory_Sound_Database/audio_and_txt_files/"
        filenames_labels = get_filenames_labels(audio_dir)
        feature_data = extract_feature_data(filenames_labels, audio_dir, long_extract=do_long)
        plt.imshow(feature_data, aspect='auto')
        plt.title('Extracted feature data (no YAMNet)')
        plt.xlabel('Features')
        plt.ylabel('Extract')
        plt.show()
    else:
        feature_data = get_precomputed_features()
else:
    feature_data = get_precomputed_features()
    feature_data = get_without_outliers(feature_data, 10)
