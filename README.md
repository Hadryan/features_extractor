<pre>Audio features extractor class for Python focusing on respiratory auscultation applications.

FeatureExtractor class is located in the extractor.py file, and can be used independently of this
project demo. The project demo requires a minimal home set up, please see the section at the bottom 
of this readme.

*** 
    This code was ported from a Google Colab notebook, with some modification for offline use.
    A copy of the notebook can be found in this repository: 
    
    respiratory_audio_sorting_notebook.ipynb
***


FeatureExtractor has the following dependencies. They may be installed with pip3 install:

numpy
matplotlib
librosa
resampy
pandas
sklearn
heartpy
tqdm
scipy.signal

The project demo includes two primary scripts:

  resp_extraction.py      --- a demo extraction (or full extraction, if you wish) of features 
                              from Resp database.
  evaluation.py           --- runs two experiments evaluating the use of extracted features in 
                              audio classification.
  
Docstrings are copied below:::

(1) RESP_EXTRACTION:

Automatic feature extraction from respiratory database.

This script organizes respiratory database metadata into a pandas dataframe. The dataframe 
includes filenames and breath start/stop times as well as appropriate labels (wheeze/crackle). 
This is fed to a FeatureExtractor object. From there a matrix of features are extracted to
the variable 'feature_data,' which is in turn imported into the evaluation.py script for SVM 
classification experiments.

Full feature extraction for this dataset takes upwards of 90 minutes with Gini filtering turned
off. With Gini filtering on, it takes more than 10 hours. For this reason, a pre-loaded 
feature_data.pickle is supplied and loaded by default. 

If you would like to see how FeatureExtractor works in a short period of time, set do_demo to 
True. This will extract features from the first 10 breath excerpts.

For the FeatureExtractor class itself, please see (the heavily annotated) extractor.py script.

EXAMPLE USE from TERMINAL: 

python3 resp_extraction.py 1 1          --- extract features from entire resp dataset, plot as 
                                            matrix. 
                                            saves features as demo_data.pickle
                                            
python3 resp_extraction.py 1            --- short extract: plot from first 10 clips, plot result
                                            as matrix.
                                            saves features as short_demo_data.pickle

(2) EVALUATION:

Evaluation of SVM classification models trained on FeatureExtractor features.

This script runs two experiments as outlined in the final project paper. In each case
wheeze and crackle detection are treated as independent tasks (i.e., multi-label paradigm).
Two bar graphs are presented: the first representing crackle detection results, the second
for wheeze detection results. Bar graphs do not include titles to facilitate formatting for
the final paper in which they are copied.

EXPERIMENT 1:
Train an SVM on subsets of the total feature set. Here in three versions:
(1) MFCCs only.
(2) YAMNet embeds, frame-wise means only.
(3) YAMNet embeds, frame-wise means and standard deviations (a.k.a. 'YAMNet all').

EXPERIMENT 2:
Train an SVM on various PCA compressions of the total feature set. Starting with 10
principle components, the the test iterates up to 20 principle components.

EXAMPLE USE from TERMINAL:

python3 evaluation.py 1 1           --- runs both experiments and plot results as bar graphs.
python3 evaluation.py 1             --- runs first experiment only.
python3 evaluation.py 0 1           --- runs second experiment only.



:::HOME DEMO SETUP:::

The FeaturesExtractor class is designed to work independently of the final project scripts which
accompany it.  It is contained in the extractor.py file. Everything else is meant for demonstration 
purposes, as part of a final project submission for MUMT 621, Music Information Retrieval at McGill 
University, Montreal: Winter semester, 2020. The demonstration will need a .pickle file that cannot 
be uploaded to Github due to size restrictions. Please contact the author if you are interested.

feature_data.pickle must be placed in the project home directory.

The accompanying scripts also assume the user has downloaded this respiratory database:
https://www.kaggle.com/vbookshelf/respiratory-sound-database

and that it is unziped and placed in the script home directory, such that audio and text files can 
be found at:

[project directory]/Respiratory_Sound_Database/audio_and_txt_files/

</pre>

