import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import resampy
import heartpy
import scipy.signal
from tqdm.auto import tqdm
tqdm.pandas()


class FeaturesExtractor:
    """Constructs a database of audio features for respiratory auscultation.

    Extracts time-invariant MIR features from a list of files, as they are
    specified in a pandas dataframe under the column ['filename']. The features
    are standardized, assuming a normal distribution. After calculation, data are
    appended to the appropriate rows (which may include, e.g., label information
    for supervised learning). Current implementation allows for specifying extract
    start and stop times, potentially augmenting the size of the dataset.

    To extract features use the self.extract() method.

    Arguments:
    filenames_labels    --  a pandas dataframe containing at least one column
                            called 'filenames'
    audio_dir           --  a string specifying the directory with audio files.

    Outputs:
    self.feature_data   --  pandas dataframe with 'filenames' column replaced
                            by multiple columns of feature data.

    """

    def __init__(self, filenames_labels, audio_dir, do_gini_filter=False,
                 do_spline_interp=False, resample_freq=16000):
        self.filenames_labels = filenames_labels.reset_index()
        self.audio_dir = audio_dir
        self.do_gini_filter = do_gini_filter
        self.do_spline_interp = do_spline_interp
        self.resample_freq = resample_freq
        self.fill_na = True
        self.filename = str()
        self.last_filename = str()
        self.t_start = str()
        self.t_stop = str()
        self.untrimmed_x = np.array([])
        self.x = np.array([])
        self.sr = int()
        self.feature_data = []
        # self.yamnet_embed_model = Model()
        self.MODEL_PATH = ("/content/gdrive/My Drive/Colab Notebooks/models/"
                           "research/audioset/yamnet/yamnet.h5")
        self.EPS = 1e-12

    @staticmethod
    def get_safe_divisor(std_):
        std_[std_ == 0] = std_[std_ == 0] + 1
        return std_

    # @staticmethod
    # def verify_sr(x, sr):
    #     if sr is not params.SAMPLE_RATE:
    #         x = resampy.resample(x, sr, params.SAMPLE_RATE)
    #     return x

    @staticmethod
    def get_framed(x, frame_length):
        x_pad = np.pad(x, int(frame_length // 2), mode='edge')
        return librosa.util.frame(x_pad, frame_length, int(frame_length // 4))

    @staticmethod
    def get_bandpass(low_f, high_f, x, sr):
        nyquist = int(sr / 2)
        b, a = scipy.signal.butter(6, [low_f / nyquist, high_f / nyquist], btype='bandpass')
        return scipy.signal.lfilter(b, a, x)

    def get_gini_index(self, array_):
        array_ = np.sort(array_) + self.EPS
        i = np.arange(0, array_.shape[0]) + 1
        N = array_.shape[0]
        return 1 - 2 * np.sum(np.abs(array_) * (N - i + 0.5) / N) / np.sum(array_)

    def get_short_time_energy(self, x, frame_length):
        framed = self.get_framed(x, frame_length=frame_length)
        return np.sum(framed ** 2, axis=0)

    def standardize(self, data):
        std_ = self.get_safe_divisor(data.std(axis=0))
        return (data - data.mean(axis=0)) / std_

    # def initialize_yamnet(self):
    #     yamnet = yamnet_model.yamnet_frames_model(params)
    #     yamnet.load_weights(self.MODEL_PATH)
    #     inp = yamnet.input
    #     out = yamnet.layers[-3].output
    #     self.yamnet_model = Model(inp, out)

    def get_x_index(self):
        i_start = int(self.t_start * self.sr)
        i_stop = int(self.t_stop * self.sr)
        return i_start, i_stop

    def set_x(self):
        if self.filename is not self.last_filename:
            filepath = os.path.join(self.audio_dir, self.filename)
            self.untrimmed_x, self.sr = librosa.load(filepath)

        if self.t_start:
            i_start, i_stop = self.get_x_index()
            self.x = self.untrimmed_x[i_start:i_stop]
        else:
            self.x = self.untrimmed_x[:]

        self.x /= np.max(self.x)
        self.last_filename = self.filename[:]
        self.clean_x()

    def clean_x(self):
        if self.resample_freq is not None:
            self.set_sr_and_filter()
        if self.do_gini_filter:
            self.gini_filter_x()
        if self.do_spline_interp:
            self.spline_interp_x()

    def set_sr_and_filter(self, low_f=150):
        """Bandpass filter and resampling audio file.

        Traditional auscultation sounds are lowpass in nature due to the dampening
        of high frequencies by the chest wall, so frequencies above 2 kHz can be
        safely attenuated. Bands below 150 Hz tend to be contaminated by heart-beat
        sounds, and this range is also typically filtered out. Lowpass filtering
        permits resampling at a lower sample rate which saves on computational
        cost.

        Pramono, Renard Xaviero Adhi, Syed Anas Imtiaz, and Esther Rodriguez-Villegas.
        2019. “Evaluation of features for classification of wheezes and normal
        respiratory sounds.” PLOS One 14 (3): 21. https://doi.org/10.1371
        /journal.pone.0213659.



        """

        y = self.get_bandpass(low_f, self.resample_freq // 4, self.x, self.sr)
        self.x = resampy.resample(y, self.sr, self.resample_freq)
        self.x /= np.max(self.x)
        self.sr = self.resample_freq

    def gini_filter_x(self, n_components=80):
        """Gini index median filtering for isolating tonal components.

        This cleaning operation is the first step in a chain of signal processing
        techniques used by Torre-Cruz et al. to isolate wheeze components of a
        respiratory signal. Columns representing tonal spectra in NMF factorization
        are thought to be more sparse, and hence to have a higher Gini index.

        Torre-Cruz, J., F. Canadas-Quesada, S. García-Galán, N. Ruiz-Reyes,
            P. Vera-Candeas, and J. Carabias-Orti. 2020. “A constrained tonal semi-
            supervised non-negative matrix factorization to classify presence/
            absence of wheezing in respiratory rounds.” Applied Acoustics 161: 13.
            https://doi.org/10.1016/j.apacoust.2019.107188.

        """

        stft_ = librosa.stft(self.x)
        spectrogram = np.abs(stft_)
        ginis_ = np.zeros(n_components)

        # NMF factorization.
        components, activations = \
            librosa.decompose.decompose(spectrogram, n_components=n_components)

        for i in range(n_components):
            ginis_[i] += self.get_gini_index(components[:, i])

        tonal_idxs = np.where(ginis_ > np.median(ginis_))[0]
        S = np.zeros(stft_.shape)

        for i in tonal_idxs:
            S += np.outer(components[:, i], activations[i, :])

        self.x = librosa.core.griffinlim(S)
        self.x /= np.max(self.x)

    def spline_interp_x(self):
        """Cubic spline interpolation of clipped signals.

        Emmanouilidou et al. (2018) propose a chain of cleaning techniques for
        noisy respiratory signals. Chief among them is cubic spline interpolation,
        meant to compensate for truncated waveforms as a result of clipping dis-
        tortion.

        Emmanouilidou, Dimitra, Eric D. McCollum, Daniel E. Park, and Mounya
        Elhilali. 2018. “Computerized lung sound screening for pediatric
        auscultation in noisy field environments.” IEEE Transactions on Biomedical
        Engineering 65 (7): 1564–74. https://doi.org/10.1109/TBME.2017.2717280.


        """

        y = heartpy.preprocessing.interpolate_clipping(self.x, self.sr, threshold=1)
        y = heartpy.preprocessing.interpolate_clipping(-y, self.sr, threshold=1)
        y /= np.max(y)
        self.x = -y

    def get_framewise_entropy(self, stft_):
        power_spec = np.abs(stft_ ** 2)
        distribution = power_spec / np.sum(power_spec, axis=0)
        spec_entropy = -np.diagonal(np.dot(distribution.T, np.log2(distribution)))
        return spec_entropy / np.log2(stft_.shape[0] + self.EPS)

    def get_mfccs(self):
        """Mel-frequency cepstral coefficients are commonly used in auscultation.

        Pramono, Renard, Stuart Bowyer, and Esther Rodriguez-Villegas. 2017.
            “Automatic adventitious respiratory sound analysis: A systematic
            review.” Edited by Thomas Penzel. PLOS One 12 (5). doi:10.1371
            /journal.pone.0177926.

        """

        mfccs = librosa.feature.mfcc(self.x, self.sr, n_mfcc=13)
        means = mfccs.mean(axis=1)
        stds = mfccs.std(axis=1)
        return np.hstack([means, stds])

    def get_chromagram(self):
        """ Chromagram peak as selected by regularization in Mendes et al. (2016).

        Mendes, L., I. M. Vogiatzis, E. Perantoni, E. Kaimakamis, I. Chouvarda,
            N. Maglaveras, J. Henriques, P. Carvalho, and R. P. Paiva. 2016.
            “Detection of crackle events using a multi-feature approach.” In 38th
            Annual International Conference of the IEEE Engineering in Medicine
            and Biology Society, 3679–3683.

        """

        chroma = librosa.feature.chroma_cqt(self.x, self.sr)
        peak = chroma.argmax(axis=0).mean()
        return np.hstack([peak])

    # def get_yamnet_embeds(self):
    #     """A pre-trained deep neural network trained on the Audioset-Youtube corpus.
    #
    #     The last 512-neuron layer, which was especially designed for a Youtube
    #     categorization task, has been removed to reveal a 1024 neuron embeddings
    #     layer. Embeddings are calculated by YAMNet per frame, as per its intention
    #     of real-time categorization. Here, embeddings are averaged and collected
    #     along with their standard deviations.
    #
    #     Plakal, Manoj and Dan Ellis. 2020. "YAMNet." GitHub repository.
    #         https://github.com/tensorflow/models/tree/master/research/audioset/yamnet
    #
    #     """
    #
    #     x = self.verify_sr(self.x, self.sr)
    #     embeddings = self.yamnet_model(np.reshape(x, [1, -1]))
    #     means = np.mean(embeddings, axis=0)
    #     stds = np.std(embeddings, axis=0)
    #     return np.hstack([means, stds])

    def get_voiced_score(self, frame_length=2048):
        """Rough estimation of voiced/unvoiced frame. Loosely based on:

        Bachu, R. G., S. Kopparthi, B. Adapa, and Buket D. Barkana. 2010. "Voiced/
            unvoiced decision for speech signals based on zero-crossing rate and
            energy." In Advanced Techniques in Computing Sciences and Software
            Engineering, 279-282.

        """

        zero_cross = librosa.feature.zero_crossing_rate(self.x, frame_length=frame_length)
        st_energy = self.get_short_time_energy(self.x, frame_length=frame_length)
        voiced_score = np.ravel(st_energy / zero_cross)
        voiced_score = np.nan_to_num(voiced_score, nan=0, posinf=0)
        return np.hstack([voiced_score.mean(), voiced_score.std(), voiced_score.max()])

    def get_spectral_rolloffs(self):
        """Spectral rolloffs for wheeze and crackle detection.

        As selected by regularization methods in Mendes et al. (2016), see above,
        and in:

        Mendes, L., I. M. Vogiatzis, E. Perantoni, E. Kaimakamis, I. Chouvarda,
            N. Maglaveras, V. Tsara, et al. 2015. “Detection of wheezes using their
            signature in the spectrogram space and musical features.” In 37th Annual
            International Conference of the IEEE Engineering in Medicine and
            Biology Society, 5581–5584.

        """

        roll75 = librosa.feature.spectral_rolloff(self.x, self.sr, roll_percent=0.75)
        roll95 = librosa.feature.spectral_rolloff(self.x, self.sr, roll_percent=0.95)
        return np.hstack([roll75.mean(), roll75.std(), roll95.mean(), roll95.std()])

    def get_spectral_moments(self):
        """Retrieve spectral mean, standard deviation, skew and kurtosis.

        In an effort to avoid redundancy with other spectral measurements, and
        with a nod to traditional MIR features in machine learning, these are
        calculated on the whole signal, i.e., on unframed data.

        Fujinaga, Ichiro. 1998. “Machine recognition of timbre using steady-state
            tone of acoustic musical instruments.” In Proceedings of the
            International Computer Music Conference, 4.

        """

        fft_ = np.fft.fft(self.x)
        hfft = len(fft_) // 2
        pos_fft_ = np.abs(fft_[:hfft])
        f = np.arange(hfft)

        centroid = np.dot(f, pos_fft_)
        var = np.dot(f ** 2, pos_fft_)
        skew = np.dot(f ** 3, pos_fft_)
        kurt = np.dot(f ** 4, pos_fft_)
        return np.hstack([centroid, var, skew, kurt])

    def get_entropy_features(self):
        """Three measurements based on spectral entropy.

        These measurements compare the span of spectral entropy in three ways,
        as a max/min ratio, as a max - min difference, and by taking the mean.

        Liu, Xi, Wee Ser, Jianmin Zhang, and Daniel Yam Thiam Goh. 2015. “Detection
            of adventitious lung sounds using entropy features and a 2-D threshold
            setting.” In Proceedings of the 10th International Conference on
            Information, Communications and Signal Processing, 1–5.

        """

        stft_ = librosa.stft(self.x)
        spec_entropy = self.get_framewise_entropy(stft_)
        ratio = np.max(spec_entropy) / (np.min(spec_entropy) + self.EPS)
        diff = np.max(spec_entropy) - np.min(spec_entropy)
        mean = np.mean(spec_entropy)
        return np.hstack([ratio, diff, mean])

    def get_lpc_tonality(self):
        """Ratio of fourth to zeroth LPC coefficient.

        Metric for wheeze detection based on the observation by Oletic et al.
        that linear predictive coding (LPC) estimation error tends to fall off
        more rapidly (i.e., in fewer coefficients) with tonal sounds than with
        non-tonal sounds. More tonal sounds will have a higher value of this ratio.

        **due to instability issues in librosa.lpc, this is currently not
        returned in self.get_features()**

        Oletic, Dinko, Bruno Arsenali, and Vedran Bilas. 2012. “Towards continuous
            wheeze detection body sensor node as a core of asthma monitoring system.”
            In Wireless Mobile Communication and Healthcare, 83:165–72.
            https://doi.org/10.1007/978-3-642-29734-2_23.

        """

        lpc_ = librosa.lpc(self.x, 4)
        return np.abs(lpc_[0] / lpc_[4])

    def get_features(self, row):
        self.filename = row["filename"]
        if "t_start" in row:
            self.t_start = row["t_start"]
            self.t_stop = row["t_stop"]

        try:
            self.set_x()
            features = np.hstack((self.get_mfccs(),
                                  self.get_chromagram(),
                                  self.get_voiced_score(),
                                  self.get_spectral_rolloffs(),
                                  self.get_spectral_moments(),
                                  self.get_entropy_features(),
                                  ))
        except RuntimeError:
            print('Bad file.')
            features = None

        return pd.Series(features)

    def extract(self):
        # print('Initializing YAMNet...')
        # self.initialize_yamnet()

        if self.do_gini_filter:
            print('Gini NMF filter enabled.')
        if self.resample_freq is not None:
            print('BP filtering and resampling at %d Hz.' % self.resample_freq)

        print('Extracting features...')
        for _, row in tqdm(self.filenames_labels.iterrows(),
                           total=self.filenames_labels.shape[0]):
            self.feature_data.append(self.get_features(row))

        self.feature_data = pd.DataFrame.from_records(self.feature_data)
        self.feature_data = self.standardize(self.feature_data)
        self.feature_data = self.feature_data.join(self.filenames_labels.iloc[:, 4:])

        if self.fill_na:
            self.feature_data.fillna(0, inplace=True)
            print('Replacing feature NaNs with zeros...')

        print('Done.')