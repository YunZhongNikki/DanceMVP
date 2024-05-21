import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank

N_MFCCs = 40
HOP_LENGTH = 512

class MusicFeatures(object):

    def __init__(self, y, sr, n_mfcc, hop_length, figures=False):
        # load music
        #'self.music_path = music_path
        self.y, self.sr = y,sr
        self.num_mfcc = n_mfcc
        self.hop_length = hop_length
        self.figures = figures
        #self.music_feature = [self.mfcc, self.mfcc_delta, self.q, self.tempogram, self.os]

    def compute_mfcc(self):
        mfccs = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=self.num_mfcc)
        mfcc_delta = librosa.feature.delta(mfccs)

        # if self.figures == True:
        #     plt.figure(figsize=(10, 4))
        #     librosa.display.specshow(mfccs, x_axis="time",cmap='BuPu')
        #     plt.colorbar()
        #     plt.title('MFCC')
        #     plt.tight_layout()
        #     plt.show()
        #     print('done')
        # else:
        #     pass
        return mfccs,mfcc_delta

    def compute_constantQ(self):
        C = np.abs(librosa.cqt(self.y, sr=self.sr))
        # if self.figures == True:
        #     fig, ax = plt.subplots()
        #     img = librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
        #                                    sr=self.sr, x_axis='time', y_axis='cqt_note', ax=ax,cmap='Purples')
        #     ax.set_title('Constant-Q power spectrum')
        #     fig.colorbar(img, ax=ax, format="%+2.0f dB")
        #     plt.show()
        #     print('done')
        return C

    def compute_tempogram(self):
        oenv = librosa.onset.onset_strength(y=self.y, sr=self.sr, hop_length=self.hop_length)
        tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=self.sr,
                                              hop_length=self.hop_length)
        # Compute global onset autocorrelation
        ac_global = librosa.autocorrelate(oenv, max_size=tempogram.shape[0])
        ac_global = librosa.util.normalize(ac_global)
        # Estimate the global tempo for display purposes
        tempo = librosa.beat.tempo(onset_envelope=oenv, sr=self.sr,
                                 hop_length=self.hop_length)[0]
        if self.figures == True:
            fig, ax = plt.subplots(nrows=4, figsize=(10, 10))
            times = librosa.times_like(oenv, sr=self.sr, hop_length=self.hop_length)
            ax[0].plot(times, oenv, label='Onset strength')
            ax[0].label_outer()
            ax[0].legend(frameon=True)

            librosa.display.specshow(tempogram, sr=self.sr, hop_length=self.hop_length,
                                     x_axis='time', y_axis='tempo', cmap='Purples',
                                     ax=ax[1])

            ax[1].axhline(tempo, color='w', linestyle='--', alpha=1,
                          label='Estimated tempo={:g}'.format(tempo))

            ax[1].legend(loc='upper right')
            ax[1].set(title='Tempogram')

            x = np.linspace(0, tempogram.shape[1] * float(self.hop_length) / self.sr,
                            num=tempogram.shape[1])
            ax[2].plot(x, np.mean(tempogram, axis=0), label='Mean local autocorrelation')
            ax[2].plot(x, ac_global, '--', alpha=0.75, label='Global autocorrelation')
            ax[2].set(xlabel='Lag (seconds)')
            ax[2].legend(frameon=True)

            freqs = librosa.tempo_frequencies(tempogram.shape[1], hop_length=self.hop_length, sr=self.sr)
            y= np.mean(tempogram[1:],axis=0)
            ax[3].semilogx(freqs[:], np.mean(tempogram[1:], axis=0),
                           label='Mean local autocorrelation', base=2)

            ax[3].semilogx(freqs[1:], ac_global[1:], '--', alpha=0.75,
                           label='Global autocorrelation', base=2)

            ax[3].axvline(tempo, color='black', linestyle='--', alpha=.8,
                          label='Estimated tempo={:g}'.format(tempo))

            ax[3].legend(frameon=True)
            ax[3].set(xlabel='BPM')
            ax[3].grid(True)
            plt.show()
            print('done')
        return tempogram

    def compute_onsetStrength(self):
        self.onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        return self.onset_env

    def generate_features(self):
        mfcc, mfcc_delta = self.compute_mfcc()
        self.music_features = np.append(mfcc, mfcc_delta, axis=0)
        ### constant Q
        self.music_features = np.append(self.music_features, self.compute_constantQ(), axis=0)
        ### tempogram
        self.music_features = np.append(self.music_features, self.compute_tempogram(), axis=0)
        ### onset_strength
        self.music_features = np.append(self.music_features, self.compute_onsetStrength()[np.newaxis,:], axis=0)
        #mfcc,mfcc_delta,q,tempogram,os
        #if self.music_features.shape[1] != 431:
            # self.music_features = np.concatenate((self.music_features,self.music_features),axis=1)
            # self.music_features = np.delete(self.music_features, -1, 1)
            #print('done')
        if self.music_features.shape[1] != 130:
            self.music_features = self.music_features[:,:130]
        assert self.music_features.shape[1] == 130
        return self.music_features
# if __name__ == "__main__":
#     # Load a wav file
#     path = '../datasets/dance_level/music/'
#     name = 'k_ch0'
#     format = '.wav'
#     y, sr = librosa.load(path+name+format)
#     print('done')

    ### visualize music
    # extract mel spectrogram feature
    # melspec = librosa.feature.melspectrogram(y, sr, n_fft=1024, hop_length=512, n_mels=128)
    # # convert to log scale
    # logmelspec = librosa.power_to_db(melspec)
    # plt.figure()
    # # plot a wavform
    # plt.subplot(2, 1, 1)
    # librosa.display.waveshow(y, sr)
    # plt.title('Beat wavform')
    # # plot mel spectrogram
    # plt.subplot(2, 1, 2)
    # librosa.display.specshow(logmelspec, sr=sr, x_axis='time', y_axis='mel')
    # plt.title('Mel spectrogram')
    # plt.tight_layout()
    # plt.show()


    ### mfcc
    # y, sr = librosa.load(path+name+format)
    # mfcc,mfcc_delta = compute_mfcc(y,sr,N_MFCCs,False)
    # ### constant Q
    # q = compute_constantQ(y,sr,False)
    # ### tempogram
    # tempogram = compute_tempogram(y,sr,HOP_LENGTH,True)
    # ### onset_strength
    # os = compute_onsetStrength(y,sr)