################################################################################################################
#   This script perform the dataset preparation and primitives segmentation of DanceMVP model - AAAI2024
#   for the pre-trained phase that only involves dance motion and music
#
#   We segment each 10s dance motion-music into 3 pieces and generate pairs for contrastive learning
#
#   Dance data - dance motion recorded by motion capatured deveices (samples,coordinations,frames,joints)
#   Dance music - MFCC, MFCC-delta, constant-Q chromagram, tempogram and onset strength features of music
#                 (samples,features_dim,130)
#################################################################################################################

import copy
from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
import numpy as np
import random
import torch
import pickle
from data_aug.process_music import MusicFeatures
from os.path import exists
import soundfile as sf
from pysndfx import AudioEffectsChain
import os
import random
from pydub import AudioSegment
import librosa
from scipy import signal
from peakdetect import peakdetect
from scipy import signal
import pickle
from data_aug.process_music import MusicFeatures

#meta parameters for music feature extraction
N_MFCCs = 40
HOP_LENGTH = 512
SR = 22050
NUM_GENRES = 3
MINI_BATCH = False
PADDING = False
SEQ_LEN = 10
MEAN = 0
VAR = 1
TRANS = 10
MUSIC_FORMAT = '.wav'
LEVELS = 3

DROP_S = 1.0
FPS = 100
SR_LIB = 22050
FAST_RATE = 1.2

N_MFCCs = 40
HOP_LENGTH = 512
SR = 22050
SPLIT_RATIO = 5
N = 75

SEG_PATH = './datasets/dance_level/music/seg_music/'

def cal_hitrate(mo_arry,mu_y,mu_sr):
    result = np.zeros((mo_arry.shape[0]))
    for k in range(mo_arry.shape[0]):
        motions = mo_arry[k,:,:,:]

        motions = motions[:,[1,4,6,7,10,14,15],:]
        velocity = np.zeros((motions.shape[0],motions.shape[1],motions.shape[2]))
        for b in range(motions.shape[0]-1):
            velocity[b,:,0] = motions[b+1,:,0] - motions[b,:,0]
            velocity[b, :,1] = motions[b + 1, :, 1] - motions[b, :, 1]
            velocity[b, :,2] = motions[b + 1, :, 2] - motions[b, :, 2]

        velo = np.zeros((motions.shape[0]))
        for k1 in range(motions.shape[0]):
            for k2 in range(motions.shape[1]):
                for k3 in range(motions.shape[2]):
                    velo[k1] += velocity[k1,k2,k3]

        dataa = velo[:1000]#[150:400]
        b, a = signal.butter(1, 0.05)
        zi = signal.lfilter_zi(b, a)
        z, _ = signal.lfilter(b, a, dataa, zi=zi * dataa[0])
        z2, _ = signal.lfilter(b, a, z, zi=zi * z[0])
        data = signal.filtfilt(b, a, dataa)

        peaks = peakdetect(data, lookahead=25)
        higherPeaks = np.array(peaks[0])
        lowerPeaks = np.array(peaks[1])
        p = np.concatenate((higherPeaks,lowerPeaks))

        o_env = librosa.onset.onset_strength(y=mu_y, sr=mu_sr)

        hop_length = 512
        tempo, beats = librosa.beat.beat_track(onset_envelope=o_env, sr=mu_sr)


        times = librosa.times_like(o_env, sr=mu_sr, hop_length=hop_length)
        a = times[beats] * 100
        a = np.round(a)


        hit = 0
        if higherPeaks.size != 0:
            peaks_id =p[:,0]#peaksss#higherPeaks[:,0]

            for k4 in range(a.shape[0]):
                hit+=1 in (abs(a[k4]-ele) <=20 for ele in list(peaks_id))
        else:
            hit = 0

        result[k] = hit / a.shape[0] * 100

    hit_rate = np.mean(result)
    return hit_rate


def spatial_aug_nosie(array):
    ori_array = copy.deepcopy(array)
    noisy_array = copy.deepcopy(array)
    for i in range(ori_array.shape[0]):
        row, col, ch = ori_array[i].shape
        MEAN = 0
        VAR = 0.5
        sigma = VAR ** 0.5
        gauss = np.random.normal(MEAN, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy_array[i,:,:,:] = ori_array[i,:,:,:]  + gauss

    return noisy_array

def spatial_aug_trans(array):
    trans_array = copy.deepcopy(array)
    trans_array = trans_array + TRANS

    return trans_array

def temporal_aug_playspeed(motion,music,gap):
    speed_array = copy.deepcopy(motion)
    playback_array = np.zeros((motion.shape[0],motion.shape[1],motion.shape[2],motion.shape[3]))

    y, sr = librosa.load(music, duration=int(SEQ_LEN-gap/100))
    music_array = np.zeros((speed_array.shape[0], y.shape[0]))
    for i in range(speed_array.shape[0]):
        if PADDING == True:
            #if SEQ_LEN == 10:
            split = random.sample(range(6, 12),1)[0]

        else:
            if SEQ_LEN == 10:
                split = random.sample(range(4, 8), 1)[0]
            else:
                split = random.sample(range(2, 4), 1)[0]
        split_id = int(FPS*split)
        front_arr = speed_array[i,:split_id,:,:]
        back_arr = speed_array[i,split_id:,:,:]
        f = signal.resample(front_arr, int(front_arr.shape[0]*FAST_RATE))
        re = int(speed_array.shape[1]-f.shape[0])
        if back_arr.shape[0] >= re:
            ratio = back_arr.shape[0]/re
        else:
            ratio = re/back_arr.shape[0]
        b = signal.resample(back_arr, int(back_arr.shape[0]/ratio))
        aug_arr = np.vstack((f,b))
        if aug_arr.shape[0] <  playback_array.shape[1]:
            gaps = int(playback_array.shape[1] - aug_arr.shape[0])
            gap_arr = aug_arr[:gaps,:,:]
            pad_arr =  np.append(aug_arr,gap_arr)
            pad_arr = np.reshape(pad_arr,(playback_array.shape[1],aug_arr.shape[1],aug_arr.shape[2]))
            playback_array[i,:,:,:] = pad_arr
        else:
            playback_array[i, :, :, :] = aug_arr

        rat = int(y.shape[0]*(split/SEQ_LEN))
        tio = int(y.shape[0]*(1-split/SEQ_LEN))
        music_f = y[:rat]
        music_b = y[:tio]
        fast_rate = front_arr.shape[0]/f.shape[0]
        music_slow = librosa.effects.time_stretch(music_f, rate=fast_rate)
        music_fast = librosa.effects.time_stretch(music_b, rate=ratio)
        aug_music = np.concatenate((music_slow,music_fast))
        if PADDING == True:
            assert aug_music.shape[0] == 374850 or aug_music.shape[0] == 374849
        music_array[i, :,] = aug_music


    return playback_array,music_array

def temporal_aug_dropFrames(motions,music,gap):

    motion_array = copy.deepcopy(motions)
    y, sr = librosa.load(music,duration=int(SEQ_LEN-gap/100))
    music_array = np.zeros((motion_array.shape[0],y.shape[0]))
    newAudio = AudioSegment.from_wav(music)[:int((SEQ_LEN-gap/100)*1000)]
    gapss = int(SEQ_LEN - newAudio.duration_seconds)*1000
    music_padding = newAudio[:gapss]
    newAudios = newAudio + music_padding
    for i in range(motion_array.shape[0]):
        num_drop = int(DROP_S*FPS)
        start_id = random.sample(range(0, int(SEQ_LEN*100-gap-num_drop)), 1)[0]
        for j in range(num_drop):
            motion_array[i,start_id+j,:,:] = motion_array[i,start_id,:,:]
        music_start_id = int(start_id*10)
        music_end_id = int((start_id+num_drop)*10)
        music1 = newAudios[:music_start_id]
        music2 = newAudios[music_start_id:music_start_id+10]*num_drop
        music3 = newAudios[music_end_id:]
        music_arr = music1 + music2 + music3
        m_name = 'temp.wav'
        music_arr.export(m_name, format="wav")
        music_array[i,:],sr = librosa.load(m_name)
    return motion_array,music_array

def check_music(ch,path):
    music_path = path + '/music/'+ ch + MUSIC_FORMAT
    print(music_path)
    file_exists = exists(music_path)
    assert file_exists == True
    return music_path


def seg_music(path,name):
    audio = AudioSegment.from_file(path, "wav")

    step = 3

    cut_parameters = [1,4,7]

    for t in cut_parameters:
        start_time = int(t *1000)

        stop_time = int((t+step)*1000)
        audio_chunk = audio[start_time:stop_time]
        save_path = SEG_PATH + name + '/'
        audio_chunk.export(save_path+name+"_segment_{}.wav".format(int(t / step)), format="wav")
    return

def seg_motion(motion_arry):
    seg_arr = np.zeros((motion_arry.shape[0], 3, 300, motion_arry.shape[2],motion_arry.shape[3]))
    for i in range(seg_arr.shape[0]):

        seg_arr[i, 0, :, :] = motion_arry[i, 100:400, :, :]
        seg_arr[i, 1, :, :] = motion_arry[i, 400:700, :, :]
        seg_arr[i, 2, :, :] = motion_arry[i, 700:1000, :, :]
    seg_arr = np.reshape(seg_arr,(int(seg_arr.shape[0]*seg_arr.shape[1]),seg_arr.shape[2],seg_arr.shape[3],seg_arr.shape[4]))
    return seg_arr

def seg_motion_test(motion_arry):
    seg_arr = np.zeros((motion_arry.shape[0], 3, 300, motion_arry.shape[2],motion_arry.shape[3],motion_arry.shape[4]))
    seg_label = np.zeros((motion_arry.shape[0],3,300,21,3,1))
    for i in range(seg_arr.shape[0]):
        seg_arr[i,0,:,:,:,:] = motion_arry[i,100:400,:,:,:]
        seg_label[i,0,:,:,:,:] = int(0)
        seg_arr[i,1,:,:,:,:] = motion_arry[i,400:700,:,:,:]
        seg_label[i,1,:,:,:,:] = int(1)
        seg_arr[i,2,:,:,:,:] = motion_arry[i,700:1000,:,:,:]
        seg_label[i,2,:,:,:,:] = int(2)
    seg_label = np.reshape(seg_label,(int(seg_label.shape[0]*seg_label.shape[1]),300,21,3,1))
    seg_arr = np.reshape(seg_arr,(int(seg_arr.shape[0]*seg_arr.shape[1]),300,21,3,1))
    return seg_arr

class ContrastiveLearningLevelDatasetSEG:
    def __init__(self, root_folder, samples,split=True,pair=True,train=True):
        self.root_folder = root_folder #meiyou yong
        self.split = split
        self.k = samples
        self.train = train
        self.create_pair = pair

    def load_level_dataset(self):
        #choe = ['b_ch0', 'b_ch1', 'b_ch2', 'k_ch0', 'k_ch1', 'k_ch2', 'k_ch3', 'k_ch4', 'u_ch0']
        # choe = ['b_ch0', 'b_ch1', 'b_ch2', 'b_ch3','b_ch4',
        #         'k_ch0', 'k_ch1', 'k_ch2', 'k_ch3', 'k_ch4',
        #         'u_ch0','u_ch1', 'u_ch3', 'h_ch0', 'h_ch1',
        #          'j_ch1', 'j_ch2', 'j_ch3']
        choe = ['b_ch0', 'b_ch2', 'b_ch4',
                'k_ch0',  'k_ch3', 'k_ch4',
                'u_ch1', 'u_ch3', 'h_ch1',
                'j_ch1']
        #choe = ['k_ch1','u_ch2','b_ch1','h_ch0','j_ch1']#,'k_ch3', 'k_ch4', 'b_ch2','b_ch4']#,
        #        'u_ch1', 'u_ch3', 'h_ch0', 'j_ch1', 'j_ch3']
        #choe = ['b_ch0', 'b_ch1', 'b_ch2', 'k_ch0', 'k_ch1', 'k_ch2', 'k_ch3', 'k_ch4', 'u_ch0']
        #choe = ['b_ch4', 'k_ch4', 'u_ch3','h_ch0','j_ch3']
        # choe = ['b_ch4', 'k_ch0', 'u_ch3','h_ch0','j_ch3']
        # choe = ['b_ch2', 'b_ch3','b_ch4', 'k_ch0', 'k_ch1', 'k_ch2','u_ch3', 'h_ch0', 'j_ch3']
        label_count = 0
        motion_ls = []
        music_ls = []
        motion_label = []
        hit_rates = []

        for k in range(len(choe)):
            dataset_path = './datasets/dance_level/' + choe[k] + '/'
            path = dataset_path.split('/')[:-2]
            path = '/'.join(path)
            ch_id = dataset_path.split('/')[-2]  # 'u_ch0'
            music_path = check_music(ch_id, path)
            # sound = temporal_aug_playspeed(music_path)
            level_dataset = {}
            data_ls = os.listdir(dataset_path)
            count = 0
            aug_motion = []
            aug_music = []
            for j in range(len(data_ls)):
                name = data_ls[j]
                if name.find(ch_id) != -1:
                    if name.find('i_' + ch_id) != -1:
                        id = 1
                    elif name.find('e_' + ch_id) != -1:
                        id = 0
                    elif name.find('b_' + ch_id) != -1:
                        id = 2
                    else:
                        print('check the choe name')
                        break
                    level_dataset[str(count)] = {}
                    assert count < 3
                    data_path = dataset_path + name
                    arr = np.load(data_path)
                    arr = np.transpose(arr, (0, 2, 3, 1))
                    arr = arr[:, :SEQ_LEN * 100, :, :]
                    if self.train == True:
                        arr = arr[:self.k,:,:,:]
                        #arr = arr[75:90 :, :, :]
                    else:

                        arr = arr[90:100, :, :, :]
                    y, sr = librosa.load(music_path, duration=int(arr.shape[1] / 100))
                    ##### cal hit rate for 10s mo and mu
                    hitR = cal_hitrate(arr,y,sr)

                    if self.create_pair == False:
                        arr = arr[:, :, :, :, np.newaxis]
                        arr = seg_motion_test(arr)
                        arr = np.transpose(arr, (0, 3, 1, 2, 4))
                        music_arr = np.zeros((int(arr.shape[0]/3), y.shape[0]))
                        for i in range(int(arr.shape[0]/3)):
                            music_arr[i,:] = y
                    else:
                        music_arr = np.zeros((arr.shape[0], y.shape[0]))
                        for i in range(arr.shape[0]):
                            music_arr[i, :] = y
                    if self.create_pair == False:
                        if SEQ_LEN == 10:
                            music_param = 130#173#431
                        else:  # 5
                            music_param = 216
                        seg_music(music_path, choe[k])
                        seg_list = os.listdir(SEG_PATH + choe[k] + '/')
                        seg_list.sort()
                        for i in range(music_arr.shape[0]):
                            for n in range(len(seg_list)):
                                y, sr = librosa.load(SEG_PATH + choe[k] + '/' + seg_list[n])
                                music_data = np.zeros(
                                    (music_arr.shape[0], len(seg_list), int(40 + 40 + 84 + 384 + 1), music_param))
                                music = MusicFeatures(y, SR, N_MFCCs, HOP_LENGTH, figures=False)
                                # music = MusicFeatures(music_arr[i], SR, N_MFCCs, HOP_LENGTH, figures=False)
                                music_data[i, n, :, :] = music.generate_features()
                        music_data = np.reshape(music_data, (int(music_data.shape[0] * music_data.shape[1]), music_data.shape[2], music_data.shape[3]))
                        #print(self.split)
                        if self.split == True:
                            if id == 0:
                                yyy = [0, 1, 2]
                            elif id ==1:
                                yyy = [3,4,5]
                            elif id ==2:
                                yyy = [6,7,8]
                            else:
                                print('wrong label')
                            yy = [x + int(label_count * 9) for x in yyy]
                            for a in range(int(arr.shape[0] / 3)):  # 15
                                motion_label = motion_label + yy
                                for a1 in range(3):
                                    hit_rates.append(hitR)
                            if self.train == True:
                                if self.k != N:
                                    motion_ls.append(arr)
                                    music_ls.append(music_data)
                                else:
                                    motion_ls.append(arr[:self.k,:,:,:])
                                    music_ls.append(music_data[:self.k, :, :])
                            else:
                                if self.k != N:
                                    motion_ls.append(arr)
                                    music_ls.append(music_data)
                                else:
                                    n = int(100-self.k)
                                    motion_ls.append(arr)
                                    music_ls.append(music_data)

                    else:
                        aug_id = random.sample(range(0, 4), 2)
                        aug_data = []

                        for i in range(int(arr.shape[0])):
                            music_arr[i, :] = y
                        gaps=0
                        if SEQ_LEN == 10:
                            music_param = 130#431
                        else: #5
                            music_param =216
                        print(aug_id)
                        if 0 in aug_id:
                            seg_music(music_path, choe[k])
                            seg_list = os.listdir(SEG_PATH + choe[k] + '/')
                            seg_list.sort()
                            for i in range(music_arr.shape[0]):
                                for n in range(len(seg_list)):
                                    y, sr = librosa.load(SEG_PATH + choe[k] + '/' + seg_list[n])
                                    muf = np.zeros(
                                        (music_arr.shape[0], len(seg_list), int(40 + 40 + 84 + 384 + 1), music_param))
                                    music = MusicFeatures(y, SR, N_MFCCs, HOP_LENGTH, figures=False)
                                    muf[i, n, :, :] = music.generate_features()
                            muf = np.reshape(muf, (int(muf.shape[0] * muf.shape[1]), muf.shape[2], muf.shape[3]))
                            aug_data.append(muf)
                            sarr = seg_motion(arr)
                            arrr =spatial_aug_nosie(sarr)
                            arrr = np.transpose(arrr, (0, 3, 1, 2)) #n,3,1000,21
                            arrr = arrr[:, :, :, :, np.newaxis]
                            aug_data.append(arrr)
                        if 1 in aug_id:
                            seg_music(music_path,choe[k])
                            seg_list = os.listdir(SEG_PATH+choe[k]+'/')
                            seg_list.sort()
                            for i in range(music_arr.shape[0]):
                                for n in range(len(seg_list)):
                                    y, sr = librosa.load(SEG_PATH+choe[k]+'/' + seg_list[n])
                                    muf = np.zeros((music_arr.shape[0], len(seg_list),int(40 + 40 + 84 + 384 + 1), music_param))
                                    music = MusicFeatures(y, SR, N_MFCCs, HOP_LENGTH, figures=False)
                                    muf[i, n, :, :] = music.generate_features()
                            muf = np.reshape(muf, (int(muf.shape[0] * muf.shape[1]), muf.shape[2], muf.shape[3]))
                            aug_data.append(muf)
                            sarr = seg_motion(arr)
                            arrr =spatial_aug_trans(sarr)
                            arrr = np.transpose(arrr, (0, 3, 1, 2))
                            arrr = arrr[:, :, :, :, np.newaxis]
                            aug_data.append(arrr)
                        if 2 in aug_id:
                            mo, mu = temporal_aug_playspeed(arr, music_path, gaps)
                            mo = seg_motion(mo)
                            seg_music(music_path, choe[k])
                            seg_list = os.listdir(SEG_PATH + choe[k] + '/')
                            seg_list.sort()
                            for i in range(mu.shape[0]):
                                for n in range(len(seg_list)):
                                    y, sr = librosa.load(SEG_PATH + choe[k] + '/' + seg_list[n])
                                    muf = np.zeros(
                                        (music_arr.shape[0], len(seg_list), int(40 + 40 + 84 + 384 + 1), music_param))
                                    music = MusicFeatures(y, SR, N_MFCCs, HOP_LENGTH, figures=False)
                                    muf[i, n, :, :] = music.generate_features()
                            muf = np.reshape(muf, (int(muf.shape[0] * muf.shape[1]), muf.shape[2], muf.shape[3]))
                            aug_data.append(muf)
                            mo = np.transpose(mo, (0, 3, 1, 2))
                            mo = mo[:, :, :, :, np.newaxis]
                            aug_data.append(mo)
                        if 3 in aug_id:
                            mo, mu = temporal_aug_dropFrames(arr, music_path, gaps)
                            mo = seg_motion(mo)
                            seg_music(music_path, choe[k])
                            seg_list = os.listdir(SEG_PATH + choe[k] + '/')
                            seg_list.sort()
                            for i in range(mu.shape[0]):
                                for n in range(len(seg_list)):
                                    y, sr = librosa.load(SEG_PATH + choe[k] + '/' + seg_list[n])
                                    muf = np.zeros(
                                        (music_arr.shape[0], len(seg_list), int(40 + 40 + 84 + 384 + 1), music_param))
                                    music = MusicFeatures(y, SR, N_MFCCs, HOP_LENGTH, figures=False)
                                    muf[i, n, :, :] = music.generate_features()
                            muf = np.reshape(muf, (int(muf.shape[0] * muf.shape[1]), muf.shape[2], muf.shape[3]))
                            aug_data.append(muf)
                            mo = np.transpose(mo, (0, 3, 1, 2))
                            mo = mo[:, :, :, :, np.newaxis]
                            aug_data.append(mo)

                        aug_pair_mo = np.stack((aug_data[1], aug_data[3]), axis=1)
                        aug_pair_mu = np.stack((aug_data[0], aug_data[2]), axis=1)

                        if self.split == True:
                            if id == 0:
                                yyy = [0, 1, 2]
                            elif id ==1:
                                yyy = [3,4,5]
                            elif id ==2:
                                yyy = [6,7,8]
                            else:
                                print('wrong label')
                            yy = [x + int(label_count * 9) for x in yyy]
                            for a in range(int(arr.shape[0])):  # 15
                                motion_label = motion_label + yy
                                for a1 in range(3):
                                    hit_rates.append(hitR)
                            if self.train == True:
                                motion_ls.append(aug_pair_mo)
                                music_ls.append(aug_pair_mu)

                            else:
                                motion_ls.append(aug_pair_mo)
                                music_ls.append(aug_pair_mu)
                                break

                    count+=1
            label_count += 1


        self.motion_data = np.concatenate(motion_ls, axis=0)
        self.music_data = np.concatenate(music_ls, axis=0)

        print(self.motion_data.shape)
        print(self.music_data.shape)
        import itertools
        self.motion_labels = motion_label
        self.music_labels = copy.deepcopy(self.motion_labels)
        self.hitR = hit_rates


        return

    def __len__(self):
        return self.motion_data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.create_pair == True:
            motion = [self.motion_data[idx, 0, :, :, :, :], self.motion_data[idx, 1, :, :, :, :]]
            music = [self.music_data[idx, 0, :, :], self.music_data[idx, 1, :, :]]
        else:
            motion = self.motion_data[idx,:, :, :, :]
            music = self.music_data[idx,:, :]

        motion_label = self.motion_labels[idx]
        music_label = self.music_labels[idx]
        hit_r = self.hitR[idx]

        return motion, motion_label, music, music_label,hit_r


