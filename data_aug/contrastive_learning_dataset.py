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


N_MFCCs = 40
HOP_LENGTH = 512
SR = 22050
NUM_GENRES = 18
class ContrastiveLearningLevelDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    def load_level_dataset(self):
        # motion_data = 'pair_dance_motion_aug.npy'
        # music_data = 'pair_dance_music_features_aug.npy'
        motion_data_k_ch0 = 'pair_dance_motion_aug_k_ch0.npy'
        music_data_k_ch0 = 'pair_dance_music_features_aug_k_ch0.npy'
        motion_data_k_ch1 = 'pair_dance_motion_aug_k_ch1.npy'
        music_data_k_ch1 = 'pair_dance_music_features_aug_k_ch1.npy'
        motion_data_k_ch2 = 'pair_dance_motion_aug_k_ch2.npy'
        music_data_k_ch2 = 'pair_dance_music_features_aug_k_ch2.npy'
        motion_data_k_ch3 = 'pair_dance_motion_aug_k_ch3.npy'
        music_data_k_ch3 = 'pair_dance_music_features_aug_k_ch3.npy'
        motion_data_k_ch4 = 'pair_dance_motion_aug_k_ch4.npy'
        music_data_k_ch4 = 'pair_dance_music_features_aug_k_ch4.npy'

        motion_data_b_ch0 = 'pair_dance_motion_aug_b_ch0.npy'
        music_data_b_ch0 = 'pair_dance_music_features_aug_b_ch0.npy'
        motion_data_b_ch1 = 'pair_dance_motion_aug_b_ch1.npy'
        music_data_b_ch1 = 'pair_dance_music_features_aug_b_ch1.npy'
        motion_data_b_ch2 = 'pair_dance_motion_aug_b_ch2.npy'
        music_data_b_ch2 = 'pair_dance_music_features_aug_b_ch2.npy'
        motion_data_b_ch3 = 'pair_dance_motion_aug_b_ch3.npy'
        music_data_b_ch3 = 'pair_dance_music_features_aug_b_ch3.npy'
        motion_data_b_ch4 = 'pair_dance_motion_aug_b_ch4.npy'
        music_data_b_ch4 = 'pair_dance_music_features_aug_b_ch4.npy'

        motion_data_u_ch0 = 'pair_dance_motion_aug_u_ch0.npy'
        music_data_u_ch0 = 'pair_dance_music_features_aug_u_ch0.npy'
        motion_data_u_ch1 = 'pair_dance_motion_aug_u_ch1.npy'
        music_data_u_ch1 = 'pair_dance_music_features_aug_u_ch1.npy'
        motion_data_u_ch3 = 'pair_dance_motion_aug_u_ch3.npy'
        music_data_u_ch3 = 'pair_dance_music_features_aug_u_ch3.npy'

        motion_data_j_ch1 = 'pair_dance_motion_aug_j_ch1.npy'
        music_data_j_ch1 = 'pair_dance_music_features_aug_j_ch1.npy'
        motion_data_j_ch2 = 'pair_dance_motion_aug_j_ch2.npy'
        music_data_j_ch2 = 'pair_dance_music_features_aug_j_ch2.npy'
        motion_data_j_ch3 = 'pair_dance_motion_aug_j_ch3.npy'
        music_data_j_ch3 = 'pair_dance_music_features_aug_j_ch3.npy'

        motion_data_h_ch0 = 'pair_dance_motion_aug_h_ch0.npy'
        music_data_h_ch0 = 'pair_dance_music_features_aug_h_ch0.npy'
        motion_data_h_ch1 = 'pair_dance_motion_aug_h_ch1.npy'
        music_data_h_ch1 = 'pair_dance_music_features_aug_h_ch1.npy'

        # train_motion_data = np.load(self.root_folder + motion_data)
        # train_music_data = np.load(self.root_folder + music_data)
        if NUM_GENRES == 18:
            train_motion_data_k_ch0 = np.load(self.root_folder + motion_data_k_ch0)
            train_music_data_k_ch0  = np.load(self.root_folder + music_data_k_ch0)
            train_motion_data_k_ch1 = np.load(self.root_folder + motion_data_k_ch1)
            train_music_data_k_ch1 = np.load(self.root_folder + music_data_k_ch1)
            train_motion_data_k_ch2 = np.load(self.root_folder + motion_data_k_ch2)
            train_music_data_k_ch2 = np.load(self.root_folder + music_data_k_ch2)
            train_motion_data_k_ch3 = np.load(self.root_folder + motion_data_k_ch3)
            train_music_data_k_ch3 = np.load(self.root_folder + music_data_k_ch3)
            train_motion_data_k_ch4 = np.load(self.root_folder + motion_data_k_ch4)
            train_music_data_k_ch4 = np.load(self.root_folder + music_data_k_ch4)

            train_motion_data_b_ch0 = np.load(self.root_folder + motion_data_b_ch0)
            train_music_data_b_ch0 = np.load(self.root_folder + music_data_b_ch0)
            train_motion_data_b_ch1 = np.load(self.root_folder + motion_data_b_ch1)
            train_music_data_b_ch1 = np.load(self.root_folder + music_data_b_ch1)
            train_motion_data_b_ch2 = np.load(self.root_folder + motion_data_b_ch2)
            train_music_data_b_ch2 = np.load(self.root_folder + music_data_b_ch2)
            train_motion_data_b_ch3 = np.load(self.root_folder + motion_data_b_ch3)
            train_music_data_b_ch3 = np.load(self.root_folder + music_data_b_ch3)
            train_motion_data_b_ch4 = np.load(self.root_folder + motion_data_b_ch4)
            train_music_data_b_ch4 = np.load(self.root_folder + music_data_b_ch4)

            train_motion_data_j_ch1 = np.load(self.root_folder + motion_data_j_ch1)
            train_music_data_j_ch1 = np.load(self.root_folder + music_data_j_ch1)
            train_motion_data_j_ch2 = np.load(self.root_folder + motion_data_j_ch2)
            train_music_data_j_ch2 = np.load(self.root_folder + music_data_j_ch2)
            train_motion_data_j_ch3 = np.load(self.root_folder + motion_data_j_ch3)
            train_music_data_j_ch3 = np.load(self.root_folder + music_data_j_ch3)

            train_motion_data_u_ch0 = np.load(self.root_folder + motion_data_u_ch0)
            train_music_data_u_ch0 =np.load(self.root_folder + music_data_u_ch0)
            train_motion_data_u_ch1 = np.load(self.root_folder + motion_data_u_ch1)
            train_music_data_u_ch1 = np.load(self.root_folder + music_data_u_ch1)
            train_motion_data_u_ch3 = np.load(self.root_folder + motion_data_u_ch3)
            train_music_data_u_ch3 = np.load(self.root_folder + music_data_u_ch3)

            train_motion_data_h_ch0 = np.load(self.root_folder + motion_data_h_ch0)
            train_music_data_h_ch0 = np.load(self.root_folder + music_data_h_ch0)
            train_motion_data_h_ch1 = np.load(self.root_folder + motion_data_h_ch1)
            train_music_data_h_ch1 = np.load(self.root_folder + music_data_h_ch1)


            self.motion_data = np.concatenate((train_motion_data_k_ch0,train_motion_data_k_ch1,train_motion_data_k_ch2,
                                                train_motion_data_k_ch3,train_motion_data_k_ch4,train_motion_data_b_ch0,
                                                train_motion_data_b_ch1,train_motion_data_b_ch2,train_motion_data_b_ch3,
                                                train_motion_data_b_ch4,train_motion_data_j_ch1,train_motion_data_j_ch2,
                                               train_motion_data_j_ch3,train_motion_data_u_ch0,train_motion_data_u_ch1,
                                               train_motion_data_u_ch3,train_motion_data_h_ch0,train_motion_data_h_ch1))
            self.music_data = np.concatenate((train_music_data_k_ch0,train_music_data_k_ch1,train_music_data_k_ch2,
                                                train_music_data_k_ch3,train_music_data_k_ch4,train_music_data_b_ch0,
                                                train_music_data_b_ch1,train_music_data_b_ch2,train_music_data_b_ch3,
                                                train_music_data_b_ch4,train_music_data_j_ch1,train_music_data_j_ch2,
                                                train_music_data_j_ch3,train_music_data_u_ch0,train_music_data_u_ch1,
                                                train_music_data_u_ch3,train_music_data_h_ch0,train_music_data_h_ch1))
        else:

            train_motion_data_k_ch1 = np.load(self.root_folder + motion_data_k_ch1)
            train_music_data_k_ch1 = np.load(self.root_folder + music_data_k_ch1)
            train_motion_data_b_ch2 = np.load(self.root_folder + motion_data_b_ch2)
            train_music_data_b_ch2 = np.load(self.root_folder + music_data_b_ch2)
            train_motion_data_u_ch0 = np.load(self.root_folder + motion_data_u_ch0)
            train_music_data_u_ch0 = np.load(self.root_folder + music_data_u_ch0)

            self.motion_data = np.concatenate((train_motion_data_k_ch1, train_motion_data_b_ch2, train_motion_data_u_ch0))
            self.music_data = np.concatenate((train_music_data_k_ch1,train_music_data_b_ch2, train_music_data_u_ch0))
        self.motion_labels = [-1] * self.motion_data.shape[0]

        #self.motion_data = np.array(train_motion_data)
        #self.motion_labels = np.array(labels)

        #self.music_data = np.array(train_music_data)
        # self.music_data = np.zeros((self.raw_music.shape[0],self.raw_music.shape[1],int(40+40+84+384+1),517))
        # for i in range(self.raw_music.shape[0]):
        #     for j in range(self.raw_music.shape[1]):
        #         music = MusicFeatures(self.raw_music[i][j], SR, N_MFCCs, HOP_LENGTH, figures=False)
        #         self.music_data[i,j,:,:] = music.generate_features()
        #         print(i)
        # save_name_mu = 'pair_dance_music_features.npy'
        # np.save('../datasets/dance_level/' + save_name_mu, self.music_data)
        # print('done')
        self.music_labels = self.motion_labels
        return

    def __len__(self):
        return self.motion_data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        motion_pair = [self.motion_data[idx,0,:,:,:,:],self.motion_data[idx,1,:,:,:,:]]
        motion_label = self.motion_labels[idx]

        music_pair = [self.music_data[idx, 0, :,:], self.music_data[idx, 1, :,:]]
        music_label = self.music_labels[idx]

        return motion_pair, motion_label,music_pair,music_label


class ContrastiveLearningDanceDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    def load_aist_data(self):
        path = self.root_folder + '/dance_data/'
        motion_name = 'test_motion.npy'#'train_motion.npy'
        music_name = 'test_music.npy'#'train_music.npy'
        label_name = 'test_label.npy'#'train_label.npy'
        train_motion_data = np.load(path + motion_name)
        train_music_data = np.load(path + music_name)
        train_label = np.load(path + label_name)
        self.motion_data = np.array(train_motion_data[:294,:,:,:,:])
        self.music_data = np.array(train_music_data[:294,:,:])
        self.labels = list(train_label[:294])
        return

    def load_aist_test_data(self):
        path = self.root_folder + '/dance_data/'
        motion_name = 'test_motion.npy'
        music_name = 'test_music.npy'
        label_name = 'test_label.npy'
        train_motion_data = np.load(path + motion_name)
        train_music_data = np.load(path + music_name)
        train_label = np.load(path + label_name)
        self.motion_data = np.array(train_motion_data[294:,:,:,:,:])
        self.music_data = np.array(train_music_data[294:,:,:])
        self.labels = list(train_label[294:])
        return

    def load_supcon_data(self,split_id,num_classes):
        path = self.root_folder + '/dance_data/'
        dname = 'train_data.npy'
        #lname = 'train_label.pkl'
        train_data = np.load(path + dname)
        n = train_data.shape[0]
        pairs = []
        pairs_index = []
        label = []
        #### pos + neg
        for i in range(n):
            for j in range(n):
                if i <20 and j <20:
                    z1, z2 = i, j
                    pairs += [[train_data[z1, :, :, :, :], train_data[z2, :, :, :, :]]]
                    pairs_index += [[z1, z2]]
                    label += [1]
                elif i<20 and j>=20:
                    z1, z2 = i, j
                    pairs += [[train_data[z1, :, :, :, :], train_data[z2, :, :, :, :]]]
                    pairs_index += [[z1, z2]]
                    label += [0]
                elif i>=20 and j>=20:
                    z1, z2 = i, j
                    pairs += [[train_data[z1, :, :, :, :], train_data[z2, :, :, :, :]]]
                    pairs_index += [[z1, z2]]
                    label += [0]
        self.data = np.array(pairs)
        self.labels = np.array(label)
        self.pairs_index = np.array(pairs_index)

        return

    def create_view_pair(self, split_id, num_classes):
        w_data = np.load(self.root_folder + '/dance_data/dance_dataset_5D.npy')
        n = int(w_data.shape[0]-split_id)
        pairs = []
        pairs_index = []
        label_neg = []
        label_pos = []
        #### pos + neg
        for i in range(n):
            for j in range(n):
                z1, z2 = i,j
                pairs += [[w_data[z1,:,:,:,:], w_data[z2,:,:,:,:]]]
                pairs_index += [[z1, z2]]
                label_pos += [1]
                # inc = random.randrange(1, num_classes)
                # dn = (d + inc) % num_classes
        for i in range(n,n+split_id):
            for j in range(n,n+split_id):
                z1,z2 = i,j
                pairs += [[w_data[z1,:,:,:,:], w_data[z2,:,:,:,:]]]
                pairs_index += [[z1, z2]]
                label_neg += [-1]
        labels = label_pos + label_neg
        self.data = np.array(pairs)
        self.labels = np.array(labels)
        self.pairs_index = np.array(pairs_index)
        return

    def create_dataset_pair(self,split_id,num_classes):
        w_data = np.load(self.root_folder + '/dance_data/dance_dataset_5D.npy')
        #self.data = w_data
        n = int(w_data.shape[0] - split_id)
        pairs = []
        pairs_index = []
        label_neg = []
        label_pos = []
        #### pos + neg
        # for i in range(n):
        #     for j in range(n):
        #         z1, z2 = i,j
        #         pairs += [[w_data[z1,:,:,:,:], w_data[z2,:,:,:,:]]]
        #         pairs_index += [[z1, z2]]
        #         label_pos += [1]
        #         # inc = random.randrange(1, num_classes)
        #         # dn = (d + inc) % num_classes
        # for i in range(n):
        #     for j in range(split_id):
        #         z1,z2 = i,j
        #         pairs += [[w_data[z1,:,:,:,:], w_data[z2+n,:,:,:,:]]]
        #         pairs_index += [[z1, z2+n]]
        #         label_neg += [0]
        # labels = label_pos + label_neg
        # self.data = np.array(pairs)
        # self.labels = np.array(labels)
        # self.pairs_index = np.array(pairs_index)

        # #### pos neg pos neg
        for i in range(n):
            for j in range(split_id):
                z1, z2 = i,j
                pairs += [[w_data[z1,:,:,:,:], w_data[z2,:,:,:,:]]]
                pairs_index += [[z1, z2]]
                label_pos += [1]
                # inc = random.randrange(1, num_classes)
                # dn = (d + inc) % num_classes
        for i in range(n):
            for j in range(split_id):
                z1,z2 = i,j
                pairs += [[w_data[z1,:,:,:,:], w_data[z2,:,:,:,:]]]
                pairs_index += [[z1, z2]]
                label_neg += [0]
        labels = label_pos + label_neg
        final_pair = []
        final_label = []
        final_pair_index = []
        for i in range(len(pairs)):
            neg_count = len(label_pos)
            pos_count = 0
            if i%2 == 0:
                final_pair.append(pairs[pos_count])
                final_label.append(labels[pos_count])
                final_pair_index.append(pairs_index[pos_count])
                pos_count+=1
            else:
                final_pair.append(pairs[neg_count])
                final_label.append(labels[neg_count])
                final_pair_index.append(pairs_index[neg_count])
                neg_count += 1

        self.data = np.array(final_pair)
        self.labels = np.array(final_label)
        self.pairs_index = np.array(final_pair_index)

        return #self.data#np.array(pairs), np.array(labels), np.array(pairs_index)

    def __len__(self):
        return self.motion_data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        motion_pair = self.motion_data[idx,:,:,:,:]#[self.motion_data[idx,0,:,:,:,:],self.motion_data[idx,1,:,:,:,:]]
        music_pair = self.music_data[idx, :, :]#[self.music_data[idx, :, :], self.music_data[idx, :, :]]
        #sample_pair = [self.data[idx, 0, :, :, :], self.data[idx, 1, :, :, :]]
        pair_label = self.labels[idx]

        # if idx <= 19:
        #     id =random.randint(0, 19)
        # elif idx > 19:
        #     id = random.randint(19, 36)
        # sample_pair = [self.data[idx,:,:,:,:],self.data[id,:,:,:,:]]
        # pair_label = []
        #print(pair_label)
        return motion_pair, pair_label,music_pair,pair_label


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder


    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views),
                                                          download=True)}

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
