################################################################################################################
#   This script provide the downstream evaluation for DanceMVP model - AAAI2024
#   for task 1 - dance vocabulary recognition
#   This script is developed based on the contrastive learning(SimCLR)method provided by Thalles Silva
#   https://github.com/sthalles/SimCLR/blob/master/feature_eval/mini_batch_logistic_regression_evaluator.ipynb
#
#   saved_model_path: ckpts/checkpoint_ten_genres_seg_mlp256_75.pth.tar
#   the above model is trained by 10 different dance choreographies including: ['b_ch0', 'b_ch2', 'b_ch4',
#   'k_ch0',  'k_ch3', 'k_ch4', 'u_ch1', 'u_ch3', 'h_ch1','j_ch1'] with their first 75 samples
#
#   We use another 5 unseen choreographies for the downstream evaluation:['k_ch1','u_ch2','b_ch1','h_ch0','j_ch2']
#   We use 75-90 samples of these five choreographies as the train data to tune the pre-train model
#   And use 90-100 samples of these five choreographies as the test data to get the recognition results
#
#   Motion Encoder - ST-GCN, Music Encoder - LSTM, Text Encoder - Transformer
#################################################################################################################

import torch
import argparse
import numpy as np
#from openTSNE import TSNE
from sklearn.manifold import TSNE
import time
#import tsneutil
import plotly.express as px
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
from models.st_gcn import GraphModel
from models.lstm import LSTM
from sklearn.linear_model import LogisticRegression
from models.mlp import MLP
import torch.nn.functional as F
from models.resnet_simclr import ResNetSimCLR
#from data_aug.contrastive_dataset import ContrastiveLearningLevelDataset
from data_aug.seg_dataset_text import ContrastiveLearningLevelDatasetSEG

import math
import torch.nn as nn

import random
from prompting.charmodel import CLIP
#from prompting.transformer import TransformerModel
import clip

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def distance(tensor1,tensor2):
    dist = (tensor1 - tensor2).pow(2).sum(3).sqrt()
    return dist


def lr_lambda(epoch):
    # LR to be 0.1 * (1/1+0.01*epoch)
    base_lr = 0.03
    factor = 0.01
    return base_lr/(1+factor*epoch)


if __name__ == "__main__":

    fp = './datasets/dance_level/eighteen_choe_10s/'
    split = True
    mini_batch = False
    num_samples = 25
    batch_size = 32
    pair = False
    useLSTM = True
    useSTGCN = True

    # load data
    train_dataset = ContrastiveLearningLevelDatasetSEG(fp, num_samples, split, pair, True)
    train_dataset.load_level_dataset()

    test_dataset = ContrastiveLearningLevelDatasetSEG(fp, num_samples, split, pair, False)
    test_dataset.load_level_dataset()

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=2*batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=True)



    #load trained model
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.init()

    torch.cuda.memory_summary(device=None, abbreviated=False)
    device = 'cuda:0'
    print("Using device:", device)

    #checkpoint = torch.load('./ckpts/checkpoint_0100_stgcn_res_75.pth.tar', map_location=device)
    checkpoint = torch.load('./ckpts/checkpoint_ten_genres_seg_mlp256_75.pth.tar', map_location=device)
    # checkpoint = torch.load('./ckpts/checkpoint_baselines_hcl_0050.pth.tar', map_location=device)
    #checkpoint = torch.load('./ckpts/checkpoint_0100_resres18_45.pth.tar', map_location=device)
    state_dict_mo = checkpoint['state_dict_mo']
    state_dict_mu = checkpoint['state_dict_mu']
    state_dict_mlp = checkpoint['state_dict_mlp']

    if useSTGCN == False:
        model_motion = ResNetSimCLR(base_model='resnet18',out_dim=256)
    else:
        graph_args = dict()
        graph_args['layout'] = 'mocap'
        graph_args['strategy'] = 'spatial'
        model_motion = GraphModel(device, graph_args=graph_args, in_channels=3, num_class=9,
                                  edge_importance_weighting=True)
    if useLSTM == False:
        model_music = ResNetSimCLR(base_model='resnet18', out_dim=256)
    else:
        model_music = LSTM(device=device, input_dim=130, hidden_dim=256, layer_dim=8, output_dim=256)
    model_mlp = MLP(device=device, input_dim=512, output_dim=256,project=True)

    model_motion.load_state_dict(state_dict_mo, strict=False)
    model_motion.eval()
    model_music.load_state_dict(state_dict_mu, strict=False)
    model_music.eval()
    model_mlp.load_state_dict(state_dict_mlp, strict=False)
    model_mlp.eval()


    # freeze all layers but the last mlp
    for name, param in model_motion.named_parameters():
        param.requires_grad = False
    for name, param in model_music.named_parameters():
        param.requires_grad = False
    for name, param in model_mlp.named_parameters():
        if name not in ['output_fc.weight', 'output_fc.bias']:
            param.requires_grad = False

    parameters = list(filter(lambda p: p.requires_grad, model_mlp.parameters()))
    assert len(parameters) == 2
    # parameters = list(filter(lambda p: p.requires_grad, model_mlps.parameters()))
    # assert len(parameters) == 0
    parameters = list(filter(lambda p: p.requires_grad, model_motion.parameters()))
    assert len(parameters) == 0
    parameters = list(filter(lambda p: p.requires_grad, model_music.parameters()))
    assert len(parameters) == 0


    embed_dim = 256
    # text
    context_length=8#3#6#8
    vocab_size = len(train_dataset.vocab)
    transformer_width = 256
    transformer_heads = 256
    transformer_layers = 12
    model = CLIP(embed_dim,context_length,vocab_size,transformer_width,transformer_heads,transformer_layers)
    model = model.to(device)

    import itertools
    params = [model_mlp.parameters(), model.parameters()]
    optimizer = torch.optim.Adam(itertools.chain(*params), lr=3e-2, weight_decay=0)


    criterion = torch.nn.CrossEntropyLoss().to(device)

    epochs = 200
    train_loss = []
    test_loss = []
    train_acc =[]
    test_acc = []

    pred_scores = []
    sigma = []

    for epoch in range(epochs):
        top1_train_accuracy = 0
        for counter, (images, y_batch_0, musics, y_batch_1,y_hitR,text,cls) in enumerate(train_loader):
                model.train()

                cls = cls.to(device)
                text_out = model.encode_text(cls)

                images = images.to(device)
                images = images.type(torch.cuda.FloatTensor)
                musics = musics.to(device)
                musics = musics.type(torch.cuda.FloatTensor)


                if useSTGCN == False:
                    images = torch.reshape(images,(-1,images.size(1),images.size(2),images.size(3)))
                feature_mo = model_motion(images)

                if useLSTM == False:
                    musics = torch.unsqueeze(musics,dim=-1)
                    musics = musics.permute((0,3,2,1))
                    musics = torch.cat((musics,musics,musics),dim=1)
                feature_mu = model_music(musics)

                feature = torch.cat((feature_mo,feature_mu),dim=1)
                features, _ = model_mlp(feature)
                combine_features = torch.cat((features,text_out),dim=1)
                y_batch = y_batch_0.to(device)
                logits = features @ text_out.t()
                loss = criterion(combine_features,y_batch)
            
                # #print(loss)

                train_loss.append(loss.item())

                top1 = accuracy(combine_features, y_batch, topk=(1,))
                top1_train_accuracy += top1[0]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        top1_train_accuracy /= (counter + 1)
        train_acc.append(top1_train_accuracy)
        top1_accuracy = 0
        top5_accuracy = 0
        with torch.no_grad():
            for counter, (images, y_batch_0, musics, y_batch_1,y_hitR,text,cls) in enumerate(test_loader):
                model.eval()


                cls = cls.to(device)
                text_out = model.encode_text(cls)

                images = images.to(device)
                images = images.type(torch.cuda.FloatTensor)

                musics = musics.to(device)
                musics = musics.type(torch.cuda.FloatTensor)

                if useSTGCN == False:
                    images = torch.reshape(images, (-1,images.size(1), images.size(2), images.size(3)))
                feature_mo = model_motion(images)

                if useLSTM == False:
                    musics = torch.unsqueeze(musics, dim=-1)
                    musics = musics.permute((0, 3, 2, 1))
                    musics = torch.cat((musics, musics, musics), dim=1)
                feature_mu = model_music(musics)

                feature = torch.cat((feature_mo, feature_mu), dim=1)
                features, _ = model_mlp(feature)
                combine_fea = torch.cat((features,text_out),dim=1)
                y_batch = y_batch_0.to(device)
                logits = features @ text_out.t()
                testt_loss = criterion(combine_fea, y_batch)

                #print(loss)
                test_loss.append(testt_loss.item())
                top1, top5 = accuracy(combine_fea, y_batch, topk=(1, 5))
                top1_accuracy += top1[0]
                top5_accuracy += top5[0]


            top1_accuracy /= (counter + 1)
            test_acc.append(top1_accuracy)
            top5_accuracy /= (counter + 1)
            print(
               f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")
            #print(f"Epoch {epoch}\ttrain loss {loss.item()}\tTest loss: {testt_loss.item()}")

    # import pickle
    # picklefile = "train_loss_pt_add_baseline_hcl3.pkl"
    # with open(picklefile, "wb") as pkl_wb_obj:
    #     pickle.dump(train_loss, pkl_wb_obj)
    #
    # picklefile1 = "test_loss_pt_add_baseline_hcl3.pkl"
    # with open(picklefile1, "wb") as pkl_wb_obj1:
    #     pickle.dump(test_loss, pkl_wb_obj1)
    #
    # print('done')
