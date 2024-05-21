################################################################################################################
#   This script provide the pre-trained encoder for DanceMVP model - AAAI2024
#   This script perform the constrastive learning on paired dance motion-music
#   The code is developed based on the contrastive learning(SimCLR)method provided by Thalles Silva
#   https://github.com/sthalles/SimCLR
#   We reformat the input data as dance music and motion, change the encoders to LSTM and ST-GCN
#################################################################################################################

import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint
import numpy as np

torch.manual_seed(0)
tau_plus = 0.1
beta = 1.0
estimator = 'hard'


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model_mo = kwargs['model_mo'].to(self.args.device)
        self.model_mu = kwargs['model_mu'].to(self.args.device)
        self.model_mlp = kwargs['model_m'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def get_negative_mask(self,batch_size):
        negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
        for i in range(batch_size):
            negative_mask[i, i] = 0
            negative_mask[i, i + batch_size] = 0

        negative_mask = torch.cat((negative_mask, negative_mask), 0)
        return negative_mask



    def HCL(self,out_1, out_2, tau_plus, batch_size, beta, estimator):
        # neg score
        out = torch.cat([out_1, out_2], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / self.args.temperature)
        old_neg = neg.clone()
        mask = self.get_negative_mask(batch_size).to(self.args.device)
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.args.temperature)
        pos = torch.cat([pos, pos], dim=0)

        # negative samples similarity scoring
        if estimator == 'hard':
            N = batch_size * 2 - 2
            imp = (beta * neg.log()).exp()
            reweight_neg = (imp * neg).sum(dim=-1) / imp.mean(dim=-1)
            Ng = (-tau_plus * N * pos + reweight_neg) / (1 - tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min=N * np.e ** (-1 / self.args.temperature))
        elif estimator == 'easy':
            Ng = neg.sum(dim=-1)
        else:
            raise Exception('Invalid estimator selected. Please use any of [hard, easy]')

        # contrastive loss
        loss = (- torch.log(pos / (pos + Ng))).mean()

        return loss

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)

        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)

        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def distance(self, tensor1, tensor2):
        dist = (tensor1 - tensor2).pow(2).sum(3).sqrt()
        return dist

    def supConLoss(self, y_true, y_pred):
        margin = 1
        square_pred = torch.square(y_pred)
        margin_square = torch.square(torch.maximum(margin - y_pred, torch.tensor(0)))
        y_true = y_true.to('cuda')
        margin_square = margin_square.to('cuda')
        square_pred = square_pred.to('cuda')
        loss = torch.mean(y_true * square_pred + (torch.tensor(1).to('cuda') - y_true) * margin_square)
        return loss

    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.device}.")
        for epoch_counter in range(self.args.epochs):

            for images, _, musics, _,_ in tqdm(train_loader):
                # pair , label
                images = torch.cat(images, dim=0)
                musics = torch.cat(musics, dim=0)
                musics.view(-1, musics.size(1), musics.size(2)).requires_grad_()

                images = images.to(self.args.device)
                images = images.type(torch.cuda.FloatTensor)
                musics = musics.to(self.args.device)
                musics = musics.type(torch.cuda.FloatTensor)

                with autocast(enabled=self.args.fp16_precision):


                    f = torch.cat((self.model_mo(images), self.model_mu(musics)), dim=1)
                    features,_ = self.model_mlp(f)

                    # ----- compute infoNCE loss  -----#
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                    # ----- compute HCL loss-----#
                    # out_1 = _[:self.args.batch_size,:]
                    # out_2 = _[self.args.batch_size:,:]
                    #
                    # loss = self.HCL(out_1, out_2, tau_plus, self.args.batch_size, beta, estimator)

                    #print('loss:', loss)



                self.optimizer.zero_grad()
                scaler.scale(loss).backward()

                clipping_value = 0.25
                torch.nn.utils.clip_grad_norm(self.model_mo.parameters(), clipping_value)
                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:

                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)
                    logging.info(f"loss, learning_rate: {loss.item(),self.scheduler.get_lr()[0]}.")

                n_iter += 1

            print('loss:', loss.item())
            print('epoch:',epoch_counter)

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({ #'arch': self.args.arch,
            'epoch': self.args.epochs,
            'state_dict_mo': self.model_mo.state_dict(),
            'state_dict_mu': self.model_mu.state_dict(),
            'state_dict_mlp': self.model_mlp.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
