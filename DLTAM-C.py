
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
device = torch.device('cuda:0')
import random
from torch.autograd import Variable
# from utils import *

## Convolutional blocks
class base_model_blk1(nn.Module):
    def __init__(self, config):
        super(base_model_blk1, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(config['input_channels'], 32, kernel_size=config['kernel_size'],
                      stride=config['stride'], bias=False, padding=(config['kernel_size'] // 2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(config['dropout'])
        )

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        return x


class base_model_blk2(nn.Module):
    def __init__(self, config):
        super(base_model_blk2, self).__init__()

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(config['dropout'])
        )

    def forward(self, x_in):
        x = self.conv_block2(x_in)
        return x


class base_model_blk3(nn.Module):
    def __init__(self, config):
        super(base_model_blk3, self).__init__()
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, config['final_out_channels'], kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(config['final_out_channels'],),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(config['dropout'])
        )

    def forward(self, x_in):
        x = self.conv_block3(x_in)
        return x


class cnn_feature_extractor(nn.Module):
    def __init__(self, config):
        super(cnn_feature_extractor, self).__init__()
        self.conv_block1_shared = base_model_blk1(config)
        self.conv_block2_shared = base_model_blk2(config)
        self.conv_block3_shared = base_model_blk3(config)

    def forward(self, input):
        out = self.conv_block1_shared(input)
        out = self.conv_block2_shared(out)
        out = self.conv_block3_shared(out)

        return out

class Self_Attn(nn.Module):
    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        m_batchsize, C, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = x.view(m_batchsize, -1, width)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width)
        out_flat = out.reshape(out.shape[0], -1)
        return out_flat

""" CNN Model """


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
class CNN_RUL(nn.Module):
    def __init__(self, input_dim, hidden_dim,dropout, device,config):
        super(CNN_RUL, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.dropout=dropout
        self.device = device
        self.config=config
        self.encoder = nn.Sequential(
            cnn_feature_extractor(self.config),
            Self_Attn(config['final_out_channels']))
        self.feature_extractor = cnn_feature_extractor(self.config)
        self.attn = Self_Attn(self.feature_extractor.conv_block3_shared.conv_block3[0].out_channels)

        self.regressor= nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim//2) ,
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim//2, 1))
    def forward(self, src):
        features = self.encoder(src)
        predictions = self.regressor(features)
        return predictions, features
#cnn_model=CNN_RUL(14,30,0.5)


""" LSTM Model """
class LSTM_RUL(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout, bid, device):
        super(LSTM_RUL, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bid = bid
        self.dropout = dropout
        self.device = device
        # encoder definition
        self.encoder = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=self.bid)
        # regressor
        self.regressor= nn.Sequential(
            nn.Linear(self.hidden_dim+self.hidden_dim*self.bid, self.hidden_dim),   
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim//2) ,  
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim//2, 1))
    def forward(self, src):
        # input shape [batch_size, seq_length, input_dim]
        # outputs = [batch size, src sent len,  hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        encoder_outputs, (hidden, cell) = self.encoder(src)
#         encoder_outputs = F.dropout(torch.relu(encoder_outputs), p=0.5, training=self.training)
        # select the last hidden state as a feature
        features = encoder_outputs[:, -1:].squeeze()
        predictions = self.regressor(features)
        return predictions, features
# model=LSTM_RUL(14, 32, 5, 0.5, True, device)

class Discriminator(nn.Module):
    def __init__(self, hidden_dims,bid):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(hidden_dims+hidden_dims*self.bid, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, 1),
            nn.LogSoftmax() )
    def forward(self, input):
        out = self.layer(input)
        return out


class ConditionalEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1)
        return -1.0 * b.mean(dim=0)


class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)

#特征和特征
class SupConLoss(torch.nn.Module):
    """支持不同 batch size 的监督对比损失"""

    def __init__(self, device, temperature=0.2):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, features1, features2, temperature=None):
        if temperature is None:
            temperature = self.temperature

        B1, B2 = features1.size(0), features2.size(0)
        features = torch.cat([features1, features2], dim=0)  # [B1+B2, D]

        # 计算余弦相似度矩阵 [B1+B2, B1+B2]
        similarity = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)

        # 构造正样本掩码 -------------------------------------------------
        positives = torch.zeros_like(similarity, dtype=torch.bool)

        # 1. 处理 features1 的样本：每个 i 对应 features2 中的 i % B2
        for i in range(B1):
            j = B1 + (i % B2)  # features2 中的对应位置
            positives[i, j] = True
            positives[j, i] = True  # 对称设置

        # 2. 排除自身对比（将对角线设为 False）
        self_mask = torch.eye(B1 + B2, dtype=torch.bool, device=self.device)
        positives = positives & ~self_mask

        # 计算对比损失 ---------------------------------------------------
        # 应用温度系数并指数化
        similarity = similarity / temperature
        exp_sim = torch.exp(similarity)

        # 排除自身对比（将对角线相似度设为 -inf）
        exp_sim = exp_sim.masked_fill(self_mask, 0.0)  # 避免自身参与分母计算

        # 分子：正样本的指数相似度之和
        pos_sum = torch.sum(exp_sim * positives, dim=1)

        # 分母：所有样本的指数相似度之和（包括正样本和负样本）
        all_sum = torch.sum(exp_sim, dim=1)

        # 计算损失（添加极小值避免 log(0)）
        loss = -torch.log(pos_sum / (all_sum + 1e-8)).mean()

        return loss
