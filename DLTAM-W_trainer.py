import sys
from utils import *
from models.models_config import get_model_config
from models.phm_models import *
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
import wandb
import os
from data_process import *
from train_eval_phm import *
import pandas as pd

# Cross_Entropy = nn.BCEWithLogitsLoss(reduction='mean')
# Cross_Entropy2 = nn.BCEWithLogitsLoss(reduction='none')

Cross_Entropy = nn.BCELoss(reduction='mean')
Cross_Entropy2 = nn.BCELoss(reduction='none')

seed = 1

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True


def claculate_weight(input1, input2, device):
    input2 = input2.squeeze()
    input2 = input2.detach()
    h = torch.abs(input1 - input2)
    # print(h)
    h = h.detach()
    h = h.cpu().numpy()
    h = np.negative(h)
    w = np.exp(h)
    weight = np.append(w, w)
    # print(weight)
    weight = torch.tensor(weight)
    weight = weight.to(device)
    return weight


def calculate_weighted_loss(preds, labels, weight, Epsilon):
    loss1 = torch.mean(weight * Cross_Entropy2(preds, labels))
    loss2 = Cross_Entropy(preds, labels)
    e = torch.tensor(Epsilon)
    e = e.expand(len(weight)).to(device)
    loss = torch.mean((weight + e) * Cross_Entropy2(preds, labels))
    return loss  # loss1 + loss2 * Epsilon


class adversarial_loss(object):
    def __init__(self, preds, labels, weight, Epsilon, epoch, PREHEAT_STEPS, selected_model):
        self.preds = preds
        self.labels = labels
        self.weight = weight
        self.Epsilon = Epsilon
        self.epoch = epoch
        self.PREHEAT_STEPS = PREHEAT_STEPS
        self.selected_model = selected_model

    def get_loss(self):
        self.preds = torch.clamp(self.preds, 0.1, 0.9)
        # 判断是否启用加权机制（公式10）
        if self.selected_model == 'DANN' or self.epoch <= self.PREHEAT_STEPS:
            # 预热阶段使用标准对抗损失
            loss = Cross_Entropy(self.preds, self.labels)
        else:
            # 正式阶段使用加权对抗损失（公式9）
            loss = calculate_weighted_loss(self.preds, self.labels, self.weight, self.Epsilon)
        return loss


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


# 内存优化版SupConLoss
class SupConLoss(torch.nn.Module):
    """内存优化的监督对比损失"""

    def __init__(self, device, temperature=0.2, chunk_size=512):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.chunk_size = chunk_size  # 分块大小，根据可用内存调整

    def forward(self, features1, features2, temperature=None):
        if temperature is None:
            temperature = self.temperature

        B1, B2 = features1.size(0), features2.size(0)
        features = torch.cat([features1, features2], dim=0)  # [B1+B2, D]
        total_size = B1 + B2

        # 1. 构造正样本掩码 - 使用更高效的方式
        positives = torch.zeros(total_size, total_size, dtype=torch.bool, device=self.device)

        # 源域样本的正样本位置
        for i in range(B1):
            j = B1 + (i % B2)
            positives[i, j] = True
            positives[j, i] = True

        # 排除自身对比
        self_mask = torch.eye(total_size, dtype=torch.bool, device=self.device)
        positives = positives & ~self_mask

        # 2. 分块计算余弦相似度矩阵
        sim_matrix = torch.zeros(total_size, total_size, device=self.device)
        for i in range(0, total_size, self.chunk_size):
            end_i = min(i + self.chunk_size, total_size)
            chunk_i = features[i:end_i]

            for j in range(0, total_size, self.chunk_size):
                end_j = min(j + self.chunk_size, total_size)
                chunk_j = features[j:end_j]

                # 计算子块之间的余弦相似度
                chunk_sim = F.cosine_similarity(
                    chunk_i.unsqueeze(1),
                    chunk_j.unsqueeze(0),
                    dim=-1
                )

                sim_matrix[i:end_i, j:end_j] = chunk_sim

        # 3. 应用温度系数并指数化
        sim_matrix = sim_matrix / temperature
        exp_sim = torch.exp(sim_matrix)
        exp_sim.masked_fill_(self_mask, 0.0)  # 排除自身对比

        # 4. 计算分母 (所有样本相似度之和)
        # 由于分母是行求和，可以按行分块计算避免大矩阵
        denominator = torch.zeros(total_size, device=self.device)
        for i in range(0, total_size, self.chunk_size):
            end_i = min(i + self.chunk_size, total_size)
            denominator[i:end_i] = torch.sum(exp_sim[i:end_i], dim=1)

        # 5. 计算分子 (正样本相似度之和)
        # 使用掩码直接计算正样本和
        numerator = torch.sum(exp_sim * positives, dim=1)

        # 6. 计算每个样本的对比损失
        losses = -torch.log((numerator + 1e-8) / (denominator + 1e-8))

        return losses.mean()


def get_ids(id):
    if id == "OC1":
        return ['Bearing1_1'], ['Bearing1_3']
    elif id == "OC2":
        return ['Bearing2_1'], ['Bearing2_6']
    else:
        return ['Bearing3_1'], ['Bearing3_3']


def get_src_ids(id):
    if id == "OC1":
        return ['Bearing1_1', 'Bearing1_2', 'Bearing1_3', 'Bearing1_4', 'Bearing1_5', 'Bearing1_6', 'Bearing1_7'], [
            'Bearing1_7']
    elif id == "OC2":
        return ['Bearing2_1', 'Bearing2_2', 'Bearing2_3', 'Bearing2_4', 'Bearing2_5', 'Bearing2_6', 'Bearing2_7'], [
            'Bearing2_6']
    else:
        return ['Bearing3_1', 'Bearing3_2', 'Bearing3_3'], ['Bearing3_3']


def get_tgt_ids(id):
    if id == "OC1":
        return ['Bearing1_1', 'Bearing1_2'], ['Bearing1_7']
    elif id == "OC2":
        return ['Bearing2_1', 'Bearing2_2'], ['Bearing2_6']
    else:
        return ['Bearing3_1', 'Bearing3_2'], ['Bearing3_3']


def get_src_data(id):
    start_time = time.time()
    x = np.load(f"data/source_{id}_x.npy")
    y = np.load(f"data/source_{id}_y.npy")
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Source data has been loaded, Time: {epoch_mins}m {epoch_secs}s')
    if id == "OC1":
        return x, y
    elif id == "OC2":
        return x[:911], y[:911]  # np.concatenate((x[:911], x[1708:]), 0), np.concatenate((y[:911], y[1708:]), 0)
    else:
        return x, y


def get_tgt_data(id):
    start_time = time.time()
    x = np.load(f"source_{id}_x.npy")
    y = np.load(f"source_{id}_y.npy")
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Target data has been loaded, Time: {epoch_mins}m {epoch_secs}s')
    if id == "OC1":
        return x[:2803], y[:2803]
    elif id == "OC2":
        return x[:1708], y[:1708]
    else:
        return x[:2152], y[:2152]


def get_test_data(id):
    start_time = time.time()
    x = np.load(f"test_{id}_x.npy")
    y = np.load(f"test_{id}_y.npy")
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Test data has been loaded, Time: {epoch_mins}m {epoch_secs}s')
    return x, y


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
import matplotlib.ticker as ticker


def cross_domain_train2(params, device, config, model, src_id, tgt_id, norm_id, data_name, network, Gpu):
    # torch.save(data_name, "D:\\Users\\lhj\\python_project\\WADA-main\\data\\phm.pt")
    Start_Time = time.time()
    best_epoch = 0
    best_rmse = 100
    best_score = 0
    best_score_score = 0
    best_epoch_score = 0
    best_mae_score = 0
    best_rmse_score = 0

    # 初始化记录列表
    epochs = []
    rmse_values = []  # 记录RMSE
    mae_values = []  # 记录MAE
    score_values = []  # 记录Score
    # 新增：记录每次评估的epoch
    eval_epochs = []

    print(f'Domain Adaptation using: {method}')
    print(f'GPU_id: {Gpu}')

    hyper = params[f'{src_id}_{tgt_id}']
    begin = hyper['begin']
    PREHEAT_STEPS = hyper['epochs'] * begin
    Epsilon = hyper['lambda_w']
    # cotmix参数
    mix_ratio = round(hyper['mix_ratio'], 3)
    temporal_shift = hyper['temporal_shift']
    h = temporal_shift // 2  # half
    # 在训练循环之前，收集所有运行时参数
    runtime_params = {
        'method': method,
        'network': network,
        'src_id': src_id,
        'tgt_id': tgt_id,
        'seed': seed,
        'epsilon': Epsilon,
        'k_disc': hyper['k_disc'],
        'batch_size': hyper['batch_size'],
        'epochs': hyper['epochs'],
        'lr': hyper['lr'],
        'lr_D': hyper['lr_D'],
        'lambda_s': hyper['lambda_s'],
        'lambda_t': hyper['lambda_t'],
        'begin': hyper['begin'],
        'mix_ratio': mix_ratio,
        'temporal_shift': temporal_shift,
        'device': f'cuda:{gpu}',
        'epoch%number': 5,
        'test_lodaer_t_batch_size': 4
    }
    # cotmix两个损失
    source_loss = SupConLoss(device, chunk_size=256).to(device)  # 添加chunk_size控制内存
    target_loss = NTXentLoss(device, hyper['batch_size'], 0.2, True).to(device)
    print(f'Epsilon: {Epsilon}')
    print(f'From_source:{src_id}--->target:{tgt_id}...')

    if network == 'Resnet':
        train_X_s, train_Y_s = get_src_data(src_id)
        train_X_t, train_Y_t = get_tgt_data(tgt_id)
        test_X_t, test_Y_t = get_test_data(tgt_id)

        print(train_X_s.shape, train_X_t.shape)
    else:
        src_trains, src_tests = get_src_ids(src_id)
        tgt_trains, tgt_tests = get_tgt_ids(tgt_id)
        train_X_s, train_Y_s = preprocess(select='train', train_bearings=src_trains, test_bearings=src_tests,
                                          norm_id=norm_id, data_name=data_name)
        train_X_t, train_Y_t = preprocess(select='train', train_bearings=tgt_trains, test_bearings=tgt_tests,
                                          norm_id=norm_id, data_name=data_name)
        test_X_t, test_Y_t = preprocess(select='test', train_bearings=tgt_trains, test_bearings=tgt_tests,
                                        norm_id=norm_id, data_name=data_name)

    train_set_s = MyDataset_new(train_X_s, train_Y_s)
    train_set_t = MyDataset_new(train_X_t, train_Y_t)
    test_set_t = MyDataset_new(test_X_t, test_Y_t)

    train_loader_s = DataLoader(dataset=train_set_s,
                                batch_size=hyper['batch_size'],
                                shuffle=True,
                                num_workers=0,
                                drop_last=True)

    train_loader_t = DataLoader(dataset=train_set_t,
                                batch_size=hyper['batch_size'],
                                shuffle=True,
                                num_workers=0,
                                drop_last=True)
    test_loader_t = DataLoader(dataset=test_set_t,
                               batch_size=4,
                               shuffle=False,
                               num_workers=0,
                               drop_last=False)

    if network == 'Resnet':
        load_path = f'trained_models/pretrained_phm/ResNet/pretrained_{src_id}_new.pt'
        checkpoint = torch.load(load_path, map_location=f'cuda:{Gpu}')
        source_model = model(resnet_name="ResNet50", use_bottleneck=True, bottleneck_dim=128, new_cls=True,
                             class_num=1).to(device)  # 减小bottleneck维度
        target_model = model(resnet_name="ResNet50", use_bottleneck=True, bottleneck_dim=128, new_cls=True,
                             class_num=1).to(device)  # 减小bottleneck维度
        discriminator = Discriminator().to(device)
        # source_model.load_state_dict(checkpoint['state_dict'])
        # target_model.load_state_dict(checkpoint['state_dict'])
    else:
        load_path = f'trained_models/pretrain_phm/CNN_RUL/pretrained_{src_id}_new.pt'
        checkpoint = torch.load(load_path, map_location=f'cuda:{Gpu}')
        source_model = CNN_RUL1().to(device)
        target_model = CNN_RUL1().to(device)
        discriminator = Discriminator2().to(device)
        # source_model.load_state_dict(checkpoint['state_dict'])
        # target_model.load_state_dict(checkpoint['state_dict'])

    print('=' * 89)
    print(f'The Model has {count_parameters(target_model):,} trainable parameters')
    print('=' * 89)

    target_encoder = target_model.encoder
    # encoder得改这里

    criterion = RMSELoss()
    criterion2 = nn.MSELoss()
    dis_critierion = nn.BCEWithLogitsLoss()
    # optimizer
    lr_t = hyper['lr']
    lr_d = hyper['lr_D']
    discriminator_optim = torch.optim.AdamW(discriminator.parameters(), lr=hyper['lr_D'], betas=(0.5, 0.9))
    target_optim = torch.optim.AdamW(target_model.parameters(), lr=hyper['lr'], betas=(0.5, 0.9), weight_decay=5e-4)

    # discriminator_optim = torch.optim.SGD(discriminator.parameters(), lr=0.02)
    # target_optim = torch.optim.SGD(target_encoder.parameters(), lr=0.02)

    scheduler_d = StepLR(discriminator_optim, step_size=50, gamma=0.5)
    scheduler_e = StepLR(target_optim, step_size=50, gamma=0.5)

    src_only_loss, src_only_mae, src_only_score, _, _ = evaluate(source_model, test_loader_t, criterion, device, tgt_id,
                                                                 config)

    # 初始评估（epoch 0）
    print("Initial evaluation (epoch 0):")
    test_loss, test_mae, test_score, _, _ = evaluate(target_model, test_loader_t, criterion, device, tgt_id, config)

    eval_epochs.append(0)
    rmse_values.append(test_loss)
    mae_values.append(test_mae)
    score_values.append(test_score)

    print(f'Initial DA RMSE:{test_loss} \t Initial DA MAE:{test_mae} \t Initial DA Score:{test_score}')
    print('-' * 50)

    for epoch in range(1, hyper['epochs'] + 1):
        total_loss = 0
        total_accuracy = 0
        target_losses, rul_losses = 0, 0
        start_time = time.time()
        target_encoder.train()
        discriminator.train()

        len_source = len(train_loader_s)
        len_target = len(train_loader_t)
        if len_source > len_target:
            num_iter = len_source
        else:
            num_iter = len_target

        for batch_idx in range(num_iter):
            if batch_idx % len_source == 0:
                iter_source = iter(train_loader_s)
            if batch_idx % len_target == 0:
                iter_target = iter(train_loader_t)

            source_x, source_y = next(iter_source)
            target_x, target_y = next(iter_target)
            # ==== 新增：时间混合生成 ====
            # 源主导混合
            source_dominant = mix_ratio * source_x + (1 - mix_ratio) * \
                              torch.mean(torch.stack([torch.roll(target_x, -i, 2) for i in range(-h, h)], 2), 2)
            # 目标主导混合
            target_dominant = mix_ratio * target_x + (1 - mix_ratio) * \
                              torch.mean(torch.stack([torch.roll(source_x, -i, 2) for i in range(-h, h)], 2), 2)
            source_x = source_x.to(device).to(torch.float32)
            source_y = source_y.to(device).to(torch.float32)
            target_x = target_x.to(device).to(torch.float32)
            target_y = target_y.to(device).to(torch.float32)
            source_dominant = source_dominant.to(device).to(torch.float32)
            target_dominant = target_dominant.to(device).to(torch.float32)

            if config['permute'] == True:
                source_x = source_x.permute(0, 2, 1)  # permute for CNN model
                target_x = target_x.permute(0, 2, 1)
                source_dominant = source_dominant.permute(0, 2, 1)
                target_dominant = target_dominant.permute(0, 2, 1)

            set_requires_grad(target_encoder, requires_grad=True)
            set_requires_grad(discriminator, requires_grad=True)

            discriminator_optim.zero_grad()
            target_optim.zero_grad()

            source_pred, source_features = target_model(source_x)
            target_pred, target_features = target_model(target_x)

            ## 新增：时间混合生成概率和特征
            source_dominant_pred, source_dominant_features = target_model(source_dominant)
            target_dominant_pred, target_dominant_features = target_model(target_dominant)

            ### 源域监督对比损失
            discriminator_source_x_suploss = torch.cat([source_features, source_dominant_features])  # 特征和特征
            suploss = source_loss(discriminator_source_x_suploss, source_features)

            ### 目标域监督对比损失
            dicriminator_target_x_conloss = target_dominant_features
            conloss = target_loss(dicriminator_target_x_conloss, target_features)
            discriminator_x = torch.cat([source_features, target_features])
            discriminator_y = torch.cat([torch.ones(source_x.shape[0], device=device),
                                         torch.zeros(target_x.shape[0], device=device)])
            preds = discriminator(discriminator_x).squeeze()
            weight = claculate_weight(source_y, target_pred, device)
            # w = np.random.rand(512)
            # w = np.append(w, w)
            # weight = torch.tensor(w).to(device)
            weight = weight.detach()

            loss = adversarial_loss(preds, discriminator_y, weight, Epsilon, epoch, PREHEAT_STEPS,
                                    selected_model=method).get_loss()  # 加权对抗损失

            total_loss += loss.item()
            total_accuracy += ((preds > 0.5).long() == discriminator_y.long()).float().mean().item()
            a = suploss
            b = suploss.item()
            rul_loss = criterion2(source_pred.squeeze(), source_y)  # 预测损失
            loss = loss + rul_loss + hyper['lambda_s'] * suploss.item() + hyper['lambda_t'] * conloss.item()

            # total loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(target_model.parameters(), 2)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 2)
            target_optim.step()
            for i in range(4):
                discriminator_optim.step()
            discriminator_optim.step()
            rul_losses += rul_loss.item()

            # 内存清理
            del source_dominant_features, target_dominant_features, source_pred, target_pred
            torch.cuda.empty_cache()
        # scheduler_d.step()
        # scheduler_e.step()

        mean_loss = total_loss / num_iter
        mean_accuracy = total_accuracy / num_iter
        mean_rul_loss = rul_losses / num_iter

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch :02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'Adv_loss:{mean_loss} \t Discriminator_accuracy:{mean_accuracy}')
        print(f'mean_rul_loss:{mean_rul_loss}')
        mod_number = 4 if (src_id == 'OC2' and tgt_id == 'OC3') else 5
        # 在训练开始前添加计时变量
        total_eval_time = 0  # 累计评估时间
        eval_count = 0  # 评估次数
        if epoch % mod_number == 0:
            eval_start = time.time()  # 开始计时
            test_loss, test_mae, test_score, _, true_labels = evaluate(target_model, test_loader_t, criterion, device,
                                                                       tgt_id, config)

            if test_loss < best_rmse:
                best_rmse = test_loss
                best_mae = test_mae
                best_score = test_score
                best_epoch = epoch
                checkpoint = {'model': target_model,
                              'epoch': epoch,
                              'state_dict': target_model.state_dict()}
                # torch.save(checkpoint,f'D:\\Users\\lhj\\python_project\\WADA-main\\checkpoints\\{src_id}_to_{tgt_id}_best.pt')
            if test_score > best_score_score:
                best_rmse_score = test_loss
                best_mae_score = test_mae
                best_score_score = test_score
                best_epoch_score = epoch
                checkpoint_score = {'model': target_model,
                                    'epoch': epoch,
                                    'state_dict': target_model.state_dict()}
                # torch.save(checkpoint_score,f'D:\\Users\\lhj\\python_project\\WADA-main\\checkpoints\\{src_id}_to_{tgt_id}_best_score.pt')
            true_labels = sorted(true_labels, reverse=True)
            # print(true_labels)
            list = [test_loss, test_score]
            data = pd.DataFrame([list])
            eval_end = time.time()  # 结束计时
            current_eval_time = eval_end - eval_start

            # 累计时间和次数
            total_eval_time += current_eval_time
            eval_count += 1

            # 记录数据
            eval_epochs.append(epoch)
            rmse_values.append(test_loss)
            mae_values.append(test_mae)
            score_values.append(test_score)

            print(f'Src_Only RMSE:{src_only_loss} \t Src_Only MAE:{src_only_mae} \t Src_Only Score:{src_only_score}')
            print(f'DA RMSE:{test_loss} \t DA MAE:{test_mae} \t DA Score:{test_score}')
            print(f'本次评估耗时: {current_eval_time:.2f}秒')

    # 最终评估
    eval_start = time.time()
    test_loss, test_mae, test_score, pred_labels_DA, true_labels_DA = evaluate(target_model, test_loader_t, criterion,
                                                                               device, tgt_id, config)
    eval_end = time.time()
    current_eval_time = eval_end - eval_start
    total_eval_time += current_eval_time
    eval_count += 1

    # 记录最终结果
    if eval_epochs[-1] != hyper['epochs']:
        eval_epochs.append(hyper['epochs'])
        rmse_values.append(test_loss)
        mae_values.append(test_mae)
        score_values.append(test_score)

    print(f'Final DA RMSE:{test_loss} \t Final DA MAE:{test_mae} \t Final DA Score:{test_score}')

    # 在训练结束后打印总统计信息
    print(f'\n=== 评估阶段时间统计 ===')
    print(f'总评估次数: {eval_count}')
    print(f'评估阶段总耗时: {total_eval_time:.2f}秒 ({total_eval_time / 60:.2f}分钟)')
    if eval_count > 0:
        print(f'平均每次评估耗时: {total_eval_time / eval_count:.2f}秒')
    print(f'Best Epoch:{best_epoch} \t Best RMSE:{best_rmse} \t Best MAE:{best_mae} \t Best Score:{best_score}')
    print(
        f'Best Epoch_score:{best_epoch_score} \t Best RMSE_score:{best_rmse_score} \t Best MAE_score:{best_mae_score} \t Best Score_score:{best_score_score}')
    End_Time = time.time()
    epoch_mins, epoch_secs = epoch_time(Start_Time, End_Time)
    print(f'All Time: {epoch_mins}m {epoch_secs}s')

    # ========== 新增：绘制并保存训练曲线 ==========
    # 创建保存目录
    save_dir = f"training_curves/{src_id}_to_{tgt_id}_{network}"
    os.makedirs(save_dir, exist_ok=True)

    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 创建一个图形，按照您图片中的布局
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # 设置颜色和样式（与您图片中一致）
    color_score = 'blue'  # Score用蓝色
    color_rmse = 'red'  # RMSE用红色
    color_mae = 'green'  # MAE用绿色

    # 绘制Score曲线（左侧Y轴）- 使用方块标记
    line_score, = ax1.plot(eval_epochs, score_values, color=color_score, linewidth=2,
                           marker='s', markersize=8, linestyle='--', label='Score')

    # 设置左侧Y轴（Score）的标签和刻度颜色
    ax1.set_xlabel('Epoch', fontsize=24)
    ax1.set_ylabel('Score', fontsize=24, color=color_score)
    ax1.tick_params(axis='y', labelcolor=color_score)
    ax1.set_xlim([0, max(eval_epochs)])
    # 设置Score的Y轴范围
    score_min, score_max = min(score_values), max(score_values)
    ax1.set_ylim([score_min * 0.95, score_max * 1.05])

    # 创建第二个y轴（右侧内层）用于RMSE
    ax2 = ax1.twinx()
    line_rmse, = ax2.plot(eval_epochs, rmse_values, color=color_rmse, linewidth=2,
                          marker='o', markersize=8, linestyle='-', label='RMSE')
    ax2.set_ylabel('RMSE', fontsize=24, color=color_rmse)
    ax2.tick_params(axis='y', labelcolor=color_rmse)
    # 设置RMSE的Y轴范围
    rmse_min, rmse_max = min(rmse_values), max(rmse_values)
    ax2.set_ylim([rmse_min * 0.9, rmse_max * 1.1])

    # 创建第三个y轴（右侧外层）用于MAE
    # 首先复制ax2的框架
    ax3 = ax1.twinx()
    # 将MAE的坐标轴向右移动一些，避免与RMSE重叠
    # ax3.spines["right"].set_position(("axes", 1.1))
    ax3.spines['right'].set_position(('outward', 90))
    line_mae, = ax3.plot(eval_epochs, mae_values, color=color_mae, linewidth=2,
                         marker='^', markersize=8, linestyle='-.', label='MAE')
    ax3.set_ylabel('MAE', fontsize=24, color=color_mae)
    ax3.tick_params(axis='y', labelcolor=color_mae)
    # 设置MAE的Y轴范围
    mae_min, mae_max = min(mae_values), max(mae_values)
    ax3.set_ylim([mae_min * 0.9, mae_max * 1.1])

    # 设置标题
    plt.title(f'Task: {src_id}_{tgt_id}', fontsize=24, fontweight='bold', pad=20)

    # 合并图例
    lines = [line_score, line_rmse, line_mae]
    labels = ['Score', 'RMSE', 'MAE']
    # 设置Y轴刻度字体大小
    ax1.tick_params(axis='y', labelsize=20)
    ax2.tick_params(axis='y', labelsize=20)
    ax3.tick_params(axis='y', labelsize=20)
    # 设置x轴刻度为5的倍数
    # 设置x轴刻度范围从0到100，并以5的倍数进行标注
    ax1.set_xlim(0, 200)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax1.tick_params(axis='x', labelsize=20)

    # 将图例放在顶部（与您的图片一致）
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.22),
               ncol=3, fontsize=20, frameon=True, fancybox=True, shadow=True)

    # 添加网格
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图形
    fig_path = os.path.join(save_dir, f'training_curves_{timestamp}.png')
    plt.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n训练曲线已保存至: {fig_path}")

    # 保存数据到CSV文件
    data_dict = {
        'epoch': eval_epochs,
        'score': score_values,
        'rmse': rmse_values,
        'mae': mae_values
    }
    df = pd.DataFrame(data_dict)
    csv_path = os.path.join(save_dir, f'training_metrics_{timestamp}.csv')
    df.to_csv(csv_path, index=False)
    print(f"训练数据已保存至: {csv_path}")

    # 打印最佳结果总结
    print("\n" + "=" * 60)
    print("训练结果总结:")
    print("=" * 60)
    print(f"最佳Score: {max(score_values):.4f}")
    print(f"最佳RMSE: {min(rmse_values):.4f}")
    print(f"最佳MAE: {min(mae_values):.4f}")
    print("=" * 60)