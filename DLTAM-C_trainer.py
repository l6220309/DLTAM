#CMAPSS
import sys

sys.path.append("..")
from utils import *
from data.mydataset import create_dataset_full
import torch
from torch import nn
import matplotlib.pyplot as plt
from trainer.train_eval import evaluate
import copy
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
from trainer.cross_domain_models.NCE_model import NCE_model,dicriminator
import wandb
from models.my_models import SupConLoss, NTXentLoss
import matplotlib.pyplot as plt
import numpy as np


def cross_domain_train(params,device, config, model, my_dataset, src_id, tgt_id, run_id):
    hyper = params[f'{src_id}_{tgt_id}']
    mix_ratio = round(hyper['mix_ratio'], 3)
    temporal_shift = hyper['temporal_shift']
    h = temporal_shift // 2  # half
    print(f'From_source:{src_id}--->target:{tgt_id}...')
    src_train_dl, src_test_dl = create_dataset_full(my_dataset[src_id],batch_size=hyper['batch_size'])
    tgt_train_dl, tgt_test_dl = create_dataset_full(my_dataset[tgt_id],batch_size=hyper['batch_size'])
    print('Restore source pre_trained model...')
    #checkpoint = torch.load(f'/home/emad/Mohamed2/Mohamed/trained_models/single_domain/pretrained_{config["model_name"]}_{src_id}_new.pt')
    #读取预训练模型的权重
    checkpoint_path = r"D:\Users\lhj\python_project\CADA-master-CNN\trained_models\single_domain\pretrained_{}_{}_best.pt".format(config["model_name"], src_id)
    checkpoint = torch.load(checkpoint_path)

    # pretrained source model
    source_model = model(14, 32, 0.5, device, config).to(device)
    if config['tensorboard']:
      wandb.watch(source_model, log='all')
    print('=' * 89)
    print(f'The {config["model_name"]} has {count_parameters(source_model):,} trainable parameters')
    print('=' * 89)
    source_model.load_state_dict(checkpoint['state_dict'])
    source_model.eval()
    set_requires_grad(source_model, requires_grad=False)
    source_encoder = source_model.encoder

    # initialize target model
    target_model = model(14, 32, 0.5, device, config).to(device)
    target_model.load_state_dict(checkpoint['state_dict'])
    target_encoder = target_model.encoder
    target_encoder.train()
    # discriminator network
    discriminator = dicriminator().to(device)
    comput_nce= NCE_model(device).to(device)
    source_loss = SupConLoss(device).to(device)
    target_loss =NTXentLoss(device,hyper['batch_size'],0.2,True).to(device)


    # criterion
    criterion = RMSELoss()
    dis_critierion = nn.BCEWithLogitsLoss()
    # optimizer
    discriminator_optim = torch.optim.AdamW(discriminator.parameters(), lr=hyper['lr'], betas=(0.5, 0.9))
    target_optim = torch.optim.AdamW(target_encoder.parameters(), lr=hyper['lr'], betas=(0.5, 0.9))

    nce_optim = torch.optim.AdamW(comput_nce.parameters(), lr=hyper['nce_lr'], betas=(0.5, 0.9))
    if config['tensorboard']:
        comment = (f'/home/emad/Mohamed2/visualize/Scenario={src_id} to {tgt_id}')
        tb = SummaryWriter(comment)
    for epoch in range(1, hyper['epochs'] + 1):
        batch_iterator = zip(loop_iterable(src_train_dl), loop_iterable(tgt_train_dl))
        total_loss = 0
        total_accuracy = 0
        alpha = hyper['alpha_nce']
        target_losses, nce = 0, 0
        start_time = time.time()
        for _ in range(config['iterations']):  # , leave=False):
            # Train discriminator
            set_requires_grad(target_encoder, requires_grad=False)
            set_requires_grad(discriminator, requires_grad=True)
            for _ in range(config['k_disc']):
                (source_x, _), (target_x, _) = next(batch_iterator)
                source_x, target_x = source_x.to(device), target_x.to(device)
                source_x = source_x.permute(0, 2, 1)
                target_x = target_x.permute(0, 2, 1)
                _, source_features = source_model(source_x)
                _, target_features = target_model(target_x)

                discriminator_x = torch.cat([source_features, target_features])
                discriminator_y = torch.cat([torch.ones(source_x.shape[0], device=device),
                                             torch.zeros(target_x.shape[0], device=device)])
                preds = discriminator(discriminator_x).squeeze()
                loss = dis_critierion(preds, discriminator_y)  # 二元交叉熵损失
                discriminator_optim.zero_grad()
                loss.backward()
                discriminator_optim.step()
                total_loss += loss.item()  # 域判别器预测的损失和
                total_accuracy += ((preds > 0).long() == discriminator_y.long()).float().mean().item()
            # Train Feature Extractor
            set_requires_grad(target_encoder, requires_grad=True)
            set_requires_grad(discriminator, requires_grad=False)
            for _ in range(config['k_clf']):
                target_optim.zero_grad()
                nce_optim.zero_grad()
                # Get a batch
                (source_x, _), (target_x, _) = next(batch_iterator)
                source_x, target_x = source_x.to(device), target_x.to(device)

                # ==== 新增：时间混合生成 ====
                # 源主导混合
                source_dominant = mix_ratio * source_x + (1 - mix_ratio) * \
                               torch.mean(torch.stack([torch.roll(target_x, -i, 2) for i in range(-h, h)], 2), 2)
                # 目标主导混合
                target_dominant = mix_ratio * target_x + (1 - mix_ratio) * \
                               torch.mean(torch.stack([torch.roll(source_x, -i, 2) for i in range(-h, h)], 2), 2)

                # 特征提取
                source_x = source_x.permute(0, 2, 1)
                target_x = target_x.permute(0, 2, 1)
                source_dominant = source_dominant.permute(0, 2, 1)
                target_dominant = target_dominant.permute(0, 2, 1)
                _, source_feat = source_model(source_x)
                _, target_feat = target_model(target_x)
                _, source_dom_feat = source_model(source_dominant)
                _, target_dom_feat = target_model(target_dominant)

                #源域监督对比损失
                # discriminator_source_x_suploss = torch.cat([source_feat.unsqueeze(1), source_dom_feat.unsqueeze(1)], dim=1)#特征和标签
                discriminator_source_x_suploss = torch.cat([source_feat, source_dom_feat])#特征和特征
                discriminator_source_y_suploss =torch.ones(source_x.shape[0], device=device)
                preds_suploss1 = discriminator(discriminator_source_x_suploss)
                preds_suploss = discriminator(discriminator_source_x_suploss).squeeze()
                suploss = source_loss(discriminator_source_x_suploss, source_feat)
                #目标域监督对比损失
                dicriminator_target_x_conloss = target_dom_feat
                discriminator_target_y_conloss = torch.zeros(target_x.shape[0], device=device)
                preds_conloss = discriminator(dicriminator_target_x_conloss).squeeze()
                conloss = target_loss(dicriminator_target_x_conloss, target_feat)

                # flipped labels
                discriminator_y_advloss = torch.zeros(target_x.shape[0], device=device)
                preds = discriminator(target_features).squeeze()
                adv_loss = dis_critierion(preds, discriminator_y_advloss)
                # Negaative Contrastive Estimtion Loss
                nce_loss= comput_nce(target_features,target_x)
                #total loss
                loss = adv_loss + alpha*nce_loss+ hyper['lambda_s']*suploss + hyper['lambda_t']*conloss
                loss.backward()
                target_optim.step()
                nce_optim.step()
                target_losses += adv_loss.item()
                nce += nce_loss.item()
        mean_loss = total_loss / (config['iterations'] * config['k_disc'])#域判别器平均损失
        mean_accuracy = total_accuracy / (config['iterations'] * config['k_disc'])
        mean_nce = nce / (config['iterations'] * config['k_clf'])#单单NCE的损失
        mean_tgt_loss = target_losses / (config['iterations'] * config['k_clf'])#目标模型试图欺骗判别器的对抗损失
        #iterations控制每个rpoch的批处理次数，k_disc控制域判别器更新频率，k_clf控制特征提取器的更新频率
        # tensorboard logging
        # log time
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'Discriminator_loss:{mean_loss} \t Discriminator_accuracy: {mean_accuracy}')
        print(f'target_loss:{mean_tgt_loss}  \t NCE_loss: {mean_nce}')
        # 在训练开始前添加计时变量
        total_eval_time = 0  # 累计评估时间
        eval_count = 0  # 评估次数
        if epoch % 10 == 0:
            eval_start = time.time()  # 开始计时
            src_only_loss, src_only_score, _, _, _, _ = evaluate(source_model, tgt_test_dl, criterion, config,device)
            test_loss, test_score, _, _, _, _ = evaluate(target_model, tgt_test_dl, criterion, config,device)
            print(f'Src_Only RMSE:{src_only_loss} \t Src_Only Score:{src_only_score}')
            print(f'DA RMSE:{test_loss} \t DA Score:{test_score}')
            eval_end = time.time()  # 结束计时
            current_eval_time = eval_end - eval_start
            # 累计时间和次数
            total_eval_time += current_eval_time
            eval_count += 1
            print(f'DA RMSE:{test_loss} \t DA Score:{test_score}')
            print(f'本次评估耗时: {current_eval_time:.2f}秒')
            if config['tensorboard_epoch']:
              tb.add_scalar('Loss/Src_Only', src_only_loss, epoch)
              tb.add_scalar('Loss/DA', test_loss, epoch)
              tb.add_scalar('Score/Src_Only', src_only_score, epoch)
              tb.add_scalar('Score/DA', test_score, epoch)

    src_only_loss, src_only_score, _, _, pred_labels, true_labels = evaluate(source_model, tgt_test_dl, criterion, config,device)#使用冻结的元模型直接预测
    test_loss, test_score, _, _, pred_labels_DA, true_labels_DA = evaluate(target_model, tgt_test_dl, criterion, config,device)#使用动态训练的目标模型预测
    # 替换原有的简单绘图代码 -> 专业可视化开始

    # # 排序处理
    # sorted_idx = np.argsort(true_labels_DA)[::-1]
    # sorted_true = np.array(true_labels_DA)[sorted_idx]
    # sorted_pred = np.array(pred_labels_DA)[sorted_idx]
    #
    # # 创建专业图表
    # plt.figure(figsize=(14, 8), dpi=120)
    #
    # # ===== 分层误差带系统 =====
    # error_bands = [
    #     {"range": 30, "color": "#FFA07A", "alpha": 0.3, "label": "±30 Cycles"},
    #     {"range": 20, "color": "#FFD700", "alpha": 0.4, "label": "±20 Cycles"},
    #     {"range": 10, "color": "#90EE90", "alpha": 0.6, "label": "±10 Cycles"}
    # ]
    #
    # for band in error_bands:
    #     plt.fill_between(range(len(sorted_true)),
    #                      sorted_true - band["range"],
    #                      sorted_true + band["range"],
    #                      color=band["color"],
    #                      alpha=band["alpha"],
    #                      label=band["label"])
    #
    # # ===== 核心数据曲线 =====
    # plt.plot(sorted_true, color='#FF7F0E', linewidth=2.5, label='True RUL')
    # plt.scatter(range(len(sorted_pred)), sorted_pred,
    #             color='black', marker='^', s=80, label='Predicted RUL')
    #
    # # ===== 安全关键区处理 =====
    # short_life_idx = np.where(sorted_true <= 30)[0]
    # if len(short_life_idx) > 0:
    #     # 红色背景高亮
    #     plt.axvspan(short_life_idx[0], short_life_idx[-1],
    #                 color='#FFCCCC', alpha=0.4, label='Critical Zone (RUL≤30)')
    #
    #     # # 专业安全标记
    #     # critical_text = "SAFETY CRITICAL\n(Immediate Maintenance Required)"
    #     # plt.text(np.median(short_life_idx), max(sorted_true) * 0.85,
    #     #          critical_text,
    #     #          fontsize=11,
    #     #          color='red',
    #     #          ha='center',
    #     #          va='center',
    #     #          bbox=dict(boxstyle="round,pad=0.5", fc='white', ec='red', lw=2))
    #
    # # ===== 专业格式设置 =====
    # plt.title(f'Precision-Graded RUL Prediction: {src_id}→{tgt_id}', fontsize=30, pad=15)
    # plt.xlabel('Test Engines (Sorted by Descending True RUL)', fontsize=26)
    # plt.ylabel('Remaining Useful Life (Cycles)', fontsize=26)
    # # 设置坐标轴刻度字体大小 (新增部分)
    # plt.xticks(fontsize=22)  # 横坐标刻度字体大小
    # plt.yticks(fontsize=22)  # 纵坐标刻度字体大小
    # plt.grid(True, linestyle='--', alpha=0.7)
    #
    # # 智能图例处理
    # handles, labels = plt.gca().get_legend_handles_labels()
    # unique_labels = dict(zip(labels, handles))
    # plt.legend(unique_labels.values(), unique_labels.keys(),
    #            loc='upper right',
    #            framealpha=0.9,
    #            fontsize=16,
    #            title="Legend")
    #
    # # 动态范围调整
    # buffer = max(10, 0.1 * (max(sorted_true) - min(sorted_true)))
    # plt.ylim(min(sorted_true) - buffer, max(sorted_true) + buffer)
    #
    # plt.tight_layout()
    # # 保存图片到指定路径
    # save_path = r"D:\Users\lhj\python_project\CADA-master-CNN\results\visualize"
    # # 确保目录存在
    # import os
    # os.makedirs(save_path, exist_ok=True)
    #
    # # 创建文件名（可以添加时间戳或其他标识符使文件名唯一）
    # filename = f"RUL_prediction_{src_id}_to_{tgt_id}.png"
    # full_path = os.path.join(save_path, filename)
    #
    # plt.savefig(full_path, dpi=120, bbox_inches='tight')
    # plt.close()

    print(f'Src_Only RMSE:{src_only_loss} \t Src_Only Score:{src_only_score}')
    print(f'After DA RMSE:{test_loss} \t After DA Score:{test_score}')
    # print the true and predicted labels
    if config['tensorboard']:
        tb.add_figure('Src_Only', fig1)
        tb.add_figure('DA', fig2)
        tb.add_scalar('Loss/Src_Only', src_only_loss, epoch)
        tb.add_scalar('Loss/DA', test_loss, epoch)
        tb.add_scalar('Score/Src_Only', src_only_score, epoch)
        tb.add_scalar('Score/DA', test_score, epoch)
        if config['tsne']:
            _, _, src_features, _, _, _ = evaluate(source_model, src_train_dl, criterion, config,device)
            _, _, tgt_features, _, _, _ = evaluate(source_model, tgt_train_dl, criterion, config,device)
            _, _, tgt_trained_features, _, _, _ = evaluate(target_model, tgt_train_dl, criterion, config,device)
            tb.add_embedding(src_features)
            tb.add_embedding(tgt_features)
            tb.add_embedding(tgt_trained_features)
    # if config['save']:
    #     torch.save(target_model.state_dict(), f'D:\\Users\\lhj\\python_project\\CADA-master-CNN\\trained_models\\cross_domains_final\\{src_id}_to_{tgt_id}_{run_id}_best_final.pt')
    return src_only_loss, src_only_score, test_loss, test_score
