from turtle import forward
from torch.autograd import Function
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F  # 添加这行导入


resnet_dict = {"ResNet18":models.resnet18, "ResNet34":models.resnet34, "ResNet50":models.resnet50, "ResNet101":models.resnet101, "ResNet152":models.resnet152}


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
        


# 修改base_model_blk的参数
class base_model_blk1(nn.Module):
    def __init__(self):
        super(base_model_blk1, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=4, stride=1),  # 32->16
            nn.BatchNorm1d(16),  # 32->16
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),  # kernel_size 2->4
            nn.Dropout(p=0.1)
        )

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        return x


class base_model_blk2(nn.Module):
    def __init__(self):
        super(base_model_blk2, self).__init__()

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(16, 64, kernel_size=4, stride=1),  # 32->16, 128->64
            nn.BatchNorm1d(64),  # 128->64
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),  # kernel_size 3->4
            nn.Dropout(p=0.1)  # 0.5->0.1
        )

    def forward(self, x_in):
        x = self.conv_block2(x_in)
        return x


class base_model_blk3(nn.Module):
    def __init__(self):
        super(base_model_blk3, self).__init__()
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=2, stride=1),  # 128->64, 256->128
            nn.BatchNorm1d(128),  # 256->128
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2),  # kernel_size 3->4, stride 3->2
            # 添加额外的卷积层（替代增加层数）
            nn.Conv1d(128, 256, kernel_size=2, stride=1),  # 新增128->256层
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=2, stride=1),  # 新增256->256层
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.1)  # 0.5->0.1
        )

    def forward(self, x_in):
        x = self.conv_block3(x_in)
        return x


# 修改自适应池化为MaxPool以匹配参考模型
class cnn_feature_extractor(nn.Module):
    def __init__(self):
        super(cnn_feature_extractor, self).__init__()
        self.conv_block1_shared = base_model_blk1()
        self.conv_block2_shared = base_model_blk2()
        self.conv_block3_shared = base_model_blk3()
        self.adaptive_pool = nn.AdaptiveMaxPool1d(output_size=36)  # 改为MaxPool

    def forward(self, input):
        out = self.conv_block1_shared(input)
        out = self.conv_block2_shared(out)
        out = self.conv_block3_shared(out)
        out = self.adaptive_pool(out)  # 输出固定为 (batch, 256, 36)
        return out


class Self_Attn(nn.Module):
    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # 添加维度检查
        if len(x.size()) == 2:
            # 如果输入是二维的(批次, 特征) 添加通道维度
            x = x.unsqueeze(1)  # 变为(批次, 1, 特征)

        # 确保有正确的三维
        m_batchsize, C, width = x.size()  # 现在应该是(批次, 通道, 宽度)

        # 计算query, key, value
        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)

        # 确保query, key, value是三维的
        if len(proj_query.size()) == 2:
            proj_query = proj_query.unsqueeze(1)
        if len(proj_key.size()) == 2:
            proj_key = proj_key.unsqueeze(1)
        if len(proj_value.size()) == 2:
            proj_value = proj_value.unsqueeze(1)

        # 计算注意力得分
        energy = torch.bmm(proj_query.transpose(1, 2), proj_key)
        attention = self.softmax(energy)

        # 计算加权的值
        out = torch.bmm(proj_value, attention.transpose(1, 2))
        out_flat = out.reshape(out.shape[0], -1)  # 展平为二维

        return out_flat


# 添加Flatten操作类
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CNN_RUL1(nn.Module):
    def __init__(self):
        super(CNN_RUL1, self).__init__()
        # 删除多余的模块和Flatten层
        self.encoder = nn.Sequential(
            cnn_feature_extractor(),  # 输出 (batch, 256, 36)
            Self_Attn(256)
        )

        # 添加自注意力层（在特征后，不需要Flatten）
        self.attn = Self_Attn(256)

        # 回归器直接接受自注意力输出
        self.regressor = nn.Sequential(
            nn.Linear(256 * 36, 15),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(15, 12),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(12, 10),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(10, 8),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(8, 6),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(6, 4),
            nn.LeakyReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, src):
        features = self.encoder(src)  # (batch, 256, 36)
        #attn_out = self.attn(features)  # (batch, 256 * 36)
        predictions = self.regressor(features)
        return predictions, features

""" CNN Model """
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
class CNN_RUL(nn.Module):
    def __init__(self):
        super(CNN_RUL, self).__init__()
        self.feature_layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(16, 64, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(64, 128, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2),
            nn.Conv1d(128, 256, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2),
            nn.Conv1d(256, 256, kernel_size=2, stride=1),
            nn.AdaptiveMaxPool1d(output_size=36),
            Flatten(),
            nn.Dropout(p=0.1)) 
        self.regressor= nn.Sequential(
            nn.Linear(9216, 15),   
            nn.Linear(15, 12), 
            nn.Linear(12, 10), 
            nn.Linear(10, 8),
            nn.Linear(8, 6), 
            nn.Linear(6, 4),   
            nn.LeakyReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid())    
    def forward(self, src):
        features = self.feature_layers(src)
        predictions = self.regressor(features)
        return predictions, features        



"""Resnet"""
class ResNetFc(nn.Module):
    def __init__(self, resnet_name, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=1000):
        super(ResNetFc, self).__init__()
        model_resnet = resnet_dict[resnet_name](pretrained=False)
        # self.conv1 = model_resnet.conv1
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

        self.use_bottleneck = use_bottleneck
        self.new_cls = new_cls
        if new_cls:
            if self.use_bottleneck:
                self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
                self.fc = nn.Linear(bottleneck_dim, class_num)
                self.bottleneck.apply(init_weights)
                self.fc.apply(init_weights)
                self.__in_features = bottleneck_dim
                self.regressor = nn.Sequential(self.bottleneck, self.fc, nn.Sigmoid())
            else:
                self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
                self.fc.apply(init_weights)
                self.__in_features = model_resnet.fc.in_features
                self.regressor = nn.Sequential(self.fc, nn.Sigmoid())
        else:
            self.fc = model_resnet.fc
            self.__in_features = model_resnet.fc.in_features
            self.regressor = nn.Sequential(self.fc, nn.Sigmoid())
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        y = self.regressor(x)
        return y, x

    def output_num(self):
        return self.__in_features

    def get_parameters(self):
        if self.new_cls:
            if self.use_bottleneck:
                parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
                                {"params":self.bottleneck.parameters(), "lr_mult":10, 'decay_mult':2}, \
                                {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
            else:
                parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
                                {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
        else:
            parameter_list = [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]
        return parameter_list
        
    
class Discriminator2(nn.Module):
    def __init__(self, a=1.0):
        super(Discriminator2, self).__init__()
        self.alpha = a
        self.layer = nn.Sequential(
            nn.Linear(9216, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid())
    def forward(self, input):
        input = ReverseLayer.apply(input, self.alpha)
        out = self.layer(input)
        return out
    

class ReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None