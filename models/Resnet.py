#coding: utf-8
#----- 標準ライブラリ -----#
import sys

#----- 専用ライブラリ -----#
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

#----- 自作モジュール -----#
#None

def Resnet50(pretrained=True):
    resnet50 = models.resnet50(pretrained=pretrained)
    resnet50.fc = nn.Linear(2048, 1)
    return resnet50