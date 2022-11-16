#coding: utf-8
#----- 標準ライブラリ -----#
import os
import random
import time
from time import sleep
#----- 専用ライブラリ -----#
import glob
import hydra
import numpy as np
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as tf

#----- 自作モジュール -----#
from utils.scheduler import CosineAnnealingWarmupRestarts
from utils.dataset import Youtube_Loader
from utils.evaluation import View_evaluator
from utils.utils import CWD
#from utils.loss import 
from models.Resnet import Resnet50


def build_transform():
    test_transform = tf.Compose([tf.Resize((240, 320)),
                                tf.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                ])

    return test_transform

############## test関数 ##############
def test(model, test_transforms, eval_calculater, device):
    # モデル→推論モード
    model.eval()

    thumbnas_url = sorted(glob.glob(f"{CWD}/thumbna/*"))
    thumbnas = [Image.open(path).convert("RGB") for path in thumbnas_url]

    result = []
    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル画像
        for inputs in tqdm(thumbnas):
            inputs = test_transforms(inputs)[None]
            
            #GPUモード高速化
            inputs = inputs.cuda(device, non_blocking=True)

            # 入力画像をモデルに入力
            output = model(inputs).flatten()

            result.append(10 ** output.item())
    
    print(thumbnas_url)
    print(result)


@hydra.main(config_path="config", config_name='main.yaml')
def main(cfg):
    print(f"epoch       :{cfg.train_conf.epoch}")
    print(f"gpu         :{cfg.train_conf.gpu}")
    print(f"multi GPU   :{cfg.train_conf.multi_gpu}")
    print(f"batch size  :{cfg.train_conf.batch_size}")
    print(f"test size   :{cfg.train_conf.test_size}")
    
    ##### GPU設定 #####
    device = torch.device('cuda:{}'.format(cfg.train_conf.gpu) if torch.cuda.is_available() else 'cpu')

    
    #精度計算クラス定義
    eval_calculater = View_evaluator()

    # モデル設定
    model = Resnet50(pretrained=cfg.train_conf.pretrained).cuda(device)
    
    model_path = "result/model.pth"
    model.load_state_dict(torch.load(model_path))

    #マルチGPU
    model = torch.nn.DataParallel(model) if cfg.train_conf.multi_gpu else model

    # 初期値の乱数設定
    rand_seed = 0
    random.seed(rand_seed)  # python
    np.random.seed(rand_seed)  # numpy
    torch.manual_seed(rand_seed)  # pytorch

    
    # データ読み込み+初期設定
    test_transforms = build_transform()

    test(model, test_transforms, eval_calculater, device)



############## main ##############
if __name__ == '__main__':
    start = time.time()

    main()

    end = time.time()

    run_time = end - start
    with open("time.txt", mode='a') as f:
        f.write("{:.8f}\n".format(run_time))
