#coding: utf-8
#----- 標準ライブラリ -----#
import os
import random
import time
from time import sleep
#----- 専用ライブラリ -----#
import hydra
import numpy as np
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
import torchvision.transforms as tf
#----- 自作モジュール -----#
from utils.scheduler import CosineAnnealingWarmupRestarts
from utils.dataset import Youtube_Loader
from utils.evaluation import View_evaluator
#from utils.loss import 
from models.Resnet import Resnet50


def build_transform():
    ### data augmentation + preprocceing ###
    train_transform = tf.Compose([tf.Resize((240,320)), #original [H,W]=[480,640]=>[240,320]
                                #tf.RandomCrop(size=(256, 256)),  # ランダムクロップ(size=256x256)
                                # ランダムに左右反転(p=probability)
                                #tf.RandomHorizontalFlip(p=0.5),
                                #tf.RandomVerticalFlip(p=0.5),
                                tf.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                ])

    test_transform = tf.Compose([tf.Resize((240, 320)),
                                tf.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                ])

    return train_transform, test_transform

############## dataloader関数##############
def dataload(cfg):
    ### data augmentation + preprocceing ###
    train_transform, test_transform = build_transform()

    train_dataset = Youtube_Loader(dataset_type='train', transform=train_transform)
    test_dataset = Youtube_Loader(dataset_type='test', transform=test_transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=cfg.train_conf.batch_size,
                                                shuffle=True,
                                                drop_last=False,
                                                num_workers=2,
                                                pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=cfg.train_conf.test_size,
                                            shuffle=False,
                                            drop_last=False,
                                            num_workers=2,
                                            pin_memory=True)


    #これらをやると
    #num_workers=os.cpu_count(),pin_memory=True
    return train_loader, test_loader

############## train関数 ##############
def train(model, train_loader, criterion, optimizer, eval_calculater, device):
    # モデル→学習モード
    model.train()

    # 初期設定
    sum_loss = 0

    # 学習ループ  inputs:入力画像  targets:教師ラベル
    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
        #GPUモード高速化
        inputs = inputs.cuda(device, non_blocking=True)
        targets = targets.cuda(device, non_blocking=True)

        # [batch, 1]に変更
        targets = targets[:, None]

        # 入力画像をモデルに入力
        output = model(inputs)

        # 損失計算
        loss = criterion(output, targets)

        # 勾配を0に
        optimizer.zero_grad()

        # 誤差逆伝搬
        loss.backward()

        # loss溜め
        sum_loss += loss.item()

        del loss  # 誤差逆伝播を実行後、計算グラフを削除

        # パラメーター更新
        optimizer.step()

        ###精度の計算###
        eval_calculater.update(output, targets)
    eval_result = eval_calculater()
    
    return sum_loss/(batch_idx+1), eval_result

############## test関数 ##############
def test(model, test_loader, criterion, eval_calculater, device):
    # モデル→推論モード
    model.eval()

    # 初期設定
    sum_loss = 0
    
    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル画像
        for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader), total=len(test_loader), leave=False):
            #GPUモード高速化
            inputs = inputs.cuda(device, non_blocking=True)
            targets = targets.cuda(device, non_blocking=True)
            # [batch, 1]に変更
            targets = targets[:, None]

            # 入力画像をモデルに入力
            output = model(inputs)

            loss = criterion(output, targets)
            
            # loss溜め
            sum_loss += loss.item()

            ###精度の計算###
            eval_calculater.update(output, targets)

        eval_result = eval_calculater()
    return sum_loss / (batch_idx + 1), eval_result


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
    
    if not os.path.exists("result"):
        os.mkdir("result")

    PATH_1 = "result/train.txt"
    PATH_2 = "result/test.txt"
    with open(PATH_1, mode='w') as f:
        f.write("epoch\ttrain loss\t")
        for k in eval_calculater.keys():
            f.write(f"{k}\t")
        f.write("\n")
    with open(PATH_2, mode='w') as f:
        f.write("epoch\ttest loss\t")
        for k in eval_calculater.keys():
            f.write(f"{k}\t")
        f.write("\n")
    

    # モデル設定
    model = Resnet50(pretrained=cfg.train_conf.pretrained).cuda(device)

    #マルチGPU
    model = torch.nn.DataParallel(model) if cfg.train_conf.multi_gpu else model

    # 初期値の乱数設定
    rand_seed = 0
    random.seed(rand_seed)  # python
    np.random.seed(rand_seed)  # numpy
    torch.manual_seed(rand_seed)  # pytorch


    # データ読み込み+初期設定
    train_loader, test_loader = dataload(cfg)

    criterion = nn.MSELoss()

    # オプティマイザー設定
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.scheduler.max_lr)  # Adam
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4) #SGD
    
    # Initialization
    scheduler = CosineAnnealingWarmupRestarts(optimizer, **cfg.scheduler)

    #高速化コードただし計算の再現性は無くなる
    torch.backends.cudnn.benchmark = True

    best_IoU = 0.0
    best_matrics = "25% Accuracy"
    # args.num_epochs = max epoch
    for epoch in range(cfg.train_conf.epoch):
        # train
        train_loss, train_result = train(model, train_loader, criterion, optimizer, eval_calculater, device)

        # test
        test_loss, test_result = test(model, test_loader, criterion, eval_calculater, device)

        ##### 結果表示 #####
        result_text = f"Epoch{epoch + 1:3d}/{cfg.train_conf.epoch:3d} \
                    Train Loss:{train_loss:.5f} train error rate:{train_result[best_matrics]:.3%} \
                    test Loss:{test_loss:.5f} test error rate:{test_result[best_matrics]:.3%}"
        print(result_text)

        ##### 出力結果を書き込み #####
        with open(PATH_1, mode='a') as f:
            f.write(f"{epoch + 1}\t{train_loss:.4f}\t")
            for v in train_result.values():
                f.write(f"{v}\t")
            f.write("\n")
        
        with open(PATH_2, mode='a') as f:
            f.write(f"{epoch + 1}\t{test_loss:.4f}\t")
            for v in test_result.values():
                f.write(f"{v}\t")
            f.write("\n")

        if test_result[best_matrics] > best_IoU:
            best_IoU = test_result[best_matrics]
            PATH = "result/model.pth"
            if cfg.train_conf.multi_gpu:
                torch.save(model.module.state_dict(), PATH)
            else:
                torch.save(model.state_dict(), PATH)
        
        scheduler.step()

    print("最高IoU:{:.3%}".format(best_IoU))


############## main ##############
if __name__ == '__main__':
    start = time.time()

    main()

    end = time.time()

    run_time = end - start
    with open("time.txt", mode='a') as f:
        f.write("{:.8f}\n".format(run_time))
