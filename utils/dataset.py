#coding: utf-8
#----- 標準ライブラリ -----#
import os
import random
import datetime

#----- 専用ライブラリ -----#
import glob
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import pandas as pd

#----- 自作モジュール -----#
from .utils import CWD
# None


RE_TAG = {
    "映画とアニメ": "1",
    "自動車と乗り物": "2",
    "音楽": "10",
    "ペットと動物": "15",
    "スポーツ": "17",
    "旅行とイベント": "19",
    "ゲーム": "20",
    "ブログ": "22",
    "コメディー": "23",
    "エンターテイメント": "24",
    "ニュースと政治": "25",
    "ハウツーとスタイル": "26",
    "教育": "27",
    "科学と技術": "28",
    "非営利団体と社会活動": "29",
    "映画": "30",
    "アニメ": "31",
    "アクション/アドベンチャー": "32",
    "クラシック": "33",
    "コメディー": "34",
    "ドキュメンタリー": "35",
    "ドラマ": "36",
    "家族向け": "37",
    "海外": "38",
    "ホラー": "39",
    "SF/ファンタジー": "40",
    "サスペンス": "41",
    "短編": "42",
    "番組": "43",
    "予告編": "44",
    }

TAG_TO_CLASS = {
    "1": 0,
    "2": 1,
    "10": 2,
    "15": 3,
    "17": 4,
    "19": 5,
    "20": 6,
    "22": 7,
    "23": 8,
    "24": 9,
    "25": 10,
    "26": 11,
    "27": 12,
    "28": 13,
    "29": 14,
    "30": 15,
    "31": 16,
    "32": 17,
    "33": 18,
    "34": 19,
    "35": 20,
    "36": 21,
    "37": 22,
    "38": 23,
    "39": 24,
    "40": 25,
    "41": 26,
    "42": 27,
    "43": 28,
    "44": 29,
    }

##### covid19 dataset #####
class Youtube_Loader(data.Dataset):
    # 初期設定
    def __init__(self, dataset_type='train', transform=None):
        self.df = pd.read_csv(f'{CWD}/dataset/dataset.csv')

        thumbna = self.df.iloc[:, 1].values.tolist()
        df_flag = np.array([t.split("/")[-2] == dataset_type for t in thumbna])

        self.df = self.df.iloc[df_flag]

        img_list = self.df.iloc[:, 1].values.tolist()
        #画像読み込み
        self.images = [Image.open(os.path.join(CWD, img)).convert("RGB") for img in img_list]

        # カウント処理
        Views = np.log10(np.clip(self.df.iloc[:, 7].values, a_min=1, a_max=None)).astype(np.float32)
        Iines = np.log10(np.clip(self.df.iloc[:, 8].values, a_min=1, a_max=None)).astype(np.float32)
        Comments = np.log10(np.clip(self.df.iloc[:, 9].values, a_min=1, a_max=None)).astype(np.float32)

        Views[np.isnan(Views)] = 0.
        Iines[np.isnan(Iines)] = 0.
        Comments[np.isnan(Comments)] = 0.

        # タグ処理
        Tags_class = np.array([TAG_TO_CLASS[RE_TAG[tag_name]] for tag_name in self.df.iloc[:, 5].values.tolist()])

        #投稿時間処理
        times = self.df.iloc[:, -1].values.tolist()
        day_times = []
        for time in times:
            #2022-04-01T12:00:34Z
            dt = datetime.datetime.strptime(time, '%Y-%m-%dT%H:%M:%S%z')
            dt = dt + datetime.timedelta(hours=9)
            #日本時間換算
            day_times.append((dt.time().hour) * 3600 + dt.time().minute * 60 + dt.time().second)
        
        day_times = np.array(day_times)
        day_times = day_times / (24 * 3600) * 2 * np.pi
        day_times = np.stack([np.cos(day_times), np.sin(day_times)], axis=0).astype(np.float32)

        #データ辞書に保存
        self.data_dict = {
            "View_Count": Views,
            "Iine_Count": Iines,
            "Comment_Count": Comments,
            "Title": self.df.iloc[:, 4].values.tolist(),
            "Time": day_times,
            "Tag": Tags_class
        }

        # self.dataset_type = dataset = 'train' or 'val' or 'test'
        self.dataset_type = dataset_type

        # self.transform = transform
        self.transform = transform

    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        #入力画像、ラベル取得
        image = self.images[index]
        label = self.data_dict["View_Count"][index]

        # もしself.transformが有効なら
        if self.transform:
            image = self.transform(image)

        return image, label

    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.images)
