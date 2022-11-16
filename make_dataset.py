#coding: utf-8
#----- 標準ライブラリ -----#
import json
import os
import shutil
import requests
from time import sleep

#----- 専用ライブラリ -----#
import pandas as pd
from tqdm import tqdm
import numpy as np
#----- 自作モジュール -----#
# None

TAG = {
    "1": "映画とアニメ",
    "2": "自動車と乗り物",
    "10": "音楽",
    "15": "ペットと動物",
    "17": "スポーツ",
    "19": "旅行とイベント",
    "20": "ゲーム",
    "22": "ブログ",
    "23": "コメディー",
    "24": "エンターテイメント",
    "25": "ニュースと政治",
    "26": "ハウツーとスタイル",
    "27": "教育",
    "28": "科学と技術",
    "29": "非営利団体と社会活動",
    "30": "映画",
    "31": "アニメ",
    "32": "アクション/アドベンチャー",
    "33": "クラシック",
    "34": "コメディー",
    "35": "ドキュメンタリー",
    "36": "ドラマ",
    "37": "家族向け",
    "38": "海外",
    "39": "ホラー",
    "40": "SF/ファンタジー",
    "41": "サスペンス",
    "42": "短編",
    "43": "番組",
    "44": "予告編",
}


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def rmdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)

def url2video(save_dir_path, video_path):
    response = requests.get(video_path)
    sleep(0.1)
    with open(save_dir_path, "wb") as saveFile:
        saveFile.write(response.content)


if __name__ == "__main__":
    #フォルダ作成
    mkdir("dataset/train")
    mkdir("dataset/test")
    
    # train rate
    train_data_ratio = 0.8

    df = pd.read_csv('dataset/data_info.csv')

    if os.path.isfile('dataset/dataset_idx.txt'):
        rand_idx = np.loadtxt("dataset/dataset_idx.txt",dtype=np.int32)
    else:
        rand_idx = np.random.randint(0, len(df), len(df),dtype=np.int32)
        np.savetxt("dataset/dataset_idx.txt", rand_idx)
    df = df.iloc[rand_idx]

    train_df = df.iloc[: int(train_data_ratio * len(df))]
    test_df = df.iloc[int(train_data_ratio * len(df)) :]

    for idx in tqdm(range(len(train_df))):
        img_url = train_df.iloc[idx, 1]
        save_path = f"dataset/train/{idx:06}.png"
        train_df.iloc[idx, 1] = save_path
        url2video(save_path, img_url)
    
    
    for idx in tqdm(range(len(test_df))):
        img_url = test_df.iloc[idx, 1]
        save_path = f"dataset/test/{idx:06}.png"
        test_df.iloc[idx, 1] = save_path
        url2video(save_path, img_url)

    
    df = pd.concat([train_df, test_df], axis=0)
    df.to_csv("dataset/dataset.csv", index=False)
