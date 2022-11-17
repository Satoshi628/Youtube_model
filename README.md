# Youtubeサムネ予測AI
機械学習でYoutubeのサムネから再生回数を予測するモデルのサンプルコード

## 実行手順
1. ライブラリのインストール
公式のDocker images nvcr.io/nvidia/pytorchをベースにそこから以下のライブラリをインストールしてください
```
pip install hydra-core
pip install pandas
pip install matplotlib
```

2. データセットのダウンロード
以下のコードを実行するとサムネ画像をダウンロードし、trainフォルダとtestフォルダに分けるます。
```
python make_dataset.py
```

3. 学習開始
学習を開始します。shファイルに実行例があるので参考にしてください。
```
sh run.sh
```
