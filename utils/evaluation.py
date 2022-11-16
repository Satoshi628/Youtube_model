#coding: utf-8
#----- 標準ライブラリ -----#
#None
#----- 専用ライブラリ -----#
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

#----- 自作モジュール -----#
#None

PAD_ID = -1

def one_hot_changer(tensor, vector_dim, dim=-1, bool_=False):
    """index tensorをone hot vectorに変換する関数

    Args:
        tensor (torch.tensor,dtype=torch.long): index tensor
        vector_dim (int): one hot vectorの次元。index tensorの最大値以上の値でなくてはならない
        dim (int, optional): one hot vectorをどこの次元に組み込むか. Defaults to -1.
        bool_ (bool, optional): Trueにするとbool型になる。Falseの場合はtorch.float型. Defaults to False.

    Raises:
        TypeError: index tensor is not torch.long
        ValueError: index tensor is greater than vector_dim

    Returns:
        torch.tensor: one hot vector
    """    
    if bool_:
        data_type = bool
    else:
        data_type = torch.float
    if tensor.dtype != torch.long:
        raise TypeError("入力テンソルがtorch.long型ではありません")
    if tensor.max() >= vector_dim:
        raise ValueError(f"入力テンソルのindex番号がvector_dimより大きくなっています\ntensor.max():{tensor.max()}")

    
    #one hot vector用単位行列
    one_hot = torch.eye(vector_dim, dtype=data_type, device=tensor.device)
    vector = one_hot[tensor,]
    
    #one hot vectorの次元変更
    dim_change_list = list(range(tensor.dim()))
    #もし-1ならそのまま出力
    if dim == -1:
        return vector
    #もしdimがマイナスならスライス表記と同じ性質にする
    if dim < 0:
        dim += 1 #omsertは-1が最後から一つ手前
    
    dim_change_list.insert(dim, tensor.dim())
    vector = vector.permute(dim_change_list)
    return vector


class View_evaluator():
    KEYS = ["50% Accuracy", "25% Accuracy", "Error rate"]
    def __init__(self):
        self.__50per_acc = 0.  #誤差率+-10%以下を正解とする
        self.__25per_acc = 0.  #誤差率+-5%以下を正解とする
        self.error_per = 0.  #誤差率の合計
        self.__ALL_DATA = 0.

    
    def _reset(self):
        self.__50per_acc = 0.
        self.__25per_acc = 0.
        self.error_per = 0.
        self.__ALL_DATA = 0.
    
    def update(self, outputs, targets):
        # log10 => value
        outputs = 10 ** outputs
        targets = 10 ** targets
        
        error = (outputs - targets) / torch.clamp(targets, min=1e-7) * 100
        
        self.__50per_acc += (torch.abs(error) <= 50.).sum().item()
        self.__25per_acc += (torch.abs(error) <= 25.).sum().item()
        self.error_per += error.sum().item()
        self.__ALL_DATA += int(targets.shape[0])



    def __call__(self):
        _50per_acc = self.__50per_acc / (self.__ALL_DATA + 1e-7)
        _25per_acc = self.__25per_acc / (self.__ALL_DATA + 1e-7)
        mean_error = self.error_per / (self.__ALL_DATA + 1e-7)
        
        result = [_50per_acc, _25per_acc, mean_error]
        
        self._reset()

        eval_dict = {key: r for key, r in zip(self.KEYS, result)}
        return eval_dict

    def keys(self):
        for key in self.KEYS:
            yield key
