import torch
import torch.nn as nn
import math


def kmax_pooling(x, dim, k):
    x =  torch.topk(x, k=k, dim = dim,sorted=False)[0]
    return x


def dynamic_k_cal(current_layer_depth, cnn_layer_count, k_top, sentence_length):
    k = max(k_top, (cnn_layer_count - current_layer_depth) * 1.0 / cnn_layer_count * sentence_length)
    return k



