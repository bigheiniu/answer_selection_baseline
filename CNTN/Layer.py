import torch
import torch.nn as nn
import math



def kmax_pooling(x, dim, k):
    # in Convolutional Neural Tensor Network Architecture for Community-based Question Answering
    # user set ordinary matrix as embedding_size * sequence_length
    # pooling is on row vector
    # However our input data is batch * sequence_size * emebdding size
    # so what we should do is baed on emebding feature, choose particular document
    try:
        x,_ =  torch.topk(x, k=k, dim=dim, sorted=False)
    except:
        print("[EROOR] max k is {}, shape of x {}".format(k, x.shape))
    return x

def matrix2vec_max_pooling(x, dim):
    result, _ = torch.max(x, dim = dim)
    return result


def dynamic_k_cal(current_layer_depth, cnn_layer_count, k_top, sentence_length):
    k = max(k_top, math.ceil((cnn_layer_count - current_layer_depth) * 1.0 / cnn_layer_count * sentence_length))
    return k



