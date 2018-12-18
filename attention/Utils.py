import torch
import numpy as np
import gensim
import os
from gensim.scripts.glove2word2vec import glove2word2vec
# from sklearn.metrics import average_precision_score

def loadEmbed(file, embed_size, vocab_size, word2idx=None, Debug=True):
    # read pretrained word2vec, convert to floattensor
    if(Debug):
        embed = np.random.rand(vocab_size, embed_size)
        return torch.FloatTensor(embed)

    #load pretrained model
    else:
        print(" [INFO] load pre-trained word2vec embedding")
        sub_dir = "/".join(file.split("/")[:-1])
        if "glove" in file:
            word2vec_file = ".".join(file.split("/")[-1].split(".")[:-1])+"word2vec"+".txt"
            if word2vec_file not in os.listdir(sub_dir):
                glove2word2vec(file, os.path.join(sub_dir, word2vec_file))
            file = os.path.join(sub_dir, word2vec_file)

        model = gensim.models.KeyedVectors.load_word2vec_format(file,
                                                                binary=False)
        embed_matrix = np.zeros(vocab_size+1, embed_size)
        for word, i in word2idx.items():
            if word in model.vocab:
                embed_matrix[i] = model.word_vec(word)

        weights = torch.FloatTensor(embed_matrix)
        return weights

# def mAP(pred, label):
#     #label is binary
#
#     return average_precision_score(label, pred)

def Accuracy(pred, label):
    target = 0
    for i in range(len(pred)):
        if pred[i] == label[i]:
            target += 1
    return target * 1.0 / len(pred)

