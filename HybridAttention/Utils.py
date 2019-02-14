import torch
import numpy as np
import gensim
import os
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.metrics import average_precision_score, precision_score

def loadEmbed(file, embed_size, vocab_size, word2idx=None, Debug=True):
    # read pretrained word2vec, convert to floattensor
    if(Debug):
        print("[WARN] load randn embedding for DEBUG")
        embed = np.random.rand(vocab_size, embed_size)
        return torch.FloatTensor(embed)

    #load pretrained model
    else:
        embed_matrix = np.zeros([len(word2idx), embed_size])
        print("[Info] load pre-trained word2vec embedding")
        sub_dir = "/".join(file.split("/")[:-1])
        if "glove" in file:
            word2vec_file = ".".join(file.split("/")[-1].split(".")[:-1])+"word2vec"+".txt"
            if word2vec_file not in os.listdir(sub_dir):
                glove2word2vec(file, os.path.join(sub_dir, word2vec_file))
            file = os.path.join(sub_dir, word2vec_file)

        model = gensim.models.KeyedVectors.load_word2vec_format(file,
                                                                binary=False)
        print("[Info] Load glove finish")

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
    zero_count = 0
    one_count = 0
    assert len(pred) == len(label),"length not equal"
    for i in range(len(pred)):
        if pred[i] == 0:
            zero_count += 1
        elif pred[i] == 1:
            one_count += 1
        if pred[i] == label[i]:
            target += 1
    return target * 1.0 / len(pred), zero_count / len(pred), one_count / len(pred)


def Mean_Average_Precesion(y_true, y_score, question_id):
    question_id_unique = torch.unique(question_id)
    result = 0.
    for question in question_id_unique:
        loc = question_id == question
        y_score_loc = y_score[loc]
        y_true_loc = y_true[loc]
        result += average_precision_score(y_true=y_true_loc.cpu().numpy(), y_score=y_score_loc.cpu().numpy())
    return result / (len(question_id_unique) * 1.0)



#Precision@1 computes the average number of times
#that the best answer is ranked on top by a certain algorithm.
# True_positive / (True_positive + False_postive)
def Precesion_At_One(y_true, y_score, question_id):
    y_score_new = []
    y_true_new = []
    question_id_unique = torch.unique(question_id).cpu().numpy()
    for i in question_id_unique:
        loc = question_id == i
        y_score_new.append(torch.max(y_score[loc]).item())
        y_true_new.append(y_true[torch.argmax(y_score[loc])].item())
    return precision_score(y_true_new, y_score_new)



#The Accuracy is the normalized criteria of accessing the
#ranking quality of the best answer, where Accuracy = 1
#(best) means that the best answer returned by a certain algo-
#rithm always ranks on top while Accuracy = 0 means the
#opposite.