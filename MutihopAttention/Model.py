import torch
import torch.nn as nn
from MutihopAttention.Layer import *


class MultihopAttention(nn.Module):
    def __init__(self, args, word2vec_pretrained):
        super(MultihopAttention, self).__init__()
        self.args = args
        self.embed_size = args.embed_size
        self.lstm_hidden_size = args.lstm_hidden_size

        self.word_embed = nn.Embedding.from_pretrained(word2vec_pretrained)
        # batch * max_len * embed_size
        self.lstm = LSTM(self.args)
        # answer-question mutual attention share attention trainable weight
        self.K = self.args.k_layers

        self.MultiLayerQuestion_Model = nn.ModuleList(
            [
                QuestionAttention(self.args)
            for _ in range(self.K)]
        )

        self.MultiLayerAnswer_Model = nn.ModuleList(
            [
                SequentialAttention(self.args)
                for _ in range(self.K)
            ]
        )
        self.cosine = MultiCosimilarity()

    def forward(self, question, postive_answer, negative_answer):
        question_embed = self.word_embed(question)
        postive_answer_embed = self.word_embed(postive_answer)
        negative_answer_embed = self.word_embed(negative_answer)
        lstm_question,_ = self.lstm(question_embed)
        lstm_postive_answer, _ = self.lstm(postive_answer_embed)
        lstm_negative_answer, _ = self.lstm(negative_answer_embed)
        m_q = torch.mean(lstm_question, dim =-2)
        score_pos = torch.zeros(question.shape[0])
        score_neg = torch.zeros(question.shape[0])
        for i in range(self.K):
            o_q, m_q = self.MultiLayerQuestion_Model(lstm_question, m_q)
            o_a_postive = self.MultiLayerAnswer_Model(o_q, lstm_postive_answer)
            o_a_negative = self.MultiLayerAnswer_Model(o_q, lstm_negative_answer)
            score_pos += F.cosine_similarity(o_q,o_a_postive)
            score_neg += F.cosine_similarity(o_q, o_a_negative)
        #hinge loss
        score_neg = score_neg / self.K
        score_pos = score_pos / self.K
        score = torch.clamp(self.args.M - score_pos + score_neg, min = 0)
        #positive answer -> negative answer
        return score


