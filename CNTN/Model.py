import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils import loadEmbed
from .Layer import kmax_pooling, dynamic_k_cal


class CNTN(nn.Module):
    def __init__(self, args, word2idx, pretrained_embed):
        super(CNTN, self).__init__()
        self.word2idx = word2idx
        self.args = args
        self.embedding = nn.Embedding.from_pretrained(pretrained_embed)
        self.cnn_list = [nn.Conv2d(cnn_shape.in_channels,cnn_shape.out_channels, cnn_shape.kernel_size) for cnn_shape in self.args.cnn_shape_list]

        self.bilinear_M = nn.Bilinear(self.args.in_features, self.args.in_features, self.out_features)
        self.linear_V = nn.Linear(2 * self.args.in_features, self.args.out_features, bias=False)
        self.linear_U = nn.Linear(self.args.out_features, 1, bias=False)

        nn.init.xavier_normal_(self.bilinear_M.weight)
        nn.init.xavier_normal_(self.bilinear_M.bias)
        nn.init.xavier_normal_(self.linear_V.weight)
        nn.init.xavier_normal_(self.linear_U.weight)

    def forward(self, question, good_answer_list, bad_answer_list, label):

        question_length = question.shape[-1]
        good_answer_length = good_answer_list.shape[-1]
        bad_answer_length = bad_answer_list.shape[-1]

        question_embed = self.embedding(question)
        good_answer_embed = self.embedding(good_answer_list)
        bad_answer_embed = self.embedding(bad_answer_list)

        cnn_count = len(self.cnn_list)
        for depth, cnn in enumerate(self.cnn_list):
            question_cnn = cnn(question_embed)
            good_answer_cnn = cnn(good_answer_embed)
            bad_answer_cnn = cnn(bad_answer_embed)
            depth = depth + 1
            #k-max-pooling
            if depth < cnn_count:
                k_question = dynamic_k_cal(current_layer_depth=depth, cnn_layer_count=cnn_count, sentence_length=question_length)
                k_good_answer = dynamic_k_cal(current_layer_depth=depth, cnn_layer_count=cnn_count, sentence_length=good_answer_length)
                k_bad_answer = dynamic_k_cal(current_layer_depth=depth, cnn_layer_count=cnn_count, sentence_length=bad_answer_length)
            else:
                k_question = self.args.k_s
                k_good_answer = self.args.k_s
                k_bad_answer = self.args.k_s

            question_embed = nn.tanh(kmax_pooling(question_cnn, -2, k_question))
            good_answer_embed = nn.tanh(kmax_pooling(good_answer_cnn, -2, k_good_answer))
            bad_answer_embed  = nn.tanh(kmax_pooling(bad_answer_cnn, -2, k_bad_answer))


        good_q_m_a = self.bilinear_M(question_embed, good_answer_embed )
        good_q_m_a = good_q_m_a + self.linear_V(torch.cat((question_embed, good_answer_embed)))
        good_score = self.linear_U(good_q_m_a)

        bad_q_m_a = self.bilinear_M(question_embed, bad_answer_embed)
        bad_q_m_a = bad_q_m_a + self.linear_V(torch.cat((question_embed, bad_answer_embed)))
        bad_score = self.linear_U(bad_q_m_a)

        score = nn.ReLU(self.args.margin - good_score + bad_score)
        result = torch.cat((good_score, bad_score))
        return score, result




