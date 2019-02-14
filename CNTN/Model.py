import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils import loadEmbed
from .Layer import kmax_pooling, dynamic_k_cal


class CNTN(nn.Module):
    def __init__(self, args, word2idx, word2_vec, content_embed):
        super(CNTN, self).__init__()
        self.word2idx = word2idx
        self.args = args
        self.word_embedding = nn.Embedding.from_pretrained(word2_vec)
        self.cnn_list = [nn.Conv2d(cnn_shape.in_channels,cnn_shape.out_channels, cnn_shape.kernel_size) for cnn_shape in self.args.cnn_shape_list]
        self.content_emebd = content_embed

        self.bilinear_M = nn.Bilinear(self.args.in_features, self.args.in_features, self.out_features)
        self.linear_V = nn.Linear(2 * self.args.in_features, self.args.out_features, bias=False)
        self.linear_U = nn.Linear(self.args.out_features, 1, bias=False)

        nn.init.xavier_normal_(self.bilinear_M.weight)
        nn.init.xavier_normal_(self.bilinear_M.bias)
        nn.init.xavier_normal_(self.linear_V.weight)
        nn.init.xavier_normal_(self.linear_U.weight)

    def forward(self, question, answer_list, label):

        question_length = question.shape[-1]
        answer_length = answer_list.shape[-1]

        question_embed = self.word_embedding(self.content_emebd.content_embed(question))
        answer_embed = self.word_embedding(self.content_emebd.content_embed(answer_list))

        cnn_count = len(self.cnn_list)
        for depth, cnn in enumerate(self.cnn_list):
            question_cnn = cnn(question_embed)
            answer_cnn = cnn(answer_embed)
            depth = depth + 1
            #k-max-pooling
            if depth < cnn_count:
                k_question = dynamic_k_cal(current_layer_depth=depth, cnn_layer_count=cnn_count, sentence_length=question_length)
                k_answer = dynamic_k_cal(current_layer_depth=depth, cnn_layer_count=cnn_count, sentence_length=answer_length)
            else:
                k_question = self.args.k_s
                k_answer = self.args.k_s

            question_embed = nn.tanh(kmax_pooling(question_cnn, -2, k_question))
            answer_embed = nn.tanh(kmax_pooling(answer_cnn, -2, k_answer))



        q_m_a = self.bilinear_M(question_embed, answer_embed )
        q_m_a = q_m_a + self.linear_V(torch.cat((question_embed, answer_embed)))
        score = self.linear_U(q_m_a)

        return score





