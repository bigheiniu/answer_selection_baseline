import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils import loadEmbed
from Layer import kmax_pooling, dynamic_k_cal, matrix2vec_max_pooling

'''
Convolutional Neural Tensor Network Architecture for Community-based Question Answering

'''
class CNTN(nn.Module):
    def __init__(self, args, word2_vec, Content_embed):
        super(CNTN, self).__init__()
        self.args = args
        self.word_embedding = nn.Embedding.from_pretrained(word2_vec)
        # input channels and output channels are the same
        self.cnn_list = [nn.Conv2d(1, 1, kernel_size) for kernel_size in self.args.cntn_kernel_size]
        self.Content_emebd = Content_embed

        self.bilinear_M = nn.Bilinear(self.args.cntn_last_max_pool_size, self.args.cntn_last_max_pool_size, self.args.cntn_feature_r)
        self.linear_V = nn.Linear(2 * self.args.cntn_last_max_pool_size, self.args.cntn_feature_r, bias=False)
        self.linear_U = nn.Linear(self.args.cntn_feature_r, 1, bias=False)

        nn.init.xavier_normal_(self.bilinear_M.weight)
        nn.init.xavier_normal_(self.linear_V.weight)
        nn.init.xavier_normal_(self.linear_U.weight)

    def forward(self, question, answer_list):

        question_embed = self.word_embedding(self.Content_emebd.content_embed(question))
        question_embed.unsqueeze_(1)
        answer_embed = self.word_embedding(self.Content_emebd.content_embed(answer_list))
        answer_embed.unsqueeze_(1)
        question_length = question_embed.shape[-2]
        answer_length = answer_embed.shape[-2]

        cnn_count = len(self.cnn_list)
        for depth, cnn in enumerate(self.cnn_list):
            # Convolution
            question_cnn = cnn(question_embed)
            answer_cnn = cnn(answer_embed)
            depth = depth + 1
            #k-max-pooling
            if depth < cnn_count:
                k_question = dynamic_k_cal(current_layer_depth=depth, cnn_layer_count=cnn_count, sentence_length=question_length,k_top=self.args.k_max_s)
                k_answer = dynamic_k_cal(current_layer_depth=depth, cnn_layer_count=cnn_count, sentence_length=answer_length,k_top=self.args.k_max_s)
            else:
                k_question = self.args.k_max_s
                k_answer = self.args.k_max_s
            #Non-linear Feature Function
            question_embed = torch.tanh(kmax_pooling(question_cnn, -2, k_question))
            answer_embed = torch.tanh(kmax_pooling(answer_cnn, -2, k_answer))



        # transpose question/answer embedding
        # Final Layer
        question_embed = matrix2vec_max_pooling(question_embed, dim=-1)
        answer_embed = matrix2vec_max_pooling(answer_embed, dim =-1)

        q_m_a = self.bilinear_M(question_embed, answer_embed)
        q_m_a = q_m_a + self.linear_V(torch.cat((question_embed, answer_embed), dim=-1))
        q_m_a = torch.tanh(q_m_a)
        score = self.linear_U(q_m_a)
        score.squeeze_(-1)
        return score





