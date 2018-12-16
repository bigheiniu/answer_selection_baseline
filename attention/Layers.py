'''
Define attention attention with bilstm layer
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention.Utils import loadEmbed
__author__ = "Yichuan Li"


class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, input, dim=2):
        p = F.softmax(input, dim)
        logp  = F.log_softmax(input, dim)
        entropy = -1.0 * (p * logp).sum(dim=dim)
        return entropy


class UserGeneration(nn.Module):
    def __init__(self, args):
        super(UserGeneration, self).__init__()
        self.args = args
        # different kernel size

        self.conv1 = nn.Conv2d(self.args.in_channels_1, self.args.out_channels_1,
                              self.args.kernel_size_1)
        self.conv2 = nn.Conv2d(self.args.in_channels_2, self.args.out_channels_2,
                              self.args.kernel_size_2)
        self.conv3 = nn.Conv2d(self.args.in_channels_3, self.args.out_channels_3,
                              self.args.kernel_size_3)

        #feature map is a vector

    def forward(self, input):
        # not stack conv
        if(input.dim() != 4):
            input.unsqueeze_(1)

        x1 = torch.max(F.relu(self.conv1(input)).contiguous().view(self.args.batch_size, self.args.out_channels_1, -1), dim=2)[0]
        x2 = torch.max(F.relu(self.conv2(input)).contiguous().view(self.args.batch_size, self.args.out_channels_2, -1), dim=2)[0]
        x3 = torch.max(F.relu(self.conv3(input)).contiguous().view(self.args.batch_size, self.args.out_channels_3, -1), dim=2)[0]
        x = torch.cat((x1, x2, x3), 1).view(self.args.batch_size,-1)
        return x



class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention,self).__init__()
        self.args = args
        self.w_1_v = nn.Linear(self.args.lstm_hidden_size, 1)

    def forward(self, input):
        # input: l_q* lstm_hidden_size
        alpha = F.softmax(self.w_1_v(input), dim=1)
        if(self.args.DEBUG):
            check = alpha.sum(dim=1)
        # if (self.args.DEBUG):
            # assert alpha.sum(dim=0)[0] != 1,"self attention each sentence sum {} is not 1".format(alpha.sum(dim=0)[0])
        return alpha



class HybridAttentionLayer(nn.Module):
    def __init__(self,args):
        super(HybridAttentionLayer, self).__init__()
        self.args = args
        self.w_q_m = nn.Linear(self.args.lstm_hidden_size, self.args.lstm_hidden_size)
        self.w_a_m = nn.Linear(self.args.lstm_hidden_size, self.args.lstm_hidden_size)
        self.w_aq_m = nn.Linear(self.args.lstm_hidden_size, self.args.lstm_hidden_size)
        self.w_2_v = nn.Linear(self.args.lstm_hidden_size, 1)
        self.entropy = Entropy()

    def forward(self, content1, content2, is_user):
        if(is_user == 2):
            # content1: user vector
            # content2: question vector
            content1 = content1.unsqueeze(1).expand(-1, self.args.max_q_len, -1)
            m = torch.tanh(self.w_q_m(content1) + self.w_a_m(content2) + self.w_aq_m(content1 * content2))
            lambda_ = F.softmax(self.w_2_v(m),dim=0)
            attention_pa = lambda_
        else:
            # content1: question or answer vector // N * L_q * k
            # content2: question or answer vecotr  // N * L_a * k
            # N * L_q * k => N * L_q * L_a * k
            if(is_user == 0): #q, a
                content1 = content1.unsqueeze(2).expand(-1, -1, self.args.max_a_len, -1)
                content2 = content2.unsqueeze(1).expand(-1, self.args.max_q_len, -1, -1)
            else:   #a, q
                content1 = content1.unsqueeze(2).expand(-1, -1, self.args.max_q_len, -1)
                content2 = content2.unsqueeze(1).expand(-1, self.args.max_a_len, -1, -1)
            m = torch.tanh(self.w_a_m(content1) + self.w_q_m(content2) + self.w_aq_m(content1 * content1))

            beta_ = self.entropy(self.w_2_v(m), 2)
            attention_pa = beta_

        assert attention_pa.dim() == 3, "[ERROR] mutual attention size is {} ".format(attention_pa.shape)
        # N * L_q * 1
        # N * L_a * 1
        return attention_pa


class RatioLayer(nn.Module):
    def __init__(self):
        super(RatioLayer, self).__init__()


    def forward(self, alpha_ , beta_, lambda_ = None):
        if lambda_ is None:
            #question-answer
            ratio = alpha_ / beta_
            yi = F.softmax(ratio, dim=1)
            coef = yi
        else:
            ratio = lambda_ * alpha_ / beta_
            theta_ = F.softmax(ratio, dim=1)
            coef = theta_

        return coef



# class HybridAttentionModel(nn.Module):
#     '''
#     word_embedding -> lstm -> self attention -> hybrid attention
#     '''
#     def __init__(self, args, isUser=False):
#         super(HybridAttentionModel, self).__init__()
#         self.args = args
#         self.embed_size = self.args.embed_size
#         self.lstm_hidden_size = self.args.lstm_hidden_size
#
#
#         self.word_embed = nn.Embedding.from_pretrained(loadEmbed(self.args.embed_fileName, self.embed_size, self.args.device, True))
#         self.lstm = nn.LSTM(self.embed_size, self.lstm_hidden_size)
#         self.user_layer = UserGeneration(self.args)
#         self.self_atten_q = SelfAttention(self.args)
#         self.self_atten_a = SelfAttention(self.args)
#         # answer-question mutual attention share attention trainable weight
#         self.hybrid_atten_q_a = HybridAttention(self.args)
#         self.hybrid_atten_u_q = HybridAttention(self.args)
#         self.ratio_layer = RatioLayer()
#         self.w_q = nn.Linear(self.args.lstm_hidden_size, self.args.lstm_hidden_size, bias=False)
#         self.w_a = nn.Linear(self.args.lstm_hidden_size, self.args.lstm_hidden_size, bias=False)
#         self.w_u = nn.Linear(self.args.lstm_hidden_size, self.args.lstm_hidden_size, bias=False)
#         self.w_final = nn.Linear(self.args.lstm_hidden_size, self.args.class_kind, bias=True)
#
#     def _init_lstm_hidden(self):
#         h_0_size_1 = 1
#         if self.args.bidirectional:
#             h_0_size_1 *= 2
#         hiddena = torch.zeros((h_0_size_1, self.args.batch_size, self.args.lstm_hidden_size),
#                               dtype=torch.FloatTensor, device=self.args.device)
#         hiddenb = torch.zeros((h_0_size_1, self.args.batch_size, self.args.lstm_hidden_size),
#                               dtype=torch.FloatTensor, device=self.args.device)
#         return hiddena, hiddenb
#
#
#     def forward(self, question, answer, user):
#         '''
#
#         :param question: N * L_q
#         :param answer: N * L_a
#         :param user: N * L_document
#         :return:
#         '''
#         q_embed = self.word_embed(question)
#         a_embed = self.word_embed(answer)
#         u_embed = self.word_embed(user)
#         hiddena, hiddenb = self._init_lstm_hidden()
#         q_lstm = self.lstm(q_embed, (hiddena, hiddenb))
#         hiddena, hiddenb = self._init_lstm_hidden()
#         a_lstm = self.lstm(a_embed, (hiddena, hiddenb))
#         u_vec = self.user_layer(u_embed)
#         #self attention
#         q_alpha_atten = self.self_atten_q(q_lstm)
#         a_alpha_atten = self.self_atten_a(a_lstm)
#         #mutal atten
#         q_beta_atten = self.hybrid_atten_q_a(q_lstm, a_lstm, 0)
#         a_beta_atten = self.hybrid_atten_q_a(a_lstm, q_lstm, 1)
#         u_lambda_atten = self.hybrid_atten_u_q(u_vec, q_lstm, 2)
#
#         #   represent with attention
#         q_yi = self.ratio_layer(q_alpha_atten, q_beta_atten)
#         a_yi = self.ratio_layer(a_alpha_atten, a_beta_atten)
#         u_theta = self.ratio_layer(q_alpha_atten, q_beta_atten, u_lambda_atten)
#         q_h_new = (q_yi * q_lstm).sum(dim=1)
#         a_h_new = (a_yi * a_lstm).sum(dim=1)
#         u_h_new = (u_theta * q_lstm).sum(dim=1)
#         h = F.tanh(self.w_q(q_h_new) + self.w_a(a_h_new) + self.w_u(u_h_new))
#         result = F.log_softmax(self.w_final(h))
#         return result
#
#

