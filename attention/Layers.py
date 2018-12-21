'''
Define attention attention with bilstm layer
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention.Utils import loadEmbed
__author__ = "Yichuan Li"

def xavier_uniform_init(w):
    return nn.init.xavier_uniform(w)

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

        self.conv1 = nn.Conv2d(self.args.in_channels, self.args.out_channels,
                              self.args.kernel_size)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.bn = nn.BatchNorm2d(self.args.out_channels)
        # self.conv2 = nn.Conv2d(self.args.in_channels_2, self.args.out_channels_2,
        #                       self.args.kernel_size_2)
        # self.conv3 = nn.Conv2d(self.args.in_channels_3, self.args.out_channels_3,
        #                       self.args.kernel_size_3)

        #feature map is a vector

    def forward(self, input):
        # not stack conv
        if(input.dim() != 4):
            input.unsqueeze_(1)

        x = F.relu(self.conv1(input))
        x = self.bn(x)
        x, _ = torch.max(x, -1)
        x, _ = torch.max(x, -1)
        return x



class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention,self).__init__()
        self.args = args
        self.w_1_v = nn.Linear(self.args.lstm_hidden_size, 1)
        self.w_1_v.weight = xavier_uniform_init(self.w_1_v.weight)

    def forward(self, input):
        # input: l_q* lstm_hidden_size
        alpha = F.softmax(self.w_1_v(input), dim=1)
        if(self.args.DEBUG):
            check = alpha.sum(dim=1)
        # if (self.args.DEBUG):
            # assert alpha.sum(dim=0)[0] != 1,"self attention each sentence sum {} is not 1".format(alpha.sum(dim=0)[0])
        return alpha



class HybridAttentionLayer(nn.Module):
    def __init__(self, args):
        super(HybridAttentionLayer, self).__init__()
        self.args = args
        self.w_q_m = nn.Linear(self.args.lstm_hidden_size, self.args.lstm_hidden_size)
        self.w_a_m = nn.Linear(self.args.lstm_hidden_size, self.args.lstm_hidden_size)
        self.w_aq_m = nn.Linear(self.args.lstm_hidden_size, self.args.lstm_hidden_size)
        self.w_2_v = nn.Linear(self.args.lstm_hidden_size, 1)

        self.w_q_m.weight = xavier_uniform_init(self.w_q_m.weight)
        self.w_a_m.weight = xavier_uniform_init(self.w_a_m.weight)
        self.w_aq_m.weight = xavier_uniform_init(self.w_aq_m.weight)
        self.w_2_v.weight = xavier_uniform_init(self.w_2_v.weight)


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





