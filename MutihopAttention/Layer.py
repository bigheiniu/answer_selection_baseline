
import torch
import torch.nn as nn
from torch.nn import functional as F


class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args
        self.lstm = nn.LSTM(args.embed_size, args.lstm_hidden_size, batch_first=True,
                            dropout=args.drop_out_lstm, num_layers=args.lstm_num_layers,bidirectional = args.bidirectional)

    def lstm_init(self, size):
        h_0_size_1 = 1
        if self.args.bidirectional:
            h_0_size_1 *= 2
        h_0_size_1 *= self.args.lstm_num_layers
        hiddena = torch.zeros((h_0_size_1, size, self.args.lstm_hidden_size),
                              dtype=torch.float, device=self.args.device)
        hiddenb = torch.zeros((h_0_size_1, size, self.args.lstm_hidden_size),
                              dtype=torch.float, device=self.args.device)
        return hiddena, hiddenb

    def forward(self, input):
        hiddena, hiddenb = self.lstm_init(input.shape[-1])
        output, _ = self.lstm(input, (hiddena, hiddenb))
        return output

class SelfAttention(nn.Module):
    def __init__(self, feature_size):
        super(SelfAttention, self).__init__()
        self.atten_linear = nn.Linear(feature_size)
        self.soft_linear = nn.Linear(feature_size)

    def forward(self, hidden_list):
        s = F.tanh(self.atten_linear(hidden_list))
        weight = F.softmax(self.soft_linear(s))
        return torch.sum(weight * hidden_list, dim=-1)


class SequentialAttention(nn.Module):
    def __init__(self, args):
        super(SequentialAttention, self).__init__()
        self.args = args
        self.lstm = LSTM(self.args)


    def forward(self, question_vector, answer_hidden_list):
        question_vector.unsqueeze_(-2)
        gamma = question_vector * answer_hidden_list
        hidden, _ = self.lstm(gamma)
        weight = F.softmax(hidden)

        return torch.sum(weight * answer_hidden_list, dim = -1)


class QuestionAttention(nn.Module):
    def __init__(self, args):
        super(QuestionAttention, self).__init__()
        self.w_q = nn.Linear(args.feature_size)
        self.w_m = nn.Linear(args.feature_size)
        self.w_attention = nn.Linear(args.feature_size)


    def forward(self, hidden_list, m_q):
        S = F.tanh(self.w_q(hidden_list)) * F.tanh(self.w_m(m_q))
        alpha = F.softmax(self.w_attention(S))
        output = torch.sum(alpha * hidden_list, dim = -1)
        m_q = m_q + output
        return output, m_q



class MultiCosimilarity(nn.Module):
    def __init__(self):
        super(MultiCosimilarity, self).__init__()

    def forward(self, question_list_feature, answer_list_feature):
        #batch * k * feature
        score = torch.sum(F.cosine_similarity(question_list_feature, answer_list_feature, dim = -1), dim = -1)
        return score