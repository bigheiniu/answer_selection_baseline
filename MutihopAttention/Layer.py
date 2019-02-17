
import torch
import torch.nn as nn
from torch.nn import functional as F


class LSTM(nn.Module):
    def __init__(self, args, embed_size=-1):
        super(LSTM, self).__init__()
        self.args = args
        if embed_size == -1:
            embed_size = args.embed_size
        self.lstm = nn.LSTM(embed_size, args.lstm_hidden_size, batch_first=True,
                            dropout=args.drop_out_lstm, num_layers=args.lstm_num_layers,bidirectional = args.bidirectional)

    def lstm_init(self, batch_size):
        h_0_size_1 = 1
        if self.args.bidirectional:
            h_0_size_1 *= 2
        h_0_size_1 *= self.args.lstm_num_layers
        hiddena = torch.zeros((h_0_size_1, batch_size, self.args.lstm_hidden_size),
                              dtype=torch.float, device=self.args.device)
        hiddenb = torch.zeros((h_0_size_1, batch_size, self.args.lstm_hidden_size),
                              dtype=torch.float, device=self.args.device)
        return hiddena, hiddenb

    def forward(self, input):
        shape = [*input.shape]
        input = input.view(-1, shape[-2], shape[-1])
        shape[-1] = self.args.lstm_hidden_size
        del shape[-2]
        hiddena, hiddenb = self.lstm_init(input.shape[0])
        output, _ = self.lstm(input, (hiddena, hiddenb))
        return output




class MLPAttention(nn.Module):
    def __init__(self, args):
        super(MLPAttention, self).__init__()
        self.args = args
        self.linear_answer = nn.Linear(self.args.bidire_layer * self.args.lstm_hidden_size, self.args.bidire_layer * self.args.lstm_hidden_size, bias=False)
        self.linear_question = nn.Linear(self.args.bidire_layer * self.args.lstm_hidden_size, self.args.bidire_layer * args.lstm_hidden_size, bias=False)
        self.linear_wm = nn.Linear(self.args.bidire_layer * self.args.lstm_hidden_size, 1, bias=False)

    def forward(self, lstm_vector_question, lstm_matrix_answer):
        m = torch.tanh(self.linear_answer(lstm_matrix_answer) + self.linear_question(lstm_vector_question).unsqueeze(-2))
        attention = F.softmax(self.linear_wm(m), dim=-2)
        return torch.sum(attention * m, dim = -2)



class BilinearAttention(nn.Module):
    def __init__(self, args):
        super(BilinearAttention, self).__init__()
        self.args = args
        self.bilinear_ws = nn.Bilinear(self.args.bidire_layer * self.args.lstm_hidden_size, self.args.bidire_layer * self.args.lstm_hidden_size, 1, bias=False)
        # self.linear_ws = nn.Linear(args.lstm_hidden_size, args.lstm_hidden_size, bias=False)

    def forward(self, question_vector, answer_matrix):
        if len(question_vector.shape) < 3:
            #batch_size, 1, hidden_size
            question_vector.unsqueeze_(-2)
        sequence_length = answer_matrix.shape[-2]
        lstm_matrix_question = question_vector.expand(-1, sequence_length, -1)
        th = self.bilinear_ws(lstm_matrix_question, answer_matrix)
        attention_weight = F.softmax(th, dim=-2)
        return torch.sum(attention_weight * answer_matrix, dim=-2)


class SequentialAttention(nn.Module):
    def __init__(self, args):
        super(SequentialAttention, self).__init__()
        self.args = args
        self.lstm = LSTM(self.args, embed_size=self.args.bidire_layer * self.args.lstm_hidden_size)


    def forward(self, question_vector, answer_matrix):
        if len(question_vector.shape) < 3:
            #batch_size, 1, hidden_size
            question_vector.unsqueeze_(-2)
        gamma = question_vector * answer_matrix
        hidden = self.lstm(gamma)
        hidden = torch.sum(hidden, dim=-1, keepdim=True)
        weight = F.softmax(hidden, dim=-2)

        return torch.sum(weight * answer_matrix, dim=-2)

class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        self.args = args
        self.linear_ws = nn.Linear(self.args.bidire_layer * self.args.lstm_hidden_size, self.args.bidire_layer * self.args.lstm_hidden_size, bias=True)
        self.linear_attention = nn.Linear(self.args.bidire_layer * self.args.lstm_hidden_size, 1, bias=False)

    def forward(self, lstm_hidden_list):
        s = torch.tanh(self.linear_ws(lstm_hidden_list))
        weight = F.softmax(self.linear_attention(s), dim=-2)
        return torch.sum(weight * lstm_hidden_list, dim=-2)


class QuestionAttention(nn.Module):
    def __init__(self, args):
        super(QuestionAttention, self).__init__()
        self.args = args
        self.w_q = nn.Linear(self.args.bidire_layer * self.args.lstm_hidden_size, self.args.bidire_layer * self.args.lstm_hidden_size)
        self.w_m = nn.Linear(self.args.bidire_layer * self.args.lstm_hidden_size, self.args.bidire_layer * self.args.lstm_hidden_size)
        self.w_attention = nn.Linear(self.args.bidire_layer * self.args.lstm_hidden_size, 1)


    def forward(self, hidden_list, m_q):
        assert len(m_q.shape) == 2,"shape of m_q is {}".format(m_q.shape)
        S = torch.tanh(self.w_q(hidden_list)) * torch.tanh(self.w_m(m_q).unsqueeze(-2))
        alpha = F.softmax(self.w_attention(S), dim=-2)
        output = torch.sum(alpha * hidden_list, dim=-2)
        m_q = m_q + output
        return output, m_q



class MultiCosimilarity(nn.Module):
    def __init__(self):
        super(MultiCosimilarity, self).__init__()

    def forward(self, question_list_feature, answer_list_feature):
        #batch * k * feature
        score = torch.sum(F.cosine_similarity(question_list_feature, answer_list_feature, dim = -1), dim = -1)
        return score