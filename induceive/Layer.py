import torch
import torch.nn as nn
import torch.nn.functional as F



class Aggregate(nn.Module):
    def __init__(self,args):
        super(Aggregate, self).__init__()
        self.args = args
        self.biLinear = nn.Bilinear(self.args.lstm_hidden_size, self.args.lstm_hidden_size,self.args.lstm_hidden_size)
        self.Linear = nn.Linear(2 * self.args.lstm_hidden_size, self.args.lstm_hidden_size)
        self.act = nn.Sigmoid() if self.args.act == 'sigmoid' else nn.Tanh()
        self.bn = nn.BatchNorm1d(self.args.lstm_hidden_size)

    def forward(self, neighbor_list, edge_list, node, edge_score=None):
        x = self.act(self.biLinear(neighbor_list, edge_list))
        if edge_score is not None:
            x = (x * edge_score).sum(dim=-2)
        else:
            x= torch.mean(x, dim=-2)

        x = x.view(node.shape)

        x = torch.cat((x, node), dim=-1)
        x = self.act(self.Linear(x)).view(-1, node.shape[-1])
        result = self.bn(x).view(node.shape)
        return result



class EdgeAttention(nn.Module):
    def __init__(self, args):
        super(EdgeAttention, self).__init__()
        self.args = args
        self.linear = nn.Linear(self.args.lstm_hidden_size, self.args.lstm_hidden_size)
        self.a_q = nn.Linear(self.args.lstm_hidden_size, self.args.lstm_hidden_size)
        self.a_u = nn.Linear(self.args.lstm_hidden_size, self.args.lstm_hidden_size)
        self.u_v = nn.Linear(self.args.lstm_hidden_size, self.args.lstm_hidden_size)

        nn.init.xavier_normal_(self.linear.weight )
        nn.init.xavier_normal_(self.a_q.weight)
        nn.init.xavier_normal_(self.a_u.weight)
        nn.init.xavier_normal_(self.u_v.weight)
    def self_attention(self, answer):

        alpha = F.softmax(self.linear(answer), dim=-1)
        return alpha

    def forward(self, answer, question, user):
        #TODO: add self.q(question)
        beta = torch.tanh(self.a_q(question * answer) + self.a_u(answer + self.u_v(user).expand(answer.shape)))
        alpha = self.self_attention(answer)

        weight = F.softmax(alpha/beta, dim=-1)
        return weight * answer


class LstmMaxPool(nn.Module):
    def __init__(self, args):
        super(LstmMaxPool, self).__init__()
        self.args = args
        self.lstm = nn.LSTM(self.args.embed_size, self.args.lstm_hidden_size, batch_first=True,
                dropout=self.args.drop_out_lstm,
                num_layers=self.args.lstm_num_layers,
                bidirectional=self.args.bidirectional
                )

    def init_hidden(self, size):
        h_0_size_1 = 1
        if self.args.bidirectional:
            h_0_size_1 *= 2
        h_0_size_1 *= self.args.lstm_num_layers

        hiddena = torch.zeros((h_0_size_1, size, self.args.lstm_hidden_size), device=self.args.device)

        hiddenb = torch.zeros((h_0_size_1, size, self.args.lstm_hidden_size), device=self.args.device)
        return hiddena, hiddenb

    def forward(self, embed):
        shape = list(embed.shape)
        if len(shape) > 3:
            embed = embed.view(-1, self.args.max_len, self.args.embed_size)
        shape = shape[0:-2]
        size = embed.shape[0]
        hiddena, hiddenb = self.init_hidden(size)
        lstm_output,_ = self.lstm(embed, (hiddena, hiddenb))
        #max-pool
        mean_pool_output = lstm_output.mean(dim = -2)
        assert mean_pool_output.shape[-1] == self.args.lstm_hidden_size, "lstm max pooling dimention error"
        shape.append(self.args.lstm_hidden_size)
        output = mean_pool_output.view(shape)
        return output
