import torch
import torch.nn as nn
import torch.nn.functional as F
from attention.Layers import UserGeneration, SelfAttention, HybridAttentionLayer,RatioLayer
from attention.Utils import loadEmbed

class HybridAttentionModel(nn.Module):
    '''
    word_embedding -> lstm -> self attention -> hybrid attention
    '''
    def __init__(self, args):
        super(HybridAttentionModel, self).__init__()
        self.args = args
        self.embed_size = args.embed_size
        self.lstm_hidden_size = args.lstm_hidden_size


        self.word_embed = nn.Embedding.from_pretrained(loadEmbed(self.args.embed_fileName, self.embed_size, self.args.vocab_size, True))
        self.lstm = nn.LSTM(self.embed_size, self.lstm_hidden_size, batch_first=True,
                            dropout=self.args.drop_out,
                            num_layers=self.args.lstm_num_layers,
                            bidirectional=self.args.bidirectional
                            )
        self.user_layer = UserGeneration(self.args)
        self.self_atten_q = SelfAttention(self.args)
        self.self_atten_a = SelfAttention(self.args)
        # answer-question mutual attention share attention trainable weight
        self.hybrid_atten_q_a = HybridAttentionLayer(self.args)
        self.hybrid_atten_u_q = HybridAttentionLayer(self.args)
        self.ratio_layer = RatioLayer()
        self.w_q = nn.Linear(self.args.lstm_hidden_size, self.args.lstm_hidden_size, bias=False)
        self.w_a = nn.Linear(self.args.lstm_hidden_size, self.args.lstm_hidden_size, bias=False)
        self.w_u = nn.Linear(self.args.lstm_hidden_size, self.args.lstm_hidden_size, bias=False)
        self.w_final = nn.Linear(self.args.lstm_hidden_size, self.args.class_kind, bias=True)

    def reset_args(self, lstm_hidden_size,
                   lstm_num_layers,
                   kernel_size,
                 drop_out_lstm,
                 drop_out_cnn
                   ):
        self.args.lstm_hidden_size = lstm_hidden_size
        self.args.lstm_num_layers = lstm_num_layers
        self.args.kernel_size= kernel_size
        self.args.drop_out_lstm = drop_out_lstm
        self.args.drop_out_cnn = drop_out_cnn

    def _init_lstm_hidden(self):
        h_0_size_1 = 1
        if self.args.bidirectional:
            h_0_size_1 *= 2
        h_0_size_1 *= self.args.lstm_num_layers
        hiddena = torch.rand((h_0_size_1, self.args.batch_size, self.args.lstm_hidden_size),
                              dtype=torch.float, device=self.args.device)
        hiddenb = torch.rand((h_0_size_1, self.args.batch_size, self.args.lstm_hidden_size),
                              dtype=torch.float, device=self.args.device)
        return hiddena, hiddenb


    def forward(self, question, answer, user):
        '''

        :param question: N * L_q
        :param answer: N * L_a
        :param user: N * L_document
        :return:
        '''
        q_embed = self.word_embed(question)
        a_embed = self.word_embed(answer)
        u_embed = self.word_embed(user)

        #WARNING: size of vector is not always have the same length
        #Reset question and answer question length


        # TODO: bidirectional lstm  vector pool
        bi = 2 if self.args.bidirectional else 1
        hiddena, hiddenb = self._init_lstm_hidden()
        q_lstm, _ = self.lstm(q_embed, (hiddena, hiddenb))
        q_lstm = torch.mean(
            q_lstm.contiguous().view(self.args.max_q_len, self.args.batch_size, bi, self.args.lstm_hidden_size),
            2
        )
        q_lstm = q_lstm.permute(1,0,2)
        hiddena, hiddenb = self._init_lstm_hidden()
        a_lstm, _ = self.lstm(a_embed, (hiddena, hiddenb))
        a_lstm = torch.mean(
            a_lstm.contiguous().view(self.args.max_a_len, self.args.batch_size, bi, self.args.lstm_hidden_size),
            2
        )
        a_lstm = a_lstm.permute(1, 0, 2)
        assert a_lstm.size()[0] == self.args.batch_size,"dimention {} != {}".format(a_lstm.size(), self.args.batch_size)
        u_vec = self.user_layer(u_embed)
        #self attention
        q_alpha_atten = self.self_atten_q(q_lstm)
        a_alpha_atten = self.self_atten_a(a_lstm)
        #mutal atten
        q_beta_atten = self.hybrid_atten_q_a(q_lstm, a_lstm, 0)
        a_beta_atten = self.hybrid_atten_q_a(a_lstm, q_lstm, 1)
        u_lambda_atten = self.hybrid_atten_u_q(u_vec, q_lstm, 2)

        #   represent with attention
        q_yi = self.ratio_layer(q_alpha_atten, q_beta_atten)
        a_yi = self.ratio_layer(a_alpha_atten, a_beta_atten)
        u_theta = self.ratio_layer(q_alpha_atten, q_beta_atten, u_lambda_atten)
        q_h_new = (q_yi * q_lstm).sum(dim=1)
        a_h_new = (a_yi * a_lstm).sum(dim=1)
        u_h_new = (u_theta * q_lstm).sum(dim=1)
        h = torch.tanh(self.w_q(q_h_new) + self.w_a(a_h_new) + self.w_u(u_h_new))
        # batch * class
        #WARNING: check softmax dimention set
        result = F.log_softmax(self.w_final(h), dim=-1)
        _, predict = result.max(0)



        return result, predict
