import torch
import torch.nn as nn
import torch.nn.functional as F
from attention.Layers import UserGeneration, SelfAttention, HybridAttentionLayer,RatioLayer
from attention.Utils import loadEmbed
from visualization.logger import  Logger


info = {}
logger = Logger('./log')

class HybridAttentionModel(nn.Module):
    '''
    word_embedding -> lstm -> self attention -> hybrid attention
    '''
    def __init__(self, args, word2idx, pretrain_embed):
        super(HybridAttentionModel, self).__init__()
        self.args = args
        self.embed_size = args.embed_size
        self.lstm_hidden_size = args.lstm_hidden_size
        self.word2idx= word2idx

        self.word_embed = nn.Embedding.from_pretrained(pretrain_embed)
        # batch * max_len * embed_size
        self.lstm = nn.LSTM(self.embed_size, self.lstm_hidden_size, batch_first=True,
                            dropout=self.args.drop_out_lstm,
                            num_layers=self.args.lstm_num_layers,
                            bidirectional=self.args.bidirectional
                            )
        self.user_layer = UserGeneration(self.args)
        self.self_atten_q = SelfAttention(self.args)
        self.self_atten_a = SelfAttention(self.args)
        # answer-question mutual attention share attention trainable weight
        self.hybrid_atten_q_a = HybridAttentionLayer(self.args)
        self.hybrid_atten_q_u = HybridAttentionLayer(self.args)
        self.ratio_layer = RatioLayer()

        self.w_q = nn.Linear(self.args.lstm_hidden_size, self.args.lstm_hidden_size, bias=False)
        self.w_a = nn.Linear(self.args.lstm_hidden_size, self.args.lstm_hidden_size, bias=False)
        self.w_u = nn.Linear(self.args.lstm_hidden_size, self.args.lstm_hidden_size, bias=False)
        self.w_final = nn.Linear(self.args.lstm_hidden_size, self.args.class_kind, bias=True)

        nn.init.xavier_normal_(self.w_q.weight)
        nn.init.xavier_normal_(self.w_a.weight)
        nn.init.xavier_normal_(self.w_u.weight)
        nn.init.xavier_normal_(self.w_final.weight)

    def _init_lstm_hidden(self):
        h_0_size_1 = 1
        if self.args.bidirectional:
            h_0_size_1 *= 2
        h_0_size_1 *= self.args.lstm_num_layers
        hiddena = torch.zeros((h_0_size_1, self.args.batch_size, self.args.lstm_hidden_size),
                              dtype=torch.float, device=self.args.device)
        hiddenb = torch.zeros((h_0_size_1, self.args.batch_size, self.args.lstm_hidden_size),
                              dtype=torch.float, device=self.args.device)
        return hiddena, hiddenb


    def forward(self, question, answer, user, i_flag=None):
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
            q_lstm.view(self.args.batch_size, self.args.max_q_len, bi, self.args.lstm_hidden_size),
            2
        )
        hiddena, hiddenb = self._init_lstm_hidden()
        a_lstm, _ = self.lstm(a_embed, (hiddena, hiddenb))
        #bilstm
        a_lstm = torch.mean(
            a_lstm.view(self.args.batch_size, self.args.max_a_len, bi, self.args.lstm_hidden_size),
            2
        )
        assert a_lstm.size()[0] == self.args.batch_size, "dimention {} != {}".format(a_lstm.size(), self.args.batch_size)
        u_vec = self.user_layer(u_embed)
        #self attention
        q_alpha_atten = self.self_atten_q(q_lstm)
        a_alpha_atten = self.self_atten_a(a_lstm)

        #mutal atten
        q_beta_atten = self.hybrid_atten_q_a(q_lstm, a_lstm, 0)
        a_beta_atten = self.hybrid_atten_q_a(a_lstm, q_lstm, 1)
        u_lambda_atten = self.hybrid_atten_q_u(u_vec, q_lstm, 2)

        #   represent with attention
        q_yi = self.ratio_layer(q_alpha_atten, q_beta_atten)
        a_yi = self.ratio_layer(a_alpha_atten, a_beta_atten)
        u_theta = self.ratio_layer(q_alpha_atten, q_beta_atten, u_lambda_atten)

        q_h_new = (q_yi.unsqueeze(2) * q_lstm).sum(dim=-2)
        a_h_new = (a_yi.unsqueeze(2) * a_lstm).sum(dim=-2)
        u_h_new = (u_theta.unsqueeze(2) * q_lstm).sum(dim=-2)
        h = torch.tanh(self.w_q(q_h_new) + self.w_a(a_h_new) + self.w_u(u_h_new))

        # h_test = torch.tanh(q_lstm.mean(dim=-2) * a_lstm.mean(-2) * u_vec)
        if i_flag is not None:
            logger.histo_summary("q_alpha_atten", q_alpha_atten.cpu().detach().numpy(), i_flag)
            logger.histo_summary("a_beta_atten", a_beta_atten.cpu().detach().numpy(), i_flag)
            logger.histo_summary("q_beta_atten", q_beta_atten.cpu().detach().numpy(), i_flag)
            logger.histo_summary("q_yi", q_yi.cpu().detach().numpy(), i_flag)
            logger.histo_summary("a_yi", a_yi.cpu().detach().numpy(), i_flag)
            logger.histo_summary("u_theta", u_theta.cpu().detach().numpy(), i_flag)
        # # batch * class
        #WARNING: check softmax dimention set
        result = F.log_softmax(self.w_final(h), dim=-1)
        _, predict = result.max(-1)
        return result, predict
