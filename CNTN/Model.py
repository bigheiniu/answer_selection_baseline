import torch
import torch.nn as nn
import torch.nn.functional as F
from attention.Utils import loadEmbed


class CNTN(nn.Module):
    def __init__(self, args, word2idx, pretrained_embed):
        super(CNTN, self).__init__()
        self.word2idx = word2idx
        self.args = args
        self.embedding = nn.Embedding.from_pretrained(pretrained_embed)
        self.cntn = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=self.args.in_channels[i],
                          out_channels=self.args.out_channgels[i],
                          kernel_size=self.args.cnn_kernel_size[i]),
                nn.BatchNorm2d(self.args.out_channgels[i]),
                nn.ReLU(),
                nn.MaxPool2d(self.pool_kernel_size[i])
            )
            for i in range(self.args.layer)
        ])
        self.bilinear_M = nn.Bilinear(self.args.in_features, self.args.int_features, self.out_features)
        self.linear_V = nn.Linear(2 * self.int_features, self.out_features, bias=False)
        self.linear_U = nn.Linear(self.out_features, 1, bias=False)

        nn.init.xavier_normal_(self.bilinear_M.weight)
        nn.init.xavier_normal_(self.bilinear_M.bias)
        nn.init.xavier_normal_(self.linear_V.weight)
        nn.init.xavier_normal_(self.linear_U.weight)

    def forward(self, question, good_answer_list, bad_answer_list, label):
        question_embed = self.embedding(question)
        question_cntn = self.cntn(question_embed)
        good_answer_embed = self.embedding(good_answer_list)
        good_answer_cntn = self.cntn(good_answer_embed)
        good_q_m_a = self.bilinear_M(question_cntn, good_answer_cntn)
        good_q_m_a = good_q_m_a + self.linear_V(nn.cat((question_cntn, good_answer_cntn)))
        good_score = self.linear_U(good_q_m_a)

        bad_answer_embed = self.embedding(bad_answer_list)
        bad_answer_cntn = self.cntn(bad_answer_embed)
        bad_q_m_a = self.bilinear_M(question_cntn, bad_answer_cntn)
        bad_score = self.linear_U(bad_q_m_a)

        good_score = good_score.unsqueeze(-2).expand(-1, bad_score.shape[-2], -1)
        score = bad_score - good_score
        score = score + self.args.gamma
        loss = torch.where(score > 0, score, 0).sum()
        result = torch.cat((good_score, bad_score))
        return loss, result




