import torch
import torch.nn.functional as F
import torch.nn as nn
from Utils import LSTM


'''
Community-Based Question Answering via Asymmetric Multi-Faceted Ranking Network Learning
'''
class AMRNL(nn.Module):
    def __init__(self, args,
                 user_count,
                 word2vec,
                 ContentEmbed,
                 user_adjance
                 ):
        super(AMRNL, self).__init__()
        self.lstm = LSTM(args)
        self.args = args
        self.user_count = user_count
        self.user_embed = nn.Embedding(self.user_count, args.lstm_hidden_size)
        self.content_matrix = ContentEmbed
        self.word2vec = nn.Embedding.from_pretrained(word2vec)
        #already normalized
        self.user_adjance_embed = nn.Embedding.from_pretrained(user_adjance)

        self.smantic_meache_bilinear = nn.Bilinear(args.lstm_hidden_size, args.lstm_hidden_size, args.lstm_hidden_size)


    def forward(self,
                question_list,
                answer_list,
                user_list,
                score_list
                ):


        question_embed_feature = self.word2vec(self.content_matrix.content_embed(question_list))
        answer_embed_feature = self.word2vec(self.content_matrix.content_embed(answer_list))

        user_embed_feature = self.user_embed(user_list)
        user_neighbor = self.user_adjance_embed(user_list)

        question_lstm = self.lstm(question_embed_feature)
        answer_lstm = self.lstm(answer_embed_feature)

        match_score = self.smantic_meache_bilinear(question_lstm, answer_lstm)
        relevance_score = F.cosine_similarity(question_lstm, user_embed_feature, dim=-1)
        relevance_score.unsqueeze_(-1)
        result = match_score * relevance_score

        #l2 norm
        regular = F.normalize(user_embed_feature - torch.matmul(user_neighbor, self.user_embed.weight), 2, dim=-1)
        return result, regular






