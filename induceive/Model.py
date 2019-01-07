import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from induceive.Layer import Aggregate, EdgeAttention, LstmMaxPool

class InduceiveModel(nn.Module):
    def __init__(self, args, word_embed, user_count):
        super(InduceiveModel, self).__init__()
        self.args = args
        self.question_aggregate = Aggregate(self.args)
        self.user_aggregate = Aggregate(self.args)
        self.edge_generate = EdgeAttention(self.args)
        self.lstm_maxpool = LstmMaxPool(self.args)

        self.word_embed = nn.Embedding.from_pretrained(word_embed)
        self.user_count = user_count
        self.user_embed = nn.Embedding(user_count, self.args.lstm_hidden_size)

        self.w_a = nn.Linear(self.args.lstm_hidden_size, self.args.lstm_hidden_size)
        self.w_q = nn.Linear(self.args.lstm_hidden_size, self.args.lstm_hidden_size)
        self.w_u = nn.Linear(self.args.lstm_hidden_size, self.args.lstm_hidden_size)
        self.w_final = nn.Linear(self.args.lstm_hidden_size, self.args.label_size)

        nn.init.xavier_normal_(self.w_q.weight)
        nn.init.xavier_normal_(self.w_a.weight)
        nn.init.xavier_normal_(self.w_u.weight)
        nn.init.xavier_normal_(self.w_final.weight)


    def forward(self,
                question1, answer1, user1, score1,
                question_first_layer, question_edge_first_layer, question_edge_score_list_first_layer,
                question_second_layer, question_edge_second_layer, question_edge_score_list_second_layer,

                user_first_layer, user_edge_first_layer, user_edge_score_list_first_layer,
                user_second_layer, user_edge_second_layer, user_edge_score_list_second_layer
                ):

        question1 = self.lstm_maxpool(self.word_embed(question1))
        answer1 = self.lstm_maxpool(self.word_embed(answer1))
        user1 = self.user_embed(user1)



        user_first_layer = self.user_embed(user_first_layer)
        user_edge_first_layer = self.lstm_maxpool(self.word_embed(user_edge_first_layer))

        user_second_layer = self.user_embed(user_second_layer)

        user_edge_second_layer = self.lstm_maxpool(self.word_embed(user_edge_second_layer))

        question_first_layer = self.lstm_maxpool(self.word_embed(question_first_layer))
        question_edge_first_layer = self.lstm_maxpool(self.word_embed(question_edge_first_layer))
        question_second_layer = self.lstm_maxpool(self.word_embed(question_second_layer))
        question_edge_second_layer = self.lstm_maxpool(self.word_embed(question_edge_second_layer))



        # Second Layer combination
        question_edge_second_layer = self.edge_generate(question_edge_second_layer, question_second_layer, user_first_layer.unsqueeze(2))
        user_edge_second_layer = self.edge_generate(user_edge_second_layer, question_first_layer.unsqueeze(2), user_edge_second_layer)

        question_first_layer = self.question_aggregate(user_second_layer, question_edge_second_layer, question_first_layer)
        user_first_layer = self.user_aggregate(question_second_layer, user_edge_second_layer, user_first_layer)

        # First Layer combination
        question_edge_first_layer = self.edge_generate(question_edge_first_layer, question1.unsqueeze(1), user_first_layer)
        user_edge_first_layer = self.edge_generate(user_edge_first_layer, question_first_layer, user1.unsqueeze(1))

        # Zero Layer combination
        question1 = self.question_aggregate(user_first_layer, question_edge_first_layer, question1)
        user1 = self.user_aggregate(question_first_layer, user_edge_first_layer, user1)
        answer1 = self.edge_generate(answer1, question1, user1)
        t = F.tanh(self.w_q(question1) + self.w_a(answer1) + self.w_u(user1))
        if torch.isnan(t).any():
            print("[NANVALUE] tanh value is null")
        result = F.log_softmax(self.w_final(F.tanh(self.w_q(question1) + self.w_a(answer1) + self.w_u(user1))), dim=-1)

        _, predict = result.max(-1)
        return result, predict
