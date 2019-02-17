import torch
import torch.nn as nn
from MutihopAttention.Layer import *
from Utils import ContentEmbed


class MultihopAttention(nn.Module):
    def __init__(self, args, word2vec_pretrained, content_embed:ContentEmbed):
        super(MultihopAttention, self).__init__()
        self.args = args
        self.embed_size = args.embed_size
        self.lstm_hidden_size = args.lstm_hidden_size
        self.content_embed = content_embed

        self.word_embed = nn.Embedding.from_pretrained(word2vec_pretrained)
        # batch * max_len * embed_size
        self.lstm = LSTM(self.args)
        # answer-question mutual attention share attention trainable weight
        self.attention_layers = self.args.attention_layers

        self.question_model = QuestionAttention(args)


        #answer selection
        # self.answer_model_bilinear = BilinearAttention(args)
        # self.asnwer_model_self = SelfAttention(args)
        # self.answer_model_mlp = MLPAttention(args)
        self.answer_model_sequential = SequentialAttention(args)

        self.cosine = MultiCosimilarity()

    def forward(self, question, answer):
        question_embed = self.word_embed(self.content_embed.content_embed(question))
        answer_embed = self.word_embed(self.content_embed.content_embed(answer))

        lstm_list_question = self.lstm(question_embed)
        lstm_list_answer = self.lstm(answer_embed)

        score = 0
        m_q = torch.mean(lstm_list_question, dim=-2)
        for i in range(self.attention_layers):
            o_q, m_q = self.question_model(lstm_list_question, m_q)
            # o_a_bilinear = self.answer_model_bilinear(o_q, lstm_list_answer)
            # o_a_self = self.asnwer_model_self(o_q, lstm_list_answer)
            # o_a_mlp = self.answer_model_mlp(o_q, lstm_list_answer)

            o_a_sequential = self.answer_model_sequential(o_q, lstm_list_answer)
            score += F.cosine_similarity(o_q.squeeze(1), o_a_sequential)
        score = score / self.attention_layers
        return score


