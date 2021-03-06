import torch
import torch.nn as nn
import torch.nn.functional as F
from Model import CNTN
from Config import config_model
from Utils import ContentEmbed


if __name__ == '__main__':
    device = torch.device('cuda' if config_model.cuda else 'cpu')
    word2vec = torch.randn(config_model.vocab_size, config_model.embed_size, device=device)
    content_length = 100
    content = torch.randint(low=0, high=config_model.vocab_size,size=(content_length,  10 * config_model.max_a_len), dtype=torch.int64, device=device)

    content_embed = ContentEmbed(content)
    model = CNTN(config_model, word2vec, content_embed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config_model.lr)


    for i in range(100):
        print(i)
        question = torch.randint(high=content_length, size=(config_model.batch_size,), dtype=torch.int64, device=device)
        good_answer = torch.randint(high=content_length, size=(config_model.batch_size,), dtype=torch.int64, device=device)
        good_score = model(question, good_answer)

        bad_answer = torch.randint(high=content_length, size=(config_model.batch_size,), dtype=torch.int64, device=device)

        bad_score = model(question, bad_answer)
        loss = torch.sum(F.relu(config_model.margin - good_score + bad_score))
        loss.backward()
        optimizer.step()

