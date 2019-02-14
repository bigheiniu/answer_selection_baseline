import torch
import torch.nn.functional as F
from Config import config_model
from Model import AMRNL
from Utils import ContentEmbed

if __name__ == '__main__':
    user_count = 100
    device = torch.device('cuda' if config_model.cuda else 'cpu')
    word2vec = torch.randn(config_model.vocab_size, config_model.embed_size, device=device)
    user_adjance = torch.randn(user_count, user_count, device=device)
    content_length = 100
    content = torch.randint(low=0, high=config_model.vocab_size,size=(content_length, config_model.max_a_len), dtype=torch.int64, device=device)

    content_embed = ContentEmbed(content)
    model = AMRNL(config_model, user_count, word2vec, content_embed, user_adjance)
    optimizer = torch.optim.Adam(model.parameters(), lr=config_model.lr)


    for i in range(100):
        print(i)
        question = torch.randint(high=content_length, size=(config_model.batch_size,), dtype=torch.int64, device=device)
        good_answer = torch.randint(high=content_length, size=(config_model.batch_size,), dtype=torch.int64, device=device)
        good_user = torch.randint(high=user_count, size=(config_model.batch_size,), dtype=torch.int64, device=device)
        score = torch.randint(high=user_count, size=(config_model.batch_size,), dtype=torch.int64, device=device)
        good_score, good_regular = model(question, good_answer, good_user, score)

        bad_answer = torch.randint(high=content_length, size=(config_model.batch_size,), dtype=torch.int64, device=device)
        bad_user = torch.randint(high=user_count, size=(config_model.batch_size,), dtype=torch.int64, device=device)

        bad_score, bad_regular = model(question, bad_answer, bad_user, score)
        loss = torch.sum(F.relu(config_model.margin - good_score + bad_score)) + torch.sum(good_regular + bad_regular)
        loss.backward()
        optimizer.step()







