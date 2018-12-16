import torch
import numpy as np
import gensim
def loadEmbed(file, embed_size, vocab_size, word2idx=None, Debug=True):
    # read pretrained word2vec, convert to floattensor
    if(Debug):
        embed = np.random.rand(vocab_size, embed_size)
        return torch.FloatTensor(embed)

    #load pretrained model
    else:
        print(" [INFO] load pre-trained word2vec embedding")
        model = gensim.models.KeyedVectors.load_word2vec_format(file,
                                                                binary=True)
        embed_matrix = np.zeros(vocab_size+1, embed_size)
        for word, i in word2idx.items():
            if word in model.vocab:
                embed_matrix[i] = model.word_vec(word)

        weights = torch.FloatTensor(embed_matrix)
        return weights
