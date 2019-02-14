from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel

import numpy as np
from gensim.models import LdaModel
from gensim.matutils import cossim
from gensim.test.utils import datapath

from joblib import dump, load
import os
import itertools




# class CoverMetricClass:
#     def __init__(self, backgroudn_data, load_pretrain, model_path, lda_topic):
#         self.tf_idf = TFIDFSimilar(backgroudn_data, load_pretrain, model_path)
#         self.lda = LDAsimilarity(backgroudn_data, load_pretrain, model_path, lda_topic)
#
#     def similarity(self, content, highRank):
#         return self.tf_idf.simiarity(content, highRank), self.lda.similarity(content, highRank)





class TFIDFSimilar:
    def __init__(self, background_data, load_pretrain, model_path ):
        file_name = "tfidf.model"
        model_path = os.path.join(model_path, file_name)
        if load_pretrain:
            self.tfModel = self.loadModel(model_path)
            self.n_features = (1, self.tfModel.idf_.shape[0])
        else:
            bc_vec = self.get_idf(background_data)
            self.tfModel = TfidfTransformer()
            self.tfModel.fit(X=bc_vec)
            self.n_features = (1, self.tfModel.idf_.shape[0])
            self.saveModel(model_path)

    def get_idf(self, background_data):
        item, count = np.unique(list(itertools.chain.from_iterable(background_data)), return_counts=True)
        count = count.reshape(len(count), 1)
        item = item.reshape(len(item), 1)
        base = np.zeros((1, np.max(item) + 1))
        for i, c  in zip(item, count):
            base[0,i] = c
        return base

    def simiarity(self, content, highRank):
        content_vec = np.zeros(self.n_features)
        highRank_vec = np.zeros(self.n_features)
        for index in content:
            content_vec[0][index] += 1
        for index in highRank:
            highRank_vec[0][index] += 1
        i = self.tfModel.transform(content_vec)
        j = self.tfModel.transform(highRank_vec)
        cosine_similarities = linear_kernel(i,j).flatten()
        return cosine_similarities

    def saveModel(self,path):
        dump(self.tfModel, path)

    def loadModel(self,path):
       return load(path)





class LDAsimilarity:
    def __init__(self, background_data, topic_count, load_pretrain, model_path):
        file_name = "lda.model"

        model_path = os.path.join(os.path.abspath(model_path), file_name)
        if load_pretrain:
            self.lda = self.loadModel(model_path)
        else:
            corpus = [self.list2tuple(line) for line in background_data]
            self.lda = LdaModel(corpus, num_topics=topic_count)
            self.saveModel(model_path)

    def list2tuple(self, data_list):
        data_list = np.array(data_list)
        y = np.bincount(data_list)
        ii = np.nonzero(y)[0]
        return list(zip(ii, y[ii]))

    def saveModel(self, path):
        temp_file = datapath(path)
        self.lda.save(temp_file)

    def loadModel(self, path):
        temp_file = datapath(path)
        lda = LdaModel.load(temp_file)
        return lda

    def similarity(self, content, highrank):
        content_corpus = self.list2tuple(content)
        highrank_corpus = self.list2tuple(highrank)
        lda_content_vec = self.lda[content_corpus]
        highrank_content_vec = self.lda[highrank_corpus]
        return cossim(lda_content_vec, highrank_content_vec)


if __name__ == '__main__':
    model_path = '/home/yichuan/course/induceiveAnswer/data/th.pi'
    back_ground_data = np.random.randint(0, 1000, (100000, 5))
    lda = LDAsimilarity(back_ground_data, 10, True, model_path)
    content = np.random.randint(0, 100, (2000,))
    highRank = np.random.randint(0, 100, (1000,))
    print("score is {}".format(lda.similarity(content, highRank)))












