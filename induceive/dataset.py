import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
'''
    G is already clean, no isolated nodes
    Question_id,  Answer_id can derectly get content
'''

class Induceive_dataset(data.Dataset):
    '''
        pair comparsion; user random embed
        output: batch_question_node_content, batch_answer_edge_content, batch_user_node_id, score/label,

                question_first_layer_neighbor, question_second_layer_neighbor,

                question_first_layer_edge, question_first_layer_edge;
                -------
                user_first_layer_neighbor, user_first_layer_neighbor,

                user_first_layer_edge, user_first_layer_edge
    '''
    def __init__(self, G,  args, content, training=True):
        super(Induceive_dataset, self).__init__()
        self.G = G
        self.args = args
        self.content = content
        self.train = training
        self.adj, self.adj_answer, self.adj_score, self.adj_degree = self.adjance()

        if(training):
            self.edges = self.train_edge()
        else:
            self.edges = self.val_edge()






    def adjance(self):
        adj = {}
        adj_answer = {}
        adj_score = {}
        adj_degree = {}
        for node in self.G.nodes():
            neighbors = np.array([neighbor for neighbor in self.G.neighbors[node]])
            adj_degree[node] = len(neighbors)

            if len(neighbors) == 0:
                continue
            if len(neighbors) < self.args.max_degree:
                neighbors = np.random.choice(neighbors, self.args.max_degree, replace=True)
            else:
                neighbors = np.random.choice(neighbors, self.args.max_degree, replace=False)

            answer = []
            score = []
            for neigh_node in neighbors:
                answer.append(self.G[node][neigh_node]['a_id'])
                score.append(self.G[node][neigh_node]['score'])

            adj[node] = neighbors

        return adj, adj_answer, adj_score, adj_degree





    def train_edge(self):
        return [e for e in self.G.edges(data=True) if not self.G[e[0]][e[1]]['train_removed']]

    def val_edge(self):
        return [e for e in self.G.edges(data=True) if self.G[e[0]][e[1]]['train_removed']]

    def contentEmbed(self, content_id):
        content_id = np.array(content_id)
        shape = list(content_id.shape).append(self.args.content_len)

        content_id = content_id.reshape(-1,1)
        return np.array([self.content[id] for id in content_id]).reshape(shape)

    # def userid2idx_function(self, user_id_array):
    #     shape = user_id_array.shape
    #     user_id_array = user_id_array.reshape(-1,1)
    #     return np.array([self.userid2idx[id]for id in user_id_array]).reshape(shape)

    def neigh_sample(self, id_array, count):
        if isinstance(id_array, list):
            id_array = np.array(id_array)
        shape = list(id_array.shape)
        id_array = id_array.reshape(-1,)
        count = count if count < self.args.max_degree else self.args.max_degree
        shape.append(count)

        neighbors = np.array([self.adj[id][:count] for id in id_array]).reshape(shape)
        answer_edge = np.array([self.adj_answer[id][:count] for id in id_array]).reshape(shape)
        score = np.array([self.adj_score[id][:count] for id in id_array]).reshape(shape)
        return neighbors, answer_edge, score

    def __len__(self):
        return len(self.edges)


    def __getitem__(self, idx):
        edge = self.edges[idx]
        question = edge[0] if self.G.node[edge[0]]['flag'] == 0 else edge[1]
        user = edge[0] if self.G.node[edge[0]]['flag'] == 1 else edge[1]
        answer = self.G[edge[0]][edge[1]]['a_id']
        score = self.G[edge[0]][edge[1]]['score']

        question_edge = []
        question_edge_score_list = []
        user_edge = []
        user_edge_score_list = []

        for layer in range(self.args.sample_layer):
            question_neighbor, question_edge_answer, question_edge_score = \
            self.neigh_sample(question[-1], self.args.sample_count[layer])

            user_neighbor, user_edge_answer, user_edge_score = \
            self.neigh_sample(user[-1], self.args.sample_count[layer])

            question.append(user_neighbor)
            user.append(question_neighbor)

            question_edge.append(question_edge_answer)
            user_edge.append(user_edge_answer)

            question_edge_score_list.append(question_edge_score)
            user_edge_score_list.append(user_edge_score)

        question = list(map(self.contentEmbed, question))
        question_edge = list(map(self.contentEmbed, question_edge))
        answer = self.contentEmbed([answer])
        user_edge = list(map(self.contentEmbed, user_edge))
        #only two layers
        return question[0], answer, user[0], score, \
               question[1], question_edge[0], question_edge_score_list[0], \
               question[2], question_edge[1], question_edge_score_list[1], \
                user[1], user_edge[0], user_edge_score_list[0], \
                user[2], user_edge[1], user_edge_score_list[1]


