import numpy as np
import torch
import torch.utils.data

from attention import Constants
import itertools



def paired_collate_fn(insts):
    question_content, answer_content, user_contex_list, label = list(zip(*insts))
    question_content = collate_fn(question_content)
    answer_content = collate_fn(answer_content)
    user_contex_list = collate_fn(user_contex_list,False)
    label = collate_fn(label, label=True)
    return question_content, answer_content, user_contex_list, label

def collate_fn(insts, label=False, not_user=True):
    ''' Pad the instance to the max seq length in batch '''
    if (label):
        return torch.LongTensor(insts)
    max_len = max(len(inst) for inst in insts)
    if (not_user):
        batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    # batch_pos = np.array([
    #     [pos_i+1 if w_i != Constants.PAD else 0
    #      for pos_i, w_i in enumerate(inst)] for inst in batch_seq])

    batch_seq = torch.LongTensor(batch_seq)
    # batch_pos = torch.LongTensor(batch_pos)

    return batch_seq

class QuestionAnswer_CNTN(torch.utils.data.Dataset):
    def __init__(self, question_answer_user, word2idx):
        idx2word = {idx: word for word, idx in word2idx.items()}
        self._word2idx = word2idx
        self._idx2word = idx2word
        self.question_answer = self._item_pair(
            self._sorted_question([item[0:2]+item[-1] for item in question_answer_user])
        )


    def _sorted_question(self, question_answer):
        return sorted(question_answer, key=lambda x: x[0])[0]

    def _item_pair(self, sorted_question_answer):
        #question_id:[[answer_ids], [labels]]
        index = []
        dic = {}
        for item in sorted_question_answer:
            if item[0] not in dic and len(dic) != 0:
                index.append(index)
                dic = {}
                dic[item[0]] = [[item[1]],[item[-1]]]
            else:
                dic[item[0]][0].append(item[1])
                #add label
                dic[item[0]][1].append(item[-1])
        index.append(dic)
        return index

    def __len__(self):
        return len(self.question_answer)
    def __getitem__(self, idx):
        question_answer_dic = self.question_answer[idx]
        question = list(question_answer_dic.keys())[0]
        answer = np.array(question_answer_dic[question][0])
        labels = np.array(question_answer_dic[question_answer_dic][1])
        good_answer_list = answer[labels == 1]
        bad_answer_list = answer[labels == 0]
        return question, good_answer_list, bad_answer_list, labels




class QuestionAnswerUser(torch.utils.data.Dataset):
    def __init__(
        self, word2idx, word_insts,
            user, question_answer_user, max_u_len, transformer=None):

        idx2word = {idx:word for word, idx in word2idx.items()}
        self._word2idx = word2idx
        self._idx2word = idx2word
        self._world_insts = word_insts
        self._user = user
        self._max_u_len = max_u_len
        self._question_answer_user = question_answer_user
        self._transformer = transformer
        self.weights = self.weights_function()
        self.data = self.balanced_data()

    @property
    def n_insts(self):
        ''' Property for dataset size '''
        return len(self._question_answer_user)

    @property
    def vocab_size(self):
        ''' Property for vocab size '''
        return len(self._word2idx)

    @property
    def word2idx(self):
        ''' Property for word dictionary '''
        return self._word2idx

    @property
    def idx2word(self):
        ''' Property for index dictionary '''
        return self._idx2word

    def _get_label(self, dataset, idx):
        return dataset[idx][3]

    def weights_function(self):
        label_to_count = {}
        for idx in list(range(self.n_insts)):
            label = self._get_label(self._question_answer_user, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(self._question_answer_user, idx)]
                   for idx in list(range(self.n_insts))]
        weights = torch.DoubleTensor(weights)
        return weights

    def balanced_data(self):
        return [self._question_answer_user[i] for i in torch.multinomial(
                self.weights, self.n_insts, replacement=True)]


    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        # add a transformer => convert numpy into tensor
        question_answer_user = self.data[idx]
        label = question_answer_user[3]
        user_id = question_answer_user[2]
        answer_id = question_answer_user[1]
        question_id = question_answer_user[0]
        user_contex_list = [self._world_insts[i] for i in self._user[user_id]]
        user_contex_list = list(itertools.chain(*user_contex_list))
        user_contex_list = user_contex_list[:self._max_u_len]
        answer_content = self._world_insts[answer_id]
        question_content = self._world_insts[question_id]
        if self._transformer is not None:
            question_content = self._transformer(question_content)
            answer_content = self._transformer(answer_content)
            user_contex_list = self._transformer(user_contex_list)
            label = self._transformer(label)


        return question_content, answer_content, user_contex_list, label
