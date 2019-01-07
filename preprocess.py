''' Handling the data io '''
import argparse
import torch
import attention.Constants as Constants
from XMLHandler import xmlhandler
from TextClean import textClean
import numpy as np
import networkx as nx


def shrink_clean_text(content, max_sent_len):
    ''' Convert file into word seq lists and vocab '''

    word_insts = []
    trimmed_sent_count = 0
    i = 0
    j = 0
    for sent in content:
        i += 1
        # words = sent.split(" ")
        words = textClean.cleanText(sent)
        if len(words) > max_sent_len:
            trimmed_sent_count += 1
        elif len(words) < max_sent_len:
            pad_sequence = [Constants.PAD_WORD] * (max_sent_len - len(words))
            words = words + pad_sequence
        word_inst = words[:max_sent_len]

        if word_inst:
            word_insts += [word_inst]
        else:
            j += 1
            word_insts += [Constants.PAD_WORD]

    print('[Info] Get {} instances'.format(len(word_insts)))
    print('[Warning] {} instances is empty'.format(j))

    if trimmed_sent_count > 0:
        print('[Warning] {} instances are trimmed to the max sentence length {}.'
              .format(trimmed_sent_count, max_sent_len))

    return word_insts

def build_vocab_idx(word_insts, min_word_count):
    ''' Trim vocab by number of occurence '''

    full_vocab = set()
    for sent in word_insts:
        for w in sent:
            full_vocab.add(w)
    # full_vocab = set(w for sent in word_insts for w in sent if sent is not None)
    print('[Info] Original Vocabulary size =', len(full_vocab))

    word2idx = {
        # Constants.BOS_WORD: Constants.BOS,
        # Constants.EOS_WORD: Constants.EOS,
        Constants.PAD_WORD: Constants.PAD,
        Constants.UNK_WORD: Constants.UNK}

    word_count = {w: 0 for w in full_vocab}

    for sent in word_insts:
        for word in sent:
            word_count[word] += 1

    ignored_word_count = 0
    for word, count in word_count.items():
        if word not in word2idx:
            if count > min_word_count:
                word2idx[word] = len(word2idx)
            else:
                ignored_word_count += 1
    print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)),
          'each with minimum occurrence = {}'.format(min_word_count))
    print("[Info] Ignored word count = {}".format(ignored_word_count))
    return word2idx

def convert_instance_to_idx_seq(word_insts, word2idx):
    ''' Mapping words to idx sequence. '''
    return [[word2idx.get(w, Constants.UNK) for w in s] for s in word_insts]

def GenerateGraph(question_answer_user_label, train_index, val_index, content_count):
    train_data = question_answer_user_label[train_index]
    val_data = question_answer_user_label[val_index]
    G = nx.Graph()
    user_list = []
    for line in train_data:
        question = line[0]
        answer = line[1]
        user = line[2] + content_count
        user_list.append(user)
        label = line[3]
        G.add_node(question,type=0)
        G.add_node(user, type=1)
        G.add_edge(question, user, a_id=answer, score=label, train_removed=False)
    for line in val_data:
        question = line[0]
        answer = line[1]
        user = line[2] + content_count
        user_list.append(user)
        label = line[3]
        G.add_node(question, type=0)
        G.add_node(user, type=1)
        G.add_edge(question, user, a_id=answer, score=label, train_removed=True)
    print("[INFO] Graph contains {} edge, contentCount is {}".format(len(G.edges()), content_count))
    # print("[INFO] There are {} users, bigggest id is {}".format(len(set(user_list)), max(user_list)))
    return G, max(user_list) - content_count + 1

def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()
    # add by yichuan li
    parser.add_argument('-raw_data',default="data/v3.2/")



    parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=60)
    parser.add_argument('-min_word_count', type=int, default=5)
    parser.add_argument('-keep_case', action='store_true')

    parser.add_argument('-share_vocab', action='store_true')
    parser.add_argument('-vocab', default=None)
    parser.add_argument('-train_size', default=0.6)
    parser.add_argument('-val_size', default=0.4)
    parser.add_argument('-test_size', default=0.0)

    opt = parser.parse_args()
    content, user_context, question_answer_user_label = xmlhandler.main(opt.raw_data)

    content_word_list = shrink_clean_text(content, opt.max_word_seq_len)
    question_answer_user_label = np.array(question_answer_user_label)
    content_count = np.max(np.array(question_answer_user_label)[:,[0,1]]) + 1


    # Build vocabulary
    word2idx = build_vocab_idx(content_word_list, opt.min_word_count)
    # word to index
    print('[Info] Convert  word instances into sequences of word index.')
    word_id = convert_instance_to_idx_seq(content_word_list, word2idx)

    #split train-valid-test dataset

    index = np.arange(len(question_answer_user_label))
    np.random.shuffle(index)
    length = len(question_answer_user_label)
    train_end = int(opt.train_size * length)
    val_end = int(opt.val_size * length) + train_end
    train_index = index[:train_end]
    val_index = index[train_end: val_end]
    test_index = index[val_end:]

    G, user_count = GenerateGraph(question_answer_user_label, train_index, val_index, content_count)



    data = {
        'settings': opt,
        'dict': word2idx,
        'content': word_id,
        'user': user_context,
        'question_answer_user_train': question_answer_user_label[train_index],
        'question_answer_user_val': question_answer_user_label[val_index],
        'question_answer_user_test': question_answer_user_label[test_index],
        'G': G,
        'user_count': user_count
    }

    opt.save_data="data/store.torchpickle"
    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    torch.save(data, opt.save_data)
    print('[Info] Finish.')





if __name__ == '__main__':
    main()
