''' Handling the data io '''
import argparse
import torch
import attention.Constants as Constants
from XMLHandler import xmlhandler
from TextClean import textClean
import os
import numpy as np

# def read_user_context_from_file(user_file):
#     #######  format ########
#     # content_1, content_2, content_3
#     ############################
#     data = []
#     with open(user_file,'r') as f1:
#         for line in f1.readlines():
#             th = []
#             for t in line.strip().split(","):
#                 try:
#                     th.append(int(t))
#                 except:
#                     i = 1
#             data.append(th)
#     return data
#
#
# def read_question_answer_user_from_file(file):
#     #######  format ############
#     # questionId, answerId, userId
#     ############################
#     data = []
#     with open(file, 'r') as f1:
#         for line in f1.readlines():
#             th = line.strip().rstrip(",").split(",")
#             data.append(th)
#
#     data = np.array(data).astype(int)
#     data = np.array(data).astype(int)
#
#     return data


def shrink_clean_text(content, max_sent_len):
    ''' Convert file into word seq lists and vocab '''

    word_insts = []
    trimmed_sent_count = 0
    i = 0
    j = 0
    for sent in content:
        i += 1
        words = sent.split(" ")
        words = textClean.cleanText(sent)
        if len(words) > max_sent_len:
            trimmed_sent_count += 1
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

def main():
    ''' Main function '''
    file_dir = "./"
    parser = argparse.ArgumentParser()
    # add by yichuan li
    # parser.add_argument('-content_file', default="content.csv")
    # parser.add_argument('-user_file', default="user.csv")
    # parser.add_argument('-q_a_u_file', default="question_answer_user.csv")
    parser.add_argument('-raw_data',default="/home/bigheiniu/course/ASU_Course/cqa/data/v3.2/")



    parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=60)
    parser.add_argument('-min_word_count', type=int, default=5)
    parser.add_argument('-keep_case', action='store_true')

    parser.add_argument('-share_vocab', action='store_true')
    parser.add_argument('-vocab', default=None)

    opt = parser.parse_args()
    content, user_context, question_answer_user_label = xmlhandler.main(opt.raw_data)

    content_word_list = shrink_clean_text(content, opt.max_word_seq_len)

    # train_src_word_insts = read_instances_from_file(
    #     opt.train_src, opt.max_word_seq_len, opt.keep_case)
    # train_tgt_word_insts = read_instances_from_file(
    #     opt.train_tgt, opt.max_word_seq_len, opt.keep_case)



    # Build vocabulary
    word2idx = build_vocab_idx(content_word_list, opt.min_word_count)
    # word to index
    print('[Info] Convert  word instances into sequences of word index.')
    word_id = convert_instance_to_idx_seq(content_word_list, word2idx)

    data = {
        'settings': opt,
        'dict': word2idx,
        'content': word_id,
        'user': user_context,
        'question_anser_user': question_answer_user_label
    }

    opt.save_data="fuck.model"
    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    torch.save(data, opt.save_data)
    print('[Info] Finish.')

if __name__ == '__main__':
    main()
