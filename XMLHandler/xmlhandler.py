"""This module produces SemEval Task 3, Subtask B datasets in JSON."""

from itertools import chain
from json import dump

from XMLHandler.XMLpreprocessing import parse
from XMLHandler.xmlfiles import XMLFiles
import os
import numpy as np



def convert_content_id(data):
    #convert answer, question, user "STRING" id to int id
    content_id2idx = {}
    index = 0
    content = []
    for file_content in data:
        for id, text in file_content.items():
            if id in content_id2idx:
                old_text = content[content_id2idx[id]]
                if len(text) < len(old_text):
                    content[content_id2idx[id]] = text
                    # difference because text is dirty
                    # print("[WARNING] content id duplicate")
            else:
                content_id2idx[id] = index
                index += 1
                content.append(text)
                assert len(content) == index, "[ERROR] content length {} != {} index".format(len(content), index)
    return content_id2idx, content


def convert_user_id(user_context_files, content_id2idx):
    '''
    convert userid from str into int
    :param user_context:
    :param content_id2idx:
    :return:
    '''
    user_id2idx = {}
    user_context_all = []
    index = 0
    for file_user in user_context_files:
        for user_id, user_context in file_user.items():
            if user_id not in user_id2idx:
                user_id2idx[user_id] = index
                index += 1
                assert len(user_id2idx) == index,"[ERROR] user length not equal"
            try:
                user_context = [content_id2idx[context_id] for context_id in user_context]
            except:
                print("[ERROR] content not in id2idx")
                exit()

            u_loc = user_id2idx[user_id]

            if(len(user_context_all) <= u_loc):
                user_context_all.append(user_context)
                assert len(user_context_all) == u_loc + 1, "[ERROR] user context "
            else:
                user_context_all[u_loc] += user_context

    return user_id2idx, user_context_all

def convert_question_answer_userId(data, user_id2idx, content_id2idx):
    #return: q_idx, a_idx, u_idx, label
    # idx means int index
    question_answer_user = []
    for file_data in data:
        for pair in file_data:
            #q_id, a_id, u_id, label
            assert len(pair) == 4, "[ERROR] question_answer_user length is not 4"
            q_idx = content_id2idx[pair[0]]
            a_idx = content_id2idx[pair[1]]
            u_id = user_id2idx[pair[2]]
            label = pair[3]
            question_answer_user.append([q_idx, a_idx, u_id, label])
    return question_answer_user







def read_xml_data(path):
    # hanle all the data under v3.2
    # for easy handle, we will read all the data and then random split data into "train, val, test"
    sub_dirs = os.listdir(path)
    sub_dirs = [ os.path.join(path,dir) for dir in sub_dirs if os.path.isdir(os.path.join(path,dir))]
    # sub_dirs = [os.path.join(path, dir) for dir in sub_dirs ]
    content = []
    question_answer_user_label = []
    user_post = []
    for sub_dir in sub_dirs:
        print(sub_dir)
        for file in os.listdir(sub_dir):
            if "xml" not in file or "subtaskA" not in file:
                continue
            file = os.path.join(sub_dir, file)
            content_file, question_answer_user_label_file, user_post_file = parse(file)
            content.append(content_file)
            question_answer_user_label.append(question_answer_user_label_file)
            user_post.append(user_post_file)

    return content, question_answer_user_label, user_post

def main(path):
    content, question_answer_user_label, user_post = read_xml_data(path)
    content_id2idx, content = convert_content_id(content)
    user_id2idx, user_context = convert_user_id(user_post, content_id2idx)
    question_answer_user_label = convert_question_answer_userId(question_answer_user_label, user_id2idx, content_id2idx)

    return content, user_context, question_answer_user_label