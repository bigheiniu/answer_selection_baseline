"""This module provides preprocessing routines for SemEval Task 3, Subtask B datasets."""

from lxml import etree

def parse_post(xmlfile):
    content_dic = {}
    title_dic = {}
    user_context = {}
    accept_answer_dic = {}
    tree = etree.parse(xmlfile)
    quesition_answer_user = {}

    for row in tree.iterfind("row"):
        post_id = int(row.attrib['Id'])
        if post_id in content_dic:
            continue
        post_type = row.attrib['PostTypeId']
        if post_type == '1':
            # question
            content_dic[post_id] = row.attrib['Body']
            title_dic[post_id] = row.attrib['Title']
            try:
                accept_answer_dic[post_id] = int(row.attrib['AcceptedAnswerId'])
            except:
                continue

        elif post_type == '2':
            #Answer
            try:
                user_id = int(row.attrib['OwnerUserId'])
            except:
                continue
            if user_id < 0:
                continue
            #ATTENTION: only store answer as user history data
            if user_id not in user_context:
                user_context[user_id] = [post_id]
            else:
                user_context[user_id].append(post_id)


            content_dic[post_id] = row.attrib["Body"]
            try:
                if "ParentID" in row.attrib:
                    question_id = int(row.attrib['ParentID'])
                else:
                    question_id = int(row.attrib['ParentId'])
            except:
                print("[ERROR] ParentId could not find in {}".format(row.attrib))
                exit(-1)
            line = [question_id, post_id, user_id]
            quesition_answer_user[post_id] = line

    return quesition_answer_user, content_dic, title_dic, accept_answer_dic, user_context

def parse_vote(xmlfile):
    vote_dic = {}
    tree = etree.parse(xmlfile)
    for row in tree.iterfind('row'):
        post_id = int(row.attrib['PostId'])
        vote_type_id = row.attrib['VoteTypeId']
        if vote_type_id == '2':
            if post_id in vote_dic:
                vote_dic[post_id] += 1
            else:
                vote_dic[post_id] = 1
    return vote_dic









