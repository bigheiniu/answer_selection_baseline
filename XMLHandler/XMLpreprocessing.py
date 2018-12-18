"""This module provides preprocessing routines for SemEval Task 3, Subtask B datasets."""

from lxml import etree
import re

def parse(xmlfile):
    tree = etree.parse(xmlfile)
    good = 1
    bad = 0
    content = {}
    user = {}
    question_answer_user_label = []
    # for orgquestion in tree.findall("Thread"):
    #     if orgquestion.attrib["ORGQ_ID"] in seen_orgquestion_ids:
    #         continue
    #     seen_orgquestion_ids.update([orgquestion.attrib["ORGQ_ID"]])

    for thread in tree.findall("Thread"):
        q_Id = thread.find("RelQuestion").attrib["RELQ_ID"]
        q_user_Id = thread.find("RelQuestion").attrib["RELQ_USERID"]

        # Store question content
        # TODO: handle non text
        if thread.find("RelQuestion/RelQBody") is None:
            continue

        if thread.find("RelQuestion/RelQClean") is not None:
            q_content = thread.find("RelQuestion/RelQClean").text
        else:
            q_content = thread.find("RelQuestion/RelQBody").text
        # question content is empty
        if q_content is None or len(q_content) == 0:
            continue

        content[q_Id] = q_content
        if q_user_Id in user:
            user[q_user_Id].add(q_Id)
        else:
            user[q_user_Id] = set([q_Id])

        for relcomment in (thread.findall("RelComment")):

            a_Id = relcomment.attrib["RELC_ID"]
            test_q_id = a_Id.split("_")
            if len(test_q_id) == 2:
                test_q_id = test_q_id[0]
            else:
                test_q_id = test_q_id[0] + "_" + test_q_id[1]
            assert q_Id == test_q_id, "[ERROR] question id {} conliction {} in {}".format(q_Id, test_q_id, xmlfile)
            a_user_Id = relcomment.attrib["RELC_USERID"]
            label = good if relcomment.attrib["RELC_RELEVANCE2RELQ"] == 'Good' else bad

            if relcomment.find("RelCClean") is not None:
                answer_content = relcomment.find("RelCClean").text
            elif relcomment.find("RelCText") is not None:
                answer_content = relcomment.find("RelCText").text
            else:
                try:
                    answer_content = relcomment.find("RelCBody").text
                except:
                    print("{} {} {} {}".format(q_Id, a_Id, a_user_Id, xmlfile))
                    exit()

            if answer_content is None or len(answer_content) == 0:
                continue

            # user here is answerer
            question_answer_user_label.append([q_Id, a_Id, a_user_Id, label])
            content[a_Id] = answer_content
            if a_user_Id in user:
                user[a_user_Id].add(a_Id)
            else:
                user[a_user_Id] = set([a_Id])


    assert len(content) > 0, "[ERROR] content data in {} is empty".format(xmlfile)
    return content, question_answer_user_label, user









