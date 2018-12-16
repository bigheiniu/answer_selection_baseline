from XMLHandler import xmlhandler
from scipy import stats

if __name__ == '__main__':
    path = "/home/bigheiniu/course/ASU_Course/cqa/data/v3.2/"
    content, question_answer_user_label, user_post = xmlhandler.read_xml_data(path)
    content_id2idx, content = xmlhandler.convert_content_id(content)
    user_id2idx, user_context = xmlhandler.convert_user_id(user_post, content_id2idx)

    question_answer_user_label = xmlhandler.convert_question_answer_userId(question_answer_user_label, user_id2idx, content_id2idx)
    with open("test_content.txt",'w') as f1:
        for line in content:
            f1.write(line)
            f1.write("\n")

    with open("test_len.xml",'w') as f1:
        for line in content:
            f1.write(str(len(line)))
            f1.write("\n")

    print(stats.describe([len(line) for line in content]))