import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer


stemmer = SnowballStemmer("english")

def cleanText(text):
    #TODO: consider each pragraph as an sentence or heriarchy lstm
    #TODO: remove most frquent/unfrequent word
    #convert
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"ur", "your ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " egcontent_list.pickle ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)



    #remove alphabet is not a-z, A-Z
    # text = re.sub(r"[^a-zA-Z]",' ', text)

    #lower case
    text = text.lower()

    #drop stop words
    word_tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in word_tokens if not w in stop_words]

    #stem words
    #loving -> love
    stem_words = [stemmer.stem(word) for word in filtered_sentence]


    return stem_words

#
# def content_clean(content_list):
#     # 1. change some word
#     # 2. remove hashtag and other comma
#     # 3. remove stopwords
#     # 4. lemma
#
#     content_clean_word_list = [cleanText(setence) for setence in content_list]
#     return content_clean_word_list