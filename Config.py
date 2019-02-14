import torch
class config_data_preprocess:

    #max length of a post
    max_len = 60

    #remove word that frequecy less than
    min_word_count = 5

    #split data into train and test data
    train_size = 0.6
    test_size = 0.4

    # location store the raw data
    raw_data = "/home/yichuan/course/data"

    # store preprocessed data
    save_data = "data/store_stackoverflow.torchpickle"


class config_model:
    #in Debug mode or not
    DEBUG = True

    #basic setting
    epoch = 60
    log = None
    batch_size = 60
    model_name = ""
    cuda = False
    device = torch.device('cuda' if cuda else 'cpu' )
    #path to store data
    data = "data/store_stackoverflow.torchpickle"

    #=====================
    #content_data setting
    #====================

    #max length of question
    max_q_len = 60
    #max length of answer
    max_a_len = 60
    #max length of user context
    max_u_len = 200



    # learning rate
    lr = 0.001

    #======================
    #word embedding setting
    #======================
    embed_fileName="data/glove/glove.6B.100d.txt"
    #vocabulary size
    vocab_size = 30000
    #word to vector embed size
    embed_size = 100

    #================
    #LSTM setting
    #================
    lstm_hidden_size = 128
    lstm_num_layers = 1
    drop_out_lstm = 0.3
    bidirectional = False

    #================
    #convolutional setting
    #================
    cnn_layers = 3
    cnn_kernel_size = [(5,5),(3,3),(4,4)]
    in_channels = [3,4,5]
    out_channels = [4,5,6]




    #============
    #evaluate settings
    #============
    # rank or classification
    is_classification = False

    #diversity setting
    div_topK = 1
    dpp_early_stop = 0.0001

    #coverage test model setting
    lda_topic = 20
    #whether the coverage test model is already trained or not
    cov_pretrain = True
    # location to store or load model
    cov_model_path = "result"

    #Rank evaluate setting
    ndcg_k = 2

    #hinge loss margin
    margin = 0.1