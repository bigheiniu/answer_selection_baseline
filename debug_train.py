import argparse
import math
import time
from tqdm import tqdm
from tqdm import trange
#pytorch import
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention.Model import HybridAttentionModel
import numpy as np
from dataset import QuestionAnswerUser, paired_collate_fn
from attention.Utils import Accuracy
#grid search for paramter
from sklearn.grid_search import ParameterGrid


def prepare_dataloaders(data, opt):
    # ========= Preparing DataLoader =========#

    train_loader = torch.utils.data.DataLoader(
        QuestionAnswerUser(
            word2idx=data['dict'],
            word_insts=data['content'],
            user=data['user'],
            question_anser_user=data['question_anser_user_train'],
            max_u_len=opt.max_u_len
        ),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        QuestionAnswerUser(
            word2idx=data['dict'],
            word_insts=data['content'],
            user=data['user'],
            question_anser_user=data['question_anser_user_val'],
            max_u_len=opt.max_u_len
        ),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        QuestionAnswerUser(
            word2idx=data['dict'],
            word_insts=data['content'],
            user=data['user'],
            question_anser_user=data['question_anser_user_test'],
            max_u_len=opt.max_u_len
        ),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)
    return train_loader, val_loader, test_loader



def train_epoch(model, data, optimizer, args):
    model.train()
    loss_fn = nn.NLLLoss()
    for batch in tqdm(
        data, mininterval=2, desc=' --(training)--',leave=True
    ):

        q_iter, a_iter, u_iter, gt_iter = map(lambda x: x.to(args.device), batch)
        args.max_q_len = q_iter.shape[1]
        args.max_a_len = a_iter.shape[1]
        args.batch_size = q_iter.shape[0]
        optimizer.zero_grad()
        result = model(q_iter, a_iter, u_iter, gt_iter)
        loss = loss_fn(result, gt_iter)
        loss.backward()
        optimizer.step()


def eval_epoch(model, data, args):
    model.eval()
    pred_all = []
    label_all = []
    loss_fn = nn.NLLLoss()
    loss = 0
    with torch.no_grad():
        for batch in tqdm(
            data, mininterval=2, desc="  ----(validation)----  ", leave=True
        ):
            q_val, a_val, u_val, gt_val = map(lambda x: x.to(args.device), batch)
            args.max_q_len = q_val.shape[1]
            args.max_a_len = a_val.shape[1]
            args.batch_size = gt_val.shape[0]
            result, predict = model(q_val, a_val, u_val, gt_val)
            loss += loss_fn(result, gt_val)
            pred_all += predict
            label_all += gt_val
    accuracy = Accuracy(pred_all, label_all)
    print("[Info] Accuacy: {}; {} samples, {} correct prediction".format(accuracy, len(pred_all), len(pred_all) * accuracy))
    return loss, accuracy



def grid_search(params_dic):
    '''

    :param params_dic: similar to {"conv_size":[0,1,2], "lstm_hiden_size":[1,2,3]}
    :return: iter {"conv_size":1, "lstm_hidden_size":1}
    '''
    grid_parameter = ParameterGrid(params_dic)
    parameter_list = []
    for params in grid_parameter:
        params_dic = {}
        for key in params_dic.keys():
            params_dic[key] = params[key]
        parameter_list.append(params_dic)
    return parameter_list



def train(args, train_data, val_data, test_data):


    model = HybridAttentionModel(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr )
    #TODO: Early stopping
    for epoch_i in range(args.epoch):
        print("[ Epoch " + epoch_i +" ]")
        train_epoch(model, train_data, optimizer, args)


        val_loss, accuracy_val = eval_epoch(model, val_data, args)
        print("[Info] Val Loss: {}, accuracy: {}".format(val_loss, accuracy_val))
        test_loss, accuracy_test = eval_epoch(model, test_data, args)
        print("[Info] Test Loss: {}, accuracy: {}".format(test_loss, accuracy_test))





def main():
    ''' setting '''
    parser = argparse.ArgumentParser()

    parser.add_argument("-epoch",type=int, default=10)
    parser.add_argument("-log", default=None)
    # load data
    # parser.add_argument("-data",required=True)
    parser.add_argument("-no_cuda", action="store_false")
    parser.add_argument("-max_q_len", type=int, default=60)
    parser.add_argument("-max_a_len", type=int, default=60)
    parser.add_argument("-max_u_len", type=int, default=200)
    parser.add_argument("-vocab_size", type=int, default=30000)
    parser.add_argument("-embed_size", type=int, default=100)
    parser.add_argument("-lstm_hidden_size",type=int, default=128)
    parser.add_argument("-bidirectional", action="store_true")
    parser.add_argument("-class_kind", type=int, default=2)
    parser.add_argument("-embed_fileName",default="/home/weiying/yichuan/data/glove/glove.6B.100d.txt")
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-num_layers", type=int, default=1)
    parser.add_argument("-drop_out", type=float, default=0.3)
    # conv parameter
    parser.add_argument("-in_channels_1", type=int, default=1)
    parser.add_argument("-in_channels_2", type=int, default=1)
    parser.add_argument("-in_channels_3",type=int, default=1)
    parser.add_argument("-out_channels_1", type=int, default=10)
    parser.add_argument("-out_channels_2", type=int, default=20)
    parser.add_argument("-out_channels_3", type=int, default=98)
    parser.add_argument("-kernel_size_1", type=int, default=3)
    parser.add_argument("-kernel_size_2", type=int, default=4)
    parser.add_argument("-kernel_size_3", type=int, default=5)
    parser.add_argument("-stride_1", type=int, default=5)
    parser.add_argument("-stride_2", type=int, default=5)
    parser.add_argument("-stride_3", type=int, default=5)
    parser.add_argument("-padding_1", type=int, default=5)
    parser.add_argument("-padding_2", type=int, default=5)
    parser.add_argument("-padding_3", type=int, default=5)


    args = parser.parse_args()
    assert args.out_channels_1 + args.out_channels_2 + args.out_channels_3 == args.lstm_hidden_size, \
    "conv out channels {} equal lstm hidden size {}".format(args.out_channels_1 + args.out_channels_2 + args.out_channels_3, args.lstm_hidden_size)
    #===========Load DataSet=============#


    args.data="/home/weiying/yichuan/data/fuck.model"
    #===========Prepare model============#
    args.cuda =  args.no_cuda
    args.device = torch.device('cuda' if args.cuda else 'cpu')
    print("cuda : {}".format(args.cuda))
    args.DEBUG=False
    data = torch.load(args.data)
    train_data, val_data, test_data = prepare_dataloaders(data, args)
    #grid search
    paragram_dic = {}
    pragram_list = grid_search(paragram_dic)
    args_dic = vars(args)
    for paragram in pragram_list:
        for key, value in paragram.items():
            # chaneg args
            args_dic[key] = value
        train(args, train_data, val_data, test_data)
if __name__ == '__main__':
    main()




