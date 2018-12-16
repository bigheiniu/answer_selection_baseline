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
import torchvision


def prepare_dataloaders(data, opt):
    # ========= Preparing DataLoader =========#

    train_loader = torch.utils.data.DataLoader(
        QuestionAnswerUser(
            word2idx=data['dict'],
            word_insts=data['content'],
            user=data['user'],
            question_anser_user=data['question_anser_user'],
            max_u_len=opt.max_u_len
        ),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)
    return train_loader



def train_epoch(model, data, optimizer, args):
    model.train()
    for batch in tqdm(
        data, mininterval=2, desc=' --(training)--',leave=True
    ):

        q_iter, a_iter, u_iter, gt_iter = map(lambda x: x.to(args.device), batch)
        args.max_q_len = q_iter.shape[1]
        args.max_a_len = a_iter.shape[1]
        optimizer.zero_grad()
        loss = model(q_iter, a_iter, u_iter, gt_iter)
        loss.backward()
        optimizer.step()




def train(args, data):
    # # valid_accus = []
    # N = 300
    # length = N * args.batch_size
    #
    # # if(args.DEBUG):
    # #     train_q = torch.randint(0, args.vocab_size, (length, args.max_q_len), dtype=torch.long, device = args.device)
    # #     train_a = torch.randint(0, args.vocab_size, (length, args.max_a_len), dtype=torch.long, device = args.device)
    # #     train_u =torch.randint(0, args.vocab_size, (length, args.max_u_len), dtype=torch.long, device = args.device)
    # # gt = torch.randint(0, 1, (3000,), dtype=torch.long, device=args.device)
    model = HybridAttentionModel(args)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
    for epoch_i in range(args.epoch):
        print("[ Epoch ", epoch_i , "]")
        train_epoch(model, data, optimizer, args)





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
    parser.add_argument("-embed_fileName",default=None)
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


    args.data="./fuck.model"
    #===========Prepare model============#
    args.cuda = not args.no_cuda
    args.device = torch.device('cuda' if args.cuda else 'cpu')
    args.DEBUG=True
    data = torch.load(args.data)
    data = prepare_dataloaders(data, args)
    train(args, data)
if __name__ == '__main__':
    main()




