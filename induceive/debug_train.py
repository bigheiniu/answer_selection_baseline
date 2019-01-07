import argparse
from tqdm import tqdm
import torch
import torch.nn as nn


from induceive.dataset  import paired_collate_fn,Induceive_dataset
from attention.Utils import loadEmbed, Accuracy
from induceive.Model import InduceiveModel
from visualization.logger import Logger
import numpy as np

loss_fn = nn.NLLLoss()

info = {}
logger = Logger('./logs')


def pre_graphsage(data, opt, user_len):

    train_loader=torch.utils.data.DataLoader(
            Induceive_dataset(
                G=data['G'],
                content=data['content'],
                args=opt,
                user_len=user_len
            ),
            num_workers=8,
            batch_size=opt.batch_size,
            collate_fn=paired_collate_fn,
            shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        Induceive_dataset(
            G = data['G'],
            content=data['content'],
            training=False,
            args=opt,
            user_len=user_len
        ),
        num_workers=8,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)
    return train_loader, val_loader


def train_epoch(data, args, optimizer, model,loss_fn):
    model.train()
    with torch.autograd.detect_anomaly():
        for batch in tqdm(data, mininterval=2, desc=' --(training)--', leave=True):
            batch = list(map(lambda x: x.to(args.device), batch))
            optimizer.zero_grad()
            result,_ = model(*batch)
            loss = loss_fn(result, batch[3])
            if torch.isnan(loss):
                print("loss lose")
                print(" result is {} ".format(result))
                exit(-10)
            loss.backward()
            optimizer.step()
    for tag, value in model.named_parameters():
        if value.grad is None:
            continue
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, value.cpu().detach().numpy(), 0)
        logger.histo_summary(tag + '/grad', value.grad.cpu().numpy(), 0)

def eval_epoch(data, args, model, loss_fn):
    model.eval()
    label = []
    predict= []
    loss = 0
    with torch.no_grad():
        for batch in tqdm(
            data, mininterval=2, desc="  ----(validation)----  ", leave=True
        ):
            batch = list(map(lambda x: x.to(args.device), batch))
            result, pred = model(*batch)
            gt = batch[3]
            loss += float(loss_fn(result, gt))
            label.append(gt)
            predict.append(pred)
        label = torch.cat(label)
        predict = torch.cat(predict)
        accuracy, zero_count, one_count = Accuracy(label, predict)
        info['eval_loss'] = loss
        info['eval_accuracy'] = accuracy
        info['zero_count'] = zero_count
        info['one_count'] = one_count
        for tag, value in info.items():
            logger.scalar_summary(tag, value, 1)
        print("[Info] Accuacy: {}; {} samples, {} correct prediction\n Zero Count Ratio{}, One Count Ratio{}".format(accuracy,
                                                                                       len(label),
                                                                                       len(label) * accuracy, zero_count, one_count))
        return loss, accuracy

def train(train_data,test_data, args, word2idx, user_count):
    word_embed = loadEmbed(args.embed_fileName, args.embed_size, args.vocab_size,word2idx=word2idx, Debug=args.DEBUG)
    model = InduceiveModel(args, word_embed, user_count).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.NLLLoss()
    for i in range(args.epoch):
        train_epoch(train_data, args, optimizer, model, loss_fn)
        eval_epoch(test_data, args, model, loss_fn)






def main():
    ''' setting '''
    parser = argparse.ArgumentParser()

    parser.add_argument("-epoch", type=int, default=60)
    parser.add_argument("-log", default=None)
    # load data
    # parser.add_argument("-data",required=True)
    parser.add_argument("-no_cuda", action="store_false")
    parser.add_argument("-lr", type=float, default=1e-4)
    #graphsahe
    parser.add_argument("-max_degree", type=int, default=4)
    parser.add_argument("-content_len", type=int, default=60)
    parser.add_argument("-sample_layer", type=int, default=2)
    parser.add_argument("-act", type=str, default="tanh")

    # 1-UIA-LSTM-CNN; 2-CNTN
    # parser.add_argument("-model", type=int, default=1)
    parser.add_argument("-max_len", type=int, default=60)
    parser.add_argument("-embed_size", type=int, default=100)
    parser.add_argument("-lstm_hidden_size", type=int, default=128)
    parser.add_argument("-bidirectional", action="store_true")
    parser.add_argument("-class_kind", type=int, default=2)
    parser.add_argument("-embed_fileName", default="../data/glove/glove.6B.100d.txt")
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-lstm_nulrm_layers", type=int, default=1)
    parser.add_argument("-drop_out_lstm", type=float, default=0.5)
    # conv parameter
    parser.add_argument("-in_channels", type=int, default=1)
    parser.add_argument("-out_channels", type=int, default=20)
    parser.add_argument("-kernel_size", type=int, default=3)

    args = parser.parse_args()
    # ===========Load DataSet=============#

    args.data = "../data/store.torchpickle"
    args.sample_count=[2,3]
    # ===========Prepare model============#
    args.cuda = args.no_cuda
    args.device = torch.device('cuda' if args.cuda else 'cpu')
    print("cuda : {}".format(args.cuda))
    args.DEBUG = False

    args.lstm_num_layers = 2
    args.bidirectional=False
    args.label_size = 2
    args.vocab_size = 30000
    data = torch.load(args.data)
    train_loader, eval_loader = pre_graphsage(data,args,data['user_count'])
    train(train_loader, eval_loader, args, data['dict'], data['user_count'])

if __name__ == '__main__':
    main()




