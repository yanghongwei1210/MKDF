import time
import math
import argparse
import numpy as np
from Weight import Weight
import mmd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt

from revmodel import *
from utils import *
from evaluate import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    
    parser = argparse.ArgumentParser(description='HAN')
    
    parser.add_argument('--nodeS', type=str, required=True)
    parser.add_argument('--linkS', type=str, required=True)
    parser.add_argument('--configS', type=str, required=True)
    parser.add_argument('--labelS', type=str, required=True)
    parser.add_argument('--outputS', type=str, required=True)
    parser.add_argument('--metaS', type=str, required=True)

    parser.add_argument('--nodeT', type=str, required=True)
    parser.add_argument('--linkT', type=str, required=True)
    parser.add_argument('--configT', type=str, required=True)
    parser.add_argument('--labelT', type=str, required=True)
    parser.add_argument('--outputT', type=str, required=True)
    parser.add_argument('--metaT', type=str, required=True)
    
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda')
    
    parser.add_argument('--size', type=int, default=50)
    parser.add_argument('--nhead', type=str, default='8')
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--dropout', type=float, default=0.4)
    
    parser.add_argument('--neigh-por', type=float, default=0.6)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=500)   
    
    parser.add_argument('--attributed', type=str, default="False")
    parser.add_argument('--supervised', type=str, default="False")

    parser.add_argument('--datasetS', required=True, type=str, help='Targeting dataset.', 
                        choices=['ACM_a','ACM_b','ACM_c','DBLP_a','DBLP_b','DBLP_c'])
    parser.add_argument('--datasetT', required=True, type=str, help='Targeting dataset.', 
                        choices=['ACM_a','ACM_b','ACM_c','DBLP_a','DBLP_b','DBLP_c'])

    parser.add_argument('--model', required=True, type=str, help='Targeting model.', 
                        choices=['metapath2vec-ESim','PTE','HIN2Vec','AspEm','HEER','R-GCN','HAN','HGT','TransE','DistMult', 'ConvE'])
    parser.add_argument('--task', required=True, type=str, help='Targeting task.',
                        choices=['nc', 'lp', 'both'])    
    
    return parser.parse_args()

#Generate emb.dat
def output(args, embeddings, id_name):
    
    with open(args.outputT, 'w') as file:
        file.write(f'size={args.size}, nhead={args.nhead}, dropout={args.dropout}, neigh-por={args.neigh_por}, lr={args.lr}, batch-size={args.batch_size}, epochs={args.epochs}, attributed={args.attributed}, supervised={args.supervised}\n')
        for nid, name in id_name.items():
            file.write('{}\t{}\n'.format(name, ' '.join(embeddings[nid].astype(str))))
    
def main():
    
    torch.cuda.synchronize()
    start = time.time()

    args = parse_args()
    
    set_seed(args.seed, args.device)

    adjsS, id_nameS, featuresS = load_data_semisupervised(args, args.nodeS, args.linkS, args.configS, list(map(lambda x: int(x), args.metaS.split(','))))

    train_poolS, train_labelS, nlabelS, multiS = load_label(args.labelS, id_nameS)

    adjsT, id_nameT, featuresT = load_data_semisupervised(args, args.nodeT, args.linkT, args.configT, list(map(lambda x: int(x), args.metaT.split(','))))

    train_poolT, train_labelT, nlabelT, multiT = load_label(args.labelT, id_nameT)
    # print('train_poolS',train_poolS)
    # print('train_labelS',train_labelS)
    # print('nlabelS',nlabelS)
    # print('multiS',multiS)

    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + 'finish loading', flush=True)    
    
    nhead = list(map(lambda x: int(x), args.nhead.split(',')))
    # print('#### nhead ####',nhead)#[8]
    nnodeS, nchannel, nlayer = len(id_nameS), len(adjsS), len(nhead)
    nnodeT, nchannel, nlayer = len(id_nameT), len(adjsT), len(nhead)
    # print('#### nchannel ####',nchannel)#2
    # print('#### nlayer ####',nlayer)#1

    if args.attributed=='True': nfeat = featuresS.shape[1]

    model = RevGrad(nchannel, nfeat, args.size, nlabelS, nlayer, nhead, args.neigh_por, args.dropout, args.alpha, args.device)
    embeddingsS = torch.from_numpy(featuresS).to(args.device) if args.attributed=='True' else torch.from_numpy(np.random.randn(nnodeS, args.size).astype(np.float32)).to(args.device)
    embeddingsT = torch.from_numpy(featuresT).to(args.device) if args.attributed=='True' else torch.from_numpy(np.random.randn(nnodeT, args.size).astype(np.float32)).to(args.device)

    train_labelS = torch.from_numpy(train_labelS.astype(np.float32)).to(args.device)
    # print('train_labelS',train_labelS.shape)
    train_labelT = torch.from_numpy(train_labelT.astype(np.float32)).to(args.device)
    # print('train_labelT',train_labelT.shape)

    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + f'start training', flush=True)
    
    test_loss = torch.empty(args.epochs)
    test_accuracy = torch.empty(args.epochs)
    micro_f1 = torch.empty(args.epochs)
    macro_f1 = torch.empty(args.epochs)
    ii = -1
    best_target_acc = 0
    best_epoch = 0.0

    for epoch in range(args.epochs):
        model.train()
        # for name,parameters in model.named_parameters():
        #     print(name,':',parameters.size())
        # exit()
        LEARNING_RATE = args.lr / math.pow((1 + 10 * epoch / args.epochs), 0.75)
        print("learning rate: ", LEARNING_RATE)

        optimizer = optim.Adam([{'params': model.sharedNet.parameters()},
        {'params': model.linear_block.parameters(), 'lr': LEARNING_RATE},
        # {'params': model.HeteroAttLayerS.parameters(), 'lr': LEARNING_RATE},
        {'params': model.HeteroAttLayerT.parameters(), 'lr': LEARNING_RATE},
        {'params': model.cls_fc1.parameters(), 'lr': LEARNING_RATE},
        # {'params': model.cls_fcF.parameters(), 'lr': LEARNING_RATE},
        {'params': model.cls_fc2.parameters(), 'lr': LEARNING_RATE}], lr=LEARNING_RATE/10, weight_decay=args.weight_decay)
        # optimizer = torch.optim.SGD([{'params': model.sharedNet.parameters()},
        # {'params': model.HeteroAttLayer.parameters(), 'lr': LEARNING_RATE},
        # # {'params': model.HeteroAttLayerT.parameters()},
        # # {'params': model.cls_fcS.parameters(), 'lr': LEARNING_RATE},
        # {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE}], lr=LEARNING_RATE/10, momentum=0.9,  weight_decay=args.weight_decay)

        batch_size = int(args.batch_size)
        num_iter = math.ceil(nnodeS/batch_size)
        print('num_iter',num_iter)
        for i in range(1, num_iter):
            curr_indexS = np.sort(np.random.choice(np.arange(len(train_poolS)), args.batch_size, replace=False))
            curr_batchS = train_poolS[curr_indexS]

            curr_indexT = np.sort(np.random.choice(np.arange(len(train_poolT)), args.batch_size, replace=False))
            curr_batchT = train_poolT[curr_indexT]

            # eta = nnodeT/(adj.sum()/adj.shape[0])**len(model_config['connection'])

            # print('embeddingsT',embeddingsT.size())#[4154, 1255]
            # print('adjsT',np.asarray(adjsT).shape)
            # print('curr_batchT',curr_batchT.shape)
            optimizer.zero_grad()
            cls_lossS,cls_lossT,  mmd_loss, l1_loss, _ ,cls_lossS_MP= model(embeddingsS, adjsS, curr_batchS, embeddingsT, adjsT, curr_batchT, train_labelS[curr_indexS],Training=True,mark=0)
            # gamma = 2 / (1 + math.exp(-10 * (epoch+1) / args.epochs)) - 1
            gamma=0.01
            loss = cls_lossS + gamma * (mmd_loss + l1_loss)#+cls_lossT
            loss.backward()
            optimizer.step()
            # new_gcn_index = clabel_predT_pre.data.max(1)[1]
            # print('train_labelT',train_labelT)
            # new_gcn_index = np.argmax(clabel_predT_pre, axis=1)
            # confidence = clabel_predT_pre.data.max()[0]
            # # print('confidence',confidence)
            # confidence_np = confidence.numpy()
            # sorted_index = np.argsort(-confidence_np)
            # sorted_index = torch.from_numpy(sorted_index)

            # target_probs = F.softmax(clabel_tgt, dim=-1)
            # target_probs = torch.clamp(target_probs, min=1e-9, max=1.0)

            # loss_entropy = torch.mean(torch.sum(-target_probs * torch.log(target_probs), dim=-1))

            # loss_mmd = mmd.lmmd(feadata_src, feadata_tgt, train_labelS[curr_indexS], torch.nn.functional.softmax(clabel_tgt, dim=1))
            

            optimizer.zero_grad()
            cls_lossS,cls_lossT, mmd_loss, l1_loss, _,cls_lossS_MP = model(embeddingsS, adjsS, curr_batchS, embeddingsT, adjsT, curr_batchT, train_labelS[curr_indexS],Training=True,mark=1)

            # gamma = 2 / (1 + math.exp(-10 * (epoch+1) / args.epochs)) - 1
            gamma=0.01
            loss = cls_lossS + gamma * (mmd_loss + l1_loss)#+cls_lossT
            loss.backward()
            optimizer.step()

            # label_loss = loss_entropy* (epoch / args.epochs * 0.01) + F.nll_loss(F.log_softmax(clabel_src, dim=1), train_labelS[curr_indexS].long())
            # label_loss = loss_entropy + F.nll_loss(F.log_softmax(clabel_src, dim=1), train_labelS[curr_indexS].long())
            
            # + F.nll_loss(F.log_softmax(clabel_tgt, dim=1), train_labelT.long())
            # print('train_labelS[curr_indexS]',train_labelS[curr_indexS].size())
            # exit()

            # loss = label_loss + 0.3 * lambd * loss_mmd
            # loss.backward()
            # optimizer.step()
            if i % 10 == 0:
                print(i)
            # print('Train Epoch: {}\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}'.format(epoch,loss.item(), label_loss.item(), loss_mmd.item()))

        print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + f'finish epoch {epoch}', flush=True)

            
        print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + 'output embedding', flush=True)
        model.eval()
        outbatch_size = int(args.batch_size)
        rounds = math.ceil(nnodeT/outbatch_size)
        print("#####@@@@@@@@@@nnodeT")
        print("#####@@@@@@@@@@nnodeT",nnodeT)
        outputs = np.zeros((nnodeT, 4)).astype(np.float32)
        for index, i in enumerate(range(rounds)):
            seed_nodes = np.arange(i*outbatch_size, min((i+1)*outbatch_size, nnodeT))

            _,cls_loss, mmd_loss, l1_loss, c_pred,_ = model(embeddingsT, adjsT, seed_nodes, embeddingsT, adjsT, seed_nodes, train_labelS[curr_indexS],Training=True,mark=-1)
            outputs[seed_nodes] = c_pred.detach().cpu().numpy()
            print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + f'finish output batch {index} -> {rounds}', flush=True)
        output(args, outputs, id_nameT)

        ii = ii + 1
        print('Load Embeddings!')
        model_folder = "/home/user/桌面/Ljx/MKDF/Model"
        emb_file_path = f'{model_folder}/{args.model}/data/{args.datasetT}/{emb_file}'
        train_para, emb_dict = load(emb_file_path)
    
        print('Start Evaluation!')
        all_tasks, all_scores = [], []

        print(f'Evaluate Node Classification Performance for Model {args.model} on Dataset {args.datasetT}!')
        data_folder = "/home/user/桌面/Ljx/MKDF/Model/HAN/data"
        label_file_path = f'{data_folder}/{args.datasetT}/{label_file}'
        print('#####label_file_path',label_file_path)
        label_test_path = f'{data_folder}/{args.datasetT}/{label_test_file}'
        print('#####label_test_path',label_test_path)

        scores = nc_evaluate(args.datasetT, args.supervised, label_file_path, label_test_path, emb_dict)

        print('scores',scores)

        macro_f1[ii], micro_f1[ii], test_accuracy[ii], test_loss[ii] = semisupervised_single_class_single_label(label_file_path, label_test_path, emb_dict)

        if test_accuracy[ii] > best_target_acc:
            best_target_acc = test_accuracy[ii]
            best_epoch = epoch
            bestscores = nc_evaluate(args.datasetT, args.supervised, label_file_path, label_test_path, emb_dict)
            print("==========================================")
            line = "{} - best_Epoch: {},  best_target_acc: {}\n"\
                .format(epoch, best_epoch, best_target_acc)
            print(line)
            #Address of the pre-trained model 
            save_model_path = "/home/user/桌面/Ljx/MKDF/Model/HAN/DBLP_ABtoC/DBLP_atoc" + str(best_epoch)+'socre_'+str(best_target_acc) + "model.pkl"
            torch.save(model.state_dict(),save_model_path)
            print("model saved in ",save_model_path)
        linee = "{} - best_Epoch: {},  best_target_acc: {}\n"\
                .format(epoch, best_epoch, best_target_acc)
        print(linee)
    print(linee)
   
    torch.cuda.synchronize()
    end = time.time()
    print('time',end-start)

if __name__ == '__main__':
    main()