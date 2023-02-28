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
from scipy.stats import wasserstein_distance

from revmodel import *
from utils import *
from evaluate import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
    parser.add_argument('--epochs', type=int, default=200)   
    
    parser.add_argument('--attributed', type=str, default="False")
    parser.add_argument('--supervised', type=str, default="False")

    parser.add_argument('--datasetS', required=True, type=str, help='Targeting dataset.', 
                        choices=['ACM_a','ACM_b','ACM_c','DBLP_a','DBLP_b','DBLP_c','DBLP','Freebase','PubMed','ACM','Yelp'])
    parser.add_argument('--datasetT', required=True, type=str, help='Targeting dataset.', 
                        choices=['ACM_a','ACM_b','ACM_c','DBLP_a','DBLP_b','DBLP_c','DBLP','Freebase','PubMed','ACM','Yelp'])

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
    
    #The cycle can directly run the distillation results from 0.1 to 0.5
    for i in range(1,6):
        distillRate = i / 10
        print("distill rate:",distillRate)
        args.nodeS = 'data/node-label/DBLP/ABtoC/DBLP_a/'+str(distillRate)+'/node.dat'
        args.linkS = 'data/node-label/DBLP/ABtoC/DBLP_a/'+str(distillRate)+'/link.dat'
        args.configS = 'data/node-label/DBLP/ABtoC/DBLP_a/'+str(distillRate)+'/config.dat'
        adjsS, id_nameS, featuresS = load_data_semisupervised(args, args.nodeS, args.linkS, args.configS, list(map(lambda x: int(x), args.metaS.split(','))))
        args.labelS = 'data/node-label/DBLP/ABtoC/DBLP_a/'+str(distillRate)+'/labelall.dat'
        args.outputS = 'data/node-label/DBLP/ABtoC/DBLP_a/'+str(distillRate)+'/emd.dat'
        train_poolS, train_labelS, nlabelS, multiS = load_label(args.labelS, id_nameS)
        adjsT, id_nameT, featuresT = load_data_semisupervised(args, args.nodeT, args.linkT, args.configT, list(map(lambda x: int(x), args.metaT.split(','))))
        train_poolT, train_labelT, nlabelT, multiT = load_label(args.labelT, id_nameT)
        # print('train_poolS',train_poolS)
        # print('train_labelS',train_labelS)
        # print('nlabelS',nlabelS)
        # print('multiS',multiS)
      

        # 1. Calculate the distance of each node

        # 2. Add and average the distances of nodes on the meta path to get the distance of the meta path

        # 3. Sort the meta path distance to get the threshold

        # 4. Delete the meta paths that are far away

        nhead = list(map(lambda x: int(x), args.nhead.split(',')))
        # print('#### nhead ####',nhead)#[8]
        nnodeS, nchannel, nlayer = len(id_nameS), len(adjsS), len(nhead)
        nnodeT, nchannel, nlayer = len(id_nameT), len(adjsT), len(nhead)
        # print('#### nchannel ####',nchannel)#2
        # print('#### nlayer ####',nlayer)#1

        if args.attributed=='True': nfeat = featuresS.shape[1]

        #Address of the model with the highest pre training accuracy
        save_model_path = "/home/user/桌面/Ljx/MKDF/Model/HAN/DBLP_ABtoC/DBLP_atoc49socre_tensor(0.7505)model.pkl"
        model = RevGrad(nchannel, nfeat, args.size, nlabelS, nlayer, nhead, args.neigh_por, args.dropout, args.alpha, args.device)
        # for p in model.parameters():
        #     print(p.requires_grad)
        model.load_state_dict(torch.load(save_model_path))
        # print("------------------------")
        # for p in model.parameters():
        #     print(p.requires_grad)
        embeddingsS = torch.from_numpy(featuresS).to(args.device) if args.attributed=='True' else torch.from_numpy(np.random.randn(nnodeS, args.size).astype(np.float32)).to(args.device)
        embeddingsT = torch.from_numpy(featuresT).to(args.device) if args.attributed=='True' else torch.from_numpy(np.random.randn(nnodeT, args.size).astype(np.float32)).to(args.device)

        train_labelS = torch.from_numpy(train_labelS.astype(np.float32)).to(args.device)
        # print('train_labelS',train_labelS.shape)
        train_labelT = torch.from_numpy(train_labelT.astype(np.float32)).to(args.device)
        # print('train_labelT',train_labelT.shape)



        test_loss = torch.empty(args.epochs)
        test_accuracy = torch.empty(args.epochs)
        micro_f1 = torch.empty(args.epochs)
        macro_f1 = torch.empty(args.epochs)
        ii = -1
        best_target_acc = 0
        best_epoch = 0.0

        for epoch in range(args.epochs):
            loss_S = 0
            model.train()
            # for name,parameters in model.named_parameters():
            #     print(name,':',parameters.size())
            # exit()
            LEARNING_RATE = args.lr / math.pow((1 + 10 * epoch / args.epochs), 0.75)

            optimizer = optim.Adam([{'params':filter(lambda p: p.requires_grad,model.parameters()),'lr':LEARNING_RATE}], lr=LEARNING_RATE/10, weight_decay=args.weight_decay)

            # optimizer = torch.optim.SGD([{'params': model.sharedNet.parameters()},
            # {'params': model.HeteroAttLayer.parameters(), 'lr': LEARNING_RATE},
            # # {'params': model.HeteroAttLayerT.parameters()},
            # # {'params': model.cls_fcS.parameters(), 'lr': LEARNING_RATE},
            # {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE}], lr=LEARNING_RATE/10, momentum=0.9,  weight_decay=args.weight_decay)

            batch_size = int(args.batch_size)
            num_iter = math.ceil(nnodeS/batch_size)
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
                loss_S = loss_S + cls_lossS  
                loss = cls_lossS + gamma * l1_loss #+cls_lossT
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
                loss_S = loss_S + cls_lossS 
                loss = cls_lossS + gamma * l1_loss#+cls_lossT
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

            model.eval()
            outbatch_size = int(args.batch_size)
            rounds = math.ceil(nnodeT/outbatch_size)
            outputs = np.zeros((nnodeT, 4)).astype(np.float32)
            for index, i in enumerate(range(rounds)):
                seed_nodes = np.arange(i*outbatch_size, min((i+1)*outbatch_size, nnodeT))

                _,cls_loss, mmd_loss, l1_loss, c_pred,_ = model(embeddingsT, adjsT, seed_nodes, embeddingsT, adjsT, seed_nodes, train_labelS[curr_indexS],Training=True,mark=-1)
                outputs[seed_nodes] = c_pred.detach().cpu().numpy()
            output(args, outputs, id_nameT)

            ii = ii + 1
            model_folder = "/home/user/桌面/Ljx/MKDF/Model"
            emb_file_path = f'{model_folder}/{args.model}/data/{args.datasetT}/{emb_file}'
            train_para, emb_dict = load(emb_file_path)

            all_tasks, all_scores = [], []

            data_folder = "/home/user/桌面/Ljx/MKDF/Model/HAN/data"
            label_file_path = f'{data_folder}/{args.datasetT}/{label_file}'
            # print('#####label_file_path',label_file_path)
            label_test_path = f'{data_folder}/{args.datasetT}/{label_test_file}'
            # print('#####label_test_path',label_test_path)

            scores = nc_evaluate(args.datasetT, args.supervised, label_file_path, label_test_path, emb_dict)


            macro_f1[ii], micro_f1[ii], test_accuracy[ii], test_loss[ii] = semisupervised_single_class_single_label(label_file_path, label_test_path, emb_dict)

            if test_accuracy[ii] > best_target_acc:
                best_target_acc = test_accuracy[ii]
                best_macro =  macro_f1[ii]
                best_epoch = epoch
                line = "{} - best_Epoch: {},  best_target_acc: {}, best_macro: {}"\
                    .format(epoch, best_epoch, best_target_acc, best_macro)
                print(line)
                #Address of the model after distillation
                save_model_path = "/home/user/桌面/Ljx/MKDF/Model/HAN/DBLP_ABtoC/DBLP_atoc" + str(distillRate) +'_'+ str(best_target_acc) +"distillmodel.pkl"
                torch.save(model.state_dict(),save_model_path)

        torch.cuda.synchronize()


if __name__ == '__main__':
    main()