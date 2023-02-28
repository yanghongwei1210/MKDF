import time
import math
import argparse
import numpy as np
from Weight import Weight
import mmd
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from mmd import lmmd
from revmodel import *
from utils import *
from evaluate import *
from calculateMMD import calculateMMDDIS
import os
from scipy.stats import wasserstein_distance
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    
    parser = argparse.ArgumentParser(description='HAN')
    
    parser.add_argument('--nodeS_1', type=str, required=True)
    parser.add_argument('--linkS_1', type=str, required=True)
    parser.add_argument('--configS_1', type=str, required=True)
    parser.add_argument('--labelS_1', type=str, required=True)
    parser.add_argument('--outputS_1', type=str, required=True)
    parser.add_argument('--metaS_1', type=str, required=True)

    parser.add_argument('--nodeS_2', type=str, required=True)
    parser.add_argument('--linkS_2', type=str, required=True)
    parser.add_argument('--configS_2', type=str, required=True)
    parser.add_argument('--labelS_2', type=str, required=True)
    parser.add_argument('--outputS_2', type=str, required=True)
    parser.add_argument('--metaS_2', type=str, required=True)
    
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

    parser.add_argument('--datasetS_1', required=True, type=str, help='Targeting dataset.', 
                        choices=['ACM_a','ACM_b','ACM_c','DBLP_a','DBLP_b','DBLP_c','DBLP','Freebase','PubMed','ACM','Yelp'])
    parser.add_argument('--datasetS_2', required=True, type=str, help='Targeting dataset.', 
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
    
    # 1.Load target domain data
    print("Load target domain data")
    torch.cuda.synchronize()
    args = parse_args()  
    set_seed(args.seed, args.device)
    adjsT, id_nameT, featuresT = load_data_semisupervised(args, args.nodeT, args.linkT, args.configT, list(map(lambda x: int(x), args.metaT.split(','))))
    train_poolT, train_labelT, nlabelT, multiT = load_label(args.labelT, id_nameT)
    nhead = list(map(lambda x: int(x), args.nhead.split(',')))
    nnodeT, nchannel, nlayer = len(id_nameT), len(adjsT), len(nhead)      
    embeddingsT = torch.from_numpy(featuresT).to(args.device) if args.attributed=='True' else torch.from_numpy(np.random.randn(nnodeT, args.size).astype(np.float32)).to(args.device)
    train_labelT = torch.from_numpy(train_labelT.astype(np.float32)).to(args.device)
    temper_labelT = train_labelT.cpu().detach().numpy()
    np_labelT = np.zeros((len(temper_labelT),4))
    for i in range(len(temper_labelT)):
        np_labelT[i][int(temper_labelT[i])] = 1.0
    np_labelT = torch.from_numpy(np_labelT.astype(np.float32)).to(args.device)
      
    # Define two arrays, which correspond to the best model of each distillation rate of 0-0.5
  
    bestepoch_1 = ['49socre_tensor(0.7505)','0.123distill','0.237distill','0.344distill','0.465distill','0.565distill']#,'40.6distill','00.7distill','70.8distill','00.9distill','11.0distill']
    bestepoch_2 = ['76socre_tensor(0.7338)','0.153distill','0.231distill','0.337distill','0.465distill','0.541distill']#,'140.6distill','10.7distill','00.8distill','00.9distill','301.0distill']
    
    
    """
    The array records whether the distance of the model has been calculated.
    If it has been calculated, it will be taken directly. 
    If not, it will be calculated. Then the corresponding value is set to 1.
    """

    emd_caculated_1 = np.zeros(6)
    emd_caculated_2 = np.zeros(6)
    # emd_caculated_1 = np.ones(6)
    # emd_caculated_2 = np.ones(6)
    best_micro=0
    best_macro=0

    for m_1 in range(6):
        for m_2 in range(6):
            if m_1 != 0:
                args.nodeS_1 = 'data/node-label/DBLP/ABtoC/DBLP_a/'+str(m_1/10)+'/node.dat'
                args.linkS_1 = 'data/node-label/DBLP/ABtoC/DBLP_a/'+str(m_1/10)+'/link.dat'
                args.configS_1 = 'data/node-label/DBLP/ABtoC/DBLP_a/'+str(m_1/10)+'/config.dat'
                args.labelS_1 = 'data/node-label/DBLP/ABtoC/DBLP_a/'+str(m_1/10)+'/labelall.dat'
                args.outputS_1 = 'data/node-label/DBLP/ABtoC/DBLP_a/'+str(m_1/10)+'/emd.dat'
    
            adjsS_1, id_nameS_1, featuresS_1 = load_data_semisupervised(args, args.nodeS_1, args.linkS_1, args.configS_1, list(map(lambda x: int(x), args.metaS_1.split(','))))
            train_poolS_1, train_labelS_1, nlabelS_1, multiS_1 = load_label(args.labelS_1, id_nameS_1)
            nhead = list(map(lambda x: int(x), args.nhead.split(',')))
            nnodeS_1, nchannel, nlayer = len(id_nameS_1), len(adjsS_1), len(nhead)      
            embeddingsS_1 = torch.from_numpy(featuresS_1).to(args.device) if args.attributed=='True' else torch.from_numpy(np.random.randn(nnodeS_1, args.size).astype(np.float32)).to(args.device)
            train_labelS_1 = torch.from_numpy(train_labelS_1.astype(np.float32)).to(args.device)
            if m_2 != 0:
                args.nodeS_2 = 'data/node-label/DBLP/ABtoC/DBLP_b/'+str(m_2/10)+'/node.dat'
                args.linkS_2 = 'data/node-label/DBLP/ABtoC/DBLP_b/'+str(m_2/10)+'/link.dat'
                args.configS_2 = 'data/node-label/DBLP/ABtoC/DBLP_b/'+str(m_2/10)+'/config.dat'
                args.labelS_2 = 'data/node-label/DBLP/ABtoC/DBLP_b/'+str(m_2/10)+'/labelall.dat'
                args.outputS_2 = 'data/node-label/DBLP/ABtoC/DBLP_b/'+str(m_2/10)+'/emd.dat'
                
            adjsS_2, id_nameS_2, featuresS_2 = load_data_semisupervised(args, args.nodeS_2, args.linkS_2, args.configS_2, list(map(lambda x: int(x), args.metaS_2.split(','))))
            train_poolS_2, train_labelS_2, nlabelS_2, multiS_2 = load_label(args.labelS_2, id_nameS_2)
            nhead = list(map(lambda x: int(x), args.nhead.split(',')))
            nnodeS_2, nchannel, nlayer = len(id_nameS_2), len(adjsS_2), len(nhead)      
            embeddingsS_2 = torch.from_numpy(featuresS_2).to(args.device) if args.attributed=='True' else torch.from_numpy(np.random.randn(nnodeS_2, args.size).astype(np.float32)).to(args.device)
            train_labelS_2 = torch.from_numpy(train_labelS_2.astype(np.float32)).to(args.device)
            
      
            
            print("First Model " + str(m_1) + ", Second Model " + str(m_2))
            # 2.Load Model
            # print("Load Model")
            if args.attributed=='True': nfeat = featuresT.shape[1]
            save_model_path_1 = "/home/user/桌面/Ljx/MKDF/Model/HAN/DBLP_ABtoC/DBLP_atoc" + bestepoch_1[m_1] + "model.pkl"
            model_1 = RevGrad(nchannel, nfeat, args.size, nlabelT, nlayer, nhead, args.neigh_por, args.dropout, args.alpha, args.device)
            model_1.load_state_dict(torch.load(save_model_path_1))

            save_model_path_2 = "/home/user/桌面/Ljx/MKDF/Model/HAN/DBLP_ABtoC/DBLP_btoc" + bestepoch_2[m_2] + "model.pkl"
            model_2 = RevGrad(nchannel, nfeat, args.size, nlabelT, nlayer, nhead, args.neigh_por, args.dropout, args.alpha, args.device)
            model_2.load_state_dict(torch.load(save_model_path_2))

            # 3.Embedding
            #3.1 Model 1
            # print(" 3.  Embedding")
            # print("Model 1")
            model_1.eval()
            curr_indexS_1 = np.sort(np.random.choice(np.arange(len(train_poolS_1)), args.batch_size, replace=False))
            outbatch_size = int(args.batch_size)
            rounds = math.ceil(nnodeT/outbatch_size)
            outputs_1 = np.zeros((nnodeT, 4)).astype(np.float32)
            for index, i in enumerate(range(rounds)):
                seed_nodes = np.arange(i*outbatch_size, min((i+1)*outbatch_size, nnodeT))
                _,cls_loss, mmd_loss, l1_loss, c_pred,_ = model_1(embeddingsT, adjsT, seed_nodes, embeddingsT, adjsT, seed_nodes, train_labelS_1[curr_indexS_1],Training=True,mark=-1)
                outputs_1[seed_nodes] = c_pred.detach().cpu().numpy()
            #3.2 Model 2
            # print(" Model 2")
            model_2.eval()
            curr_indexS_2 = np.sort(np.random.choice(np.arange(len(train_poolS_2)), args.batch_size, replace=False))
            outbatch_size = int(args.batch_size)
            rounds = math.ceil(nnodeT/outbatch_size)
            outputs_2 = np.zeros((nnodeT, 4)).astype(np.float32)
            for index, i in enumerate(range(rounds)):
                seed_nodes = np.arange(i*outbatch_size, min((i+1)*outbatch_size, nnodeT))
                _,cls_loss, mmd_loss, l1_loss, c_pred,_ = model_2(embeddingsT, adjsT, seed_nodes, embeddingsT, adjsT, seed_nodes, train_labelS_2[curr_indexS_2],Training=True,mark=-1)
                outputs_2[seed_nodes] = c_pred.detach().cpu().numpy()  

            # 4.Aggregate logits
            # 4.1 calculate mmd
            # print(" 4.  Aggregate logits")
            # print(" 4.1 calculate emd and w")
            # Aggregate emd distance wasserstein_distance featuresT:numpy.ndarray
            # print("featuresT:",type(featuresT))
            # print("featuresS_1:",featuresS_1)
            # Define an array of featuresT length

            emdT_1 = np.zeros(len(featuresT))
            if emd_caculated_1[m_1] == 0:
    
                # print("emdT:",emdT_1)
                concat_featuresS_1 = np.concatenate(featuresS_1)
                for i in range(len(featuresS_1)):
                    concat_featuresS_1 = np.concatenate((concat_featuresS_1,featuresS_1[i]))

                print("concat_featuresS_1:",len(concat_featuresS_1))
                # Calculate the distance between each node of featuresT and featuresS_1 and add it to the list
                for i in range(len(featuresT)):
                    # print(str(i) + "th LOOP")
                    emdT_1[i] = wasserstein_distance(featuresT[i],concat_featuresS_1)
                print("emdT_1:",emdT_1)
                # # #Calculate once takes 7 minutes, exists directly in the file next time read directly
                path_1 = '/home/user/桌面/Ljx/MKDF/Model/HAN/data/'+str(m_1/10)+'a.txt'
                with open(path_1,'w') as file:
                    for i in range(len(emdT_1)):
                        if i == len(emdT_1)-1:
                            file.write(str(emdT_1[i]))
                        else:
                            file.write(str(emdT_1[i]) +",")
                file.close()
                emd_caculated_1[m_1] = 1
            # Get emd
            path_1 = '/home/user/桌面/Ljx/MKDF/Model/HAN/data/'+str(m_1/10)+'a.txt'
            with open(path_1,'r') as file:
                for line in file:
                    emdT_1 = np.array(line.split(',')).astype(np.float32)
            print("emdT_1:",emdT_1)
            w1 = np.zeros(len(emdT_1))
            for i in range(len(emdT_1)):
                w1[i] = math.exp((-math.pow(emdT_1[i],2)/2))
            print("w1:",w1)

            # Aggregate emd distance wasserstein_distance featuresT:numpy.ndarray
            # print("featuresT:",type(featuresT))
            # print("featuresS_2:",featuresS_2)
            # Define an array of featuresT length
            emdT_2 = np.zeros(len(featuresT))
            if emd_caculated_2[m_2] == 0:
                
                # print("emdT_2:",emdT_2)
                concat_featuresS_2 = np.concatenate(featuresS_2)
                for i in range(len(featuresS_2)):
                    concat_featuresS_2 = np.concatenate((concat_featuresS_2,featuresS_2[i]))

                print("concat_featuresS_2:",len(concat_featuresS_2))
                # Calculate the distance between each node of featuresT and featuresS_2 and add it to the list
                for i in range(len(featuresT)):
                    # print(str(i) + "th LOOP")
                    emdT_2[i] = wasserstein_distance(featuresT[i],concat_featuresS_2)
                print("emdT_2:",emdT_2)
                # #Calculate once takes 7 minutes, exists directly in the file next time read directly
                path_2 = '/home/user/桌面/Ljx/MKDF/Model/HAN/data/'+str(m_2/10)+'c.txt'
                with open(path_2,'w') as file:
                    for i in range(len(emdT_2)):
                        if i == len(emdT_2)-1:
                            file.write(str(emdT_2[i]))
                        else:
                            file.write(str(emdT_2[i]) +",")
                file.close()
                emd_caculated_2[m_2] = 1
            #Get emd
            path_2 = '/home/user/桌面/Ljx/MKDF/Model/HAN/data/'+str(m_2/10)+'c.txt'
            with open(path_2,'r') as file:
                for line in file:
                    emdT_2 = np.array(line.split(',')).astype(np.float32)
            w2 = np.zeros(len(emdT_2))
            for i in range(len(emdT_2)):
                w2[i] = math.exp((-math.pow(emdT_2[i],2)/2))
            print("w2:",w2)

            # 4.2 Merge outputs 
            # print("Merge outputs")

            outputs = np.zeros((len(outputs_1),len(outputs_1[0])))
            for i in range(len(outputs_1)):
                outputs[i] = np.multiply(outputs_1[i], w1[i]) + np.multiply(outputs_2[i], w2[i])

            output(args, outputs, id_nameT)

            # 5.Classification Evaluation
            # print("Classification Evaluation")
            # print('Load Embeddings!')
            model_folder = "/home/user/桌面/Ljx/MKDF/Model"
            emb_file_path = f'{model_folder}/{args.model}/data/{args.datasetT}/{emb_file}'
            train_para, emb_dict = load(emb_file_path)
            # print('Start Evaluation!')


            # print(f'Evaluate Node Classification Performance for Model {args.model} on Dataset {args.datasetT}!')
            data_folder = "/home/user/桌面/Ljx/MKDF/Model/HAN/data"
            label_file_path = f'{data_folder}/{args.datasetT}/{label_file}'
            # print('#####label_file_path',label_file_path)
            label_test_path = f'{data_folder}/{args.datasetT}/{label_test_file}'
            # print('#####label_test_path',label_test_path)


            macro_f1, micro_f1, test_accuracy, test_loss = semisupervised_single_class_single_label(label_file_path, label_test_path, emb_dict)


            # 6.Finished！
            print("Finished")
            print("micro_f1:",micro_f1)
            print("macro_f1:",macro_f1)
            print("accuracy:",test_accuracy)
            if micro_f1>best_micro:
                best_micro=micro_f1
                best_macro=macro_f1

    print("-------------------")
    print('best_micro:',best_micro)
    print('best_macro:',best_macro)
    torch.cuda.synchronize()
    end = time.time()

if __name__ == '__main__':
    main()