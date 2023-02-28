import random
import numpy as np

import torch
import torch.nn as nn
from Weight import Weight
import mmd

import torch.nn.functional as F


class HomoAttLayer(nn.Module):
    # curr_in_dim1255, out_dim64, dropout, alpha, device
    def __init__(self, in_dim, out_dim, dropout, alpha, device):
        super(HomoAttLayer, self).__init__()
        
        self.dropout = dropout
        self.device = device
        
        self.W = nn.Parameter(torch.zeros(size=(in_dim, out_dim)))#1255，64
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_dim, 1)))#128，1
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(alpha)
    
    def forward(self, features, adj, target_len, neighbor_len, target_index_out):
        # print('features',features.size()) # [72, 1255]
        h = torch.mm(features, self.W)
        # print('h',h.size()) # [72, 64]
        compare = torch.cat([h[adj[0]], h[adj[1]]], dim=1)
        # print('compare',compare.size())# [72,128]
        e = self.leakyrelu(torch.matmul(compare, self.a).squeeze(1))
        # print('e',e.size())#[72]
        # 节点聚合的权重参数维度是
        attention = torch.full((target_len, neighbor_len), -9e15).to(self.device)
        # print('attention',attention.size())#[8, 72]
        attention[target_index_out, adj[1]] = e
        # print('attention',attention.size())#[8, 72]
        attention = F.softmax(attention, dim=1)
        # print('attention',attention.size())#[8, 72]
        attention = F.dropout(attention, self.dropout, training=self.training)
        # print('attention',attention.size())#[8, 72]
        h_prime = torch.matmul(attention, h)
        # print('h_prime',h_prime.size())#[8, 64]
        return F.elu(h_prime)
    # 通过attention聚合邻居特征得到中心节点的特征/伪标签。这个伪标签用于求lmmd→
    
class HomoAttModel(nn.Module):
    # nfeat, nhid, dropout, alpha, device, nheads, nlayer, neigh_por
    def __init__(self, in_dim, out_dim, dropout, alpha, device, nheads, nlayer, neigh_por):
        super(HomoAttModel, self).__init__()
        
        self.neigh_por = neigh_por
        self.nlayer = nlayer
        self.dropout = dropout
        
        self.homo_atts = []
        for i in range(nlayer):
            # print('50iiiiiiiiii',i)#0
            if i==0: curr_in_dim = in_dim
            else: curr_in_dim = out_dim*nheads[i-1]
            # print('in_dim',in_dim)#1255
            # print('curr_in_dim',curr_in_dim)#1255
            # print('out_dim',out_dim)#64
            # print('nheads[i]',nheads[i])#8

            layer_homo_atts = []
            
            for j in range(nheads[i]):#8
                layer_homo_atts.append(HomoAttLayer(curr_in_dim, out_dim, dropout, alpha, device).to(device))
                self.add_module('homo_atts_layer{}_head{}'.format(i,j), layer_homo_atts[j])
            self.homo_atts.append(layer_homo_atts)
            self.linear_block = nn.Sequential(nn.Linear(out_dim*nheads[-1], out_dim*nheads[-1]), nn.Tanh()).to(device) 
                
    def sample(self, adj, samples):#adjsS[i], curr_batchS

        # print('adj',adj)
        # print('samples',samples)
        sample_list, adj_list = [samples], []
        # print('sample_list',sample_list)
        # print('adj_list',adj_list)
        '''
        adj (array([  58,  126,  177, ..., 3208, 3246, 3564]), array([17,  3, 14, ...,  3,  0,  0]))
samples [ 202  322  557  869 1200 1477 1492 1624 2118 2618 2674 2686 2840 3104
 3237 3760]
sample_list [array([ 202,  322,  557,  869, 1200, 1477, 1492, 1624, 2118, 2618, 2674,
       2686, 2840, 3104, 3237, 3760])]
adj_list []
        '''
        for _ in range(self.nlayer):
            new_samples, new_adjs = set(sample_list[-1]), []
            for sample in sample_list[-1]:
                neighbor_size = adj[1][sample]
                nneighbor = int(self.neigh_por*neighbor_size)+1
                start = adj[1][:sample].sum()
                
                if neighbor_size<=nneighbor:
                    curr_new_samples = adj[0][start:start+neighbor_size]   
                else:
                    curr_new_samples = random.sample(adj[0][start:start+neighbor_size].tolist(), nneighbor)
                new_samples = new_samples.union(set(curr_new_samples))
                curr_new_adjs = np.stack(([sample]*len(curr_new_samples), curr_new_samples), axis=-1).tolist()
                curr_new_adjs.append([sample, sample])
                new_adjs.append(curr_new_adjs)

            sample_list.append(np.array(list(new_samples)))
            adj_list.append(np.array([pair for chunk in new_adjs for pair in chunk]).T)
            # print('sample_list',sample_list)
            # print('adj_list',adj_list)
        return sample_list, adj_list
    
    def transform(self, sample_list, adj_list):
        
        trans_adj_list, target_index_outs = [], []
        
        base_index_dict = {k:v for v,k in enumerate(sample_list[0])}        
        for i, adjs in enumerate(adj_list):
            target_index_outs.append([base_index_dict[k] for k in adjs[0]])
            base_index_dict = {k:v for v,k in enumerate(sample_list[i+1])}
            neighbor_index_out, neighbor_index_in = [base_index_dict[k] for k in adjs[0]], [base_index_dict[k] for k in adjs[1]]
            trans_adj_list.append([neighbor_index_out, neighbor_index_in])            
            
        return target_index_outs, trans_adj_list
    
    def forward(self, feats, adj, samples):#embeddingsS, adjsS[i], curr_batchS
        
        sample_list, adj_list = self.sample(adj, samples)
        target_index_outs, trans_adj_list = self.transform(sample_list, adj_list)
        
        x = feats[sample_list[-1]]
        # print('x',x)
        for i, layer_homo_atts in enumerate(self.homo_atts):
            # print('i',i)
            # print('layer_homo_atts',layer_homo_atts)
            x = F.dropout(x, self.dropout, training=self.training)
            x = torch.cat([att(x, trans_adj_list[-i-1], len(sample_list[-i-2]), len(sample_list[-i-1]), target_index_outs[-i-1]) for att in layer_homo_atts], dim=1)
        # print('x',x)
        x = self.linear_block(x)
        return x
    
    
class HeteroAttLayer(nn.Module):
    # nchannel, nhid*nheads[-1], nhid, device, dropout
    def __init__(self, nchannel, in_dim, att_dim, device, dropout):
        super(HeteroAttLayer, self).__init__()
        
        self.nchannel = nchannel
        self.in_dim = in_dim
        self.att_dim = att_dim
        self.device = device
        # print('in_dim',in_dim)#512=64*8
        # print('att_dim',att_dim)#64
        self.meta_att = nn.Parameter(torch.zeros(size=(nchannel, att_dim)))#2*64
        nn.init.xavier_uniform_(self.meta_att.data, gain=1.414)
        
        self.linear_block2 = nn.Sequential(nn.Linear(att_dim, att_dim), nn.Tanh())

    def forward(self, hs, nnode):#homo_out, len(samples)
        #hs: [2, 128, 512]
        #----------------------
        # new_hs = torch.cat([self.linear_block(hs[i]).view(1,nnodLKe,-1) for i in range(self.nchannel)], dim=0)
        hs = torch.cat([self.linear_block2(hs[i]).view(1,nnode,-1) for i in range(self.nchannel)], dim=0)
        # print('new_hs',new_hs)
        # print('new_hs.size()',new_hs.size())#[2, 16, 64]2个元路径，16个点，每个点维度64
        meta_att = []
        for i in range(self.nchannel):
            # print('self.meta_att[i]',self.meta_att[i])
            # print('self.meta_att[i].view(-1,1)',self.meta_att[i].view(-1,1))
            meta_att.append(torch.sum(torch.mm(hs[i], self.meta_att[i].view(-1,1)).squeeze(1)) / nnode)
            # print('meta_att',meta_att)
        # print('meta_att',meta_att)
        meta_att = torch.stack(meta_att, dim=0)
        # print('meta_att',meta_att)
        meta_att = F.softmax(meta_att, dim=0)
        # print('meta_att',meta_att)
        aggre_hid = []
        # print('nnode',nnode)
        for i in range(nnode):
            aggre_hid.append(torch.mm(meta_att.view(1,-1), hs[:,i,:]))
        aggre_hid = torch.stack(aggre_hid, dim=0).view(nnode, self.att_dim)#16*64
        # print('aggre_hid',aggre_hid)
        return aggre_hid
    
    
class HANModel(nn.Module):
    # nhid = args.size
    def __init__(self, nchannel, nfeat, nhid, nlabel, nlayer, nheads, neigh_por, dropout, alpha, device):
        super(HANModel, self).__init__()

        self.HomoAttModels = [HomoAttModel(nfeat, nhid, dropout, alpha, device, nheads, nlayer, neigh_por) for i in range(nchannel)]
        # self.HeteroAttLayer = HeteroAttLayer(nchannel, nhid*nheads[-1], nhid, device, dropout).to(device)        
        
        for i, homo_att in enumerate(self.HomoAttModels):
            self.add_module('homo_att_{}'.format(i), homo_att)
        # self.add_module('hetero_att', self.HeteroAttLayer)
        
        # self.LinearLayer = torch.nn.Linear(nhid, nlabel).to(device)
        # self.add_module('linear', self.LinearLayer)
        
    def forward(self, x, adjs, samples):#embeddingsS, adjsS, curr_batchS
        
        homo_out = []
        for i, homo_att in enumerate(self.HomoAttModels):
            homo_out.append(homo_att(x, adjs[i], samples))
        homo_out = torch.stack(homo_out, dim=0)
        # aggre_hid = self.HeteroAttLayer(homo_out, len(samples))
        
        # if self.supervised:
        #     pred = self.LinearLayer(aggre_hid)
        # else:
        #     pred = None
        
        return homo_out

class RevGrad(nn.Module):

    def __init__(self,nchannel, nfeat, nhid, nlabel, nlayer, nheads, neigh_por, dropout, alpha, device):
        super(RevGrad, self).__init__()
        self.nchannel = nchannel
    # nchannel, nfeat , args.size, nlabelS, nlayer, nhead, args.neigh_por, args.dropout, args.alpha, args.device
        self.sharedNet = resnet50(nchannel, nfeat, nhid, nlabel, nlayer, nheads, neigh_por, dropout, alpha, device,pretrained=False)
        # 这层是否共享
        self.linear_block = nn.Sequential(nn.Linear(nhid*nheads[-1], nhid), nn.Tanh()).to(device)

        # self.HeteroAttLayerS = HeteroAttLayer(nchannel, nhid*nheads[-1], nlabel, device, dropout).to(device)
        self.HeteroAttLayerT = HeteroAttLayer(nchannel, nhid*nheads[-1], nlabel, device, dropout).to(device)
        # self.add_module('hetero_att', self.HeteroAttLayer)
        # self.add_module('hetero_attT', self.HeteroAttLayerT)

        self.cls_fc1 = torch.nn.Linear(nhid, nlabel).to(device)
        self.cls_fc2 = torch.nn.Linear(nhid, nlabel).to(device)
        # self.cls_fc3 = torch.nn.Linear(nhid, nlabel).to(device)
        # self.cls_fcF = torch.nn.Linear(nhid, nlabel).to(device)
        # self.domain_fc = nn.Linear(nhid, 2).to(device)

    def forward(self, xS, adjsS, samplesS, xT, adjsT, samplesT, S_label,Training, mark):#embeddingsS, adjsS, curr_batchS
        mmd_loss=0
        if Training==True:
            if mark == 0:
                homo_outS = self.sharedNet(xS, adjsS, samplesS)       
                homo_outT = self.sharedNet(xT, adjsT, samplesT)
                new_hsS = torch.cat([self.linear_block(homo_outS[i]).view(1,len(samplesS),-1) for i in range(self.nchannel)], dim=0)
                new_hsT = torch.cat([self.linear_block(homo_outT[i]).view(1,len(samplesT),-1) for i in range(self.nchannel)], dim=0)
                clabel_predS1 = self.cls_fc1(new_hsS[mark])

                clabel_predT1 = self.cls_fc1(new_hsT[mark])

                mmd_loss += mmd.lmmd(homo_outS[mark], homo_outT[mark], S_label, torch.nn.functional.softmax(clabel_predT1, dim=1))
                
                clabel_predT2 = self.cls_fc2(new_hsT[mark])
                # clabel_predT3 = self.cls_fc3(new_hsT[mark])

                clabel_predS2 = self.cls_fc2(new_hsS[mark])
                # clabel_predS3 = self.cls_fc3(new_hsS[mark])

                # l1_loss = torch.mean(torch.abs(torch.nn.functional.softmax(clabel_predT1, dim=1)
                #                                 - torch.nn.functional.softmax(clabel_predT2, dim=1)) )


                l1_loss = torch.mean( torch.abs(torch.nn.functional.softmax(clabel_predS1, dim=1)
                                                - torch.nn.functional.softmax(clabel_predS2, dim=1)) )
                # l1_loss += torch.mean( torch.abs(torch.nn.functional.softmax(clabel_predS1, dim=1)
                                                # - torch.nn.functional.softmax(clabel_predS3, dim=1)) )

                cls_lossS_MP = F.nll_loss(F.log_softmax(clabel_predS1, dim=1), S_label.long())

                tworeS = torch.cat((clabel_predS1.view(1,len(samplesS),-1),clabel_predS2.view(1,len(samplesS),-1)), dim=0)
                clabel_predS = self.HeteroAttLayerT(tworeS, len(samplesS))
                cls_lossS = F.nll_loss(F.log_softmax(clabel_predS, dim=1), S_label.long())

            if mark == 1:
                homo_outS = self.sharedNet(xS, adjsS, samplesS)       
                homo_outT = self.sharedNet(xT, adjsT, samplesT)
                new_hsS = torch.cat([self.linear_block(homo_outS[i]).view(1,len(samplesS),-1) for i in range(self.nchannel)], dim=0)
                new_hsT = torch.cat([self.linear_block(homo_outT[i]).view(1,len(samplesT),-1) for i in range(self.nchannel)], dim=0)
                clabel_predS2 = self.cls_fc2(new_hsS[mark])
                clabel_predT2 = self.cls_fc2(new_hsT[mark])

                mmd_loss += mmd.lmmd(homo_outS[mark], homo_outT[mark], S_label, torch.nn.functional.softmax(clabel_predT2, dim=1))
                clabel_predT1 = self.cls_fc1(new_hsT[mark])
                # clabel_predT3 = self.cls_fc3(new_hsT[mark])

                clabel_predS1 = self.cls_fc1(new_hsS[mark])
                # clabel_predS3 = self.cls_fc3(new_hsS[mark])

                # l1_loss = torch.mean(torch.abs(torch.nn.functional.softmax(clabel_predT2, dim=1)
                #                                 - torch.nn.functional.softmax(clabel_predT1, dim=1)) )
                # l1_loss += torch.mean( torch.abs(torch.nn.functional.softmax(clabel_predT2, dim=1)
                #                                 - torch.nn.functional.softmax(clabel_predT3, dim=1)) )

                l1_loss = torch.mean( torch.abs(torch.nn.functional.softmax(clabel_predS1, dim=1)
                                                - torch.nn.functional.softmax(clabel_predS2, dim=1)) )
                # l1_loss += torch.mean( torch.abs(torch.nn.functional.softmax(clabel_predS1, dim=1)
                #                                 - torch.nn.functional.softmax(clabel_predS3, dim=1)) )

                cls_lossS_MP = F.nll_loss(F.log_softmax(clabel_predS2, dim=1), S_label.long())
                tworeS = torch.cat((clabel_predS1.view(1,len(samplesS),-1),clabel_predS2.view(1,len(samplesS),-1)), dim=0)
                clabel_predS = self.HeteroAttLayerT(tworeS, len(samplesS))
                cls_lossS = F.nll_loss(F.log_softmax(clabel_predS, dim=1), S_label.long())
           
            if mark == -1 :
                # homo_outS = self.sharedNet(xS, adjsS, samplesS)       
                homo_outT = self.sharedNet(xT, adjsT, samplesT)
                # new_hsS = torch.cat([self.linear_block(homo_outS[i]).view(1,nnode,-1) for i in range(self.nchannel)], dim=0)
                new_hsT = torch.cat([self.linear_block(homo_outT[i]).view(1,len(samplesT),-1) for i in range(self.nchannel)], dim=0)
                clabel_predT1 = self.cls_fc1(new_hsT[0])
                clabel_predT2 = self.cls_fc2(new_hsT[1])
                # clabel_predT3 = self.cls_fc3(new_hsT[2])
                mmd_loss = 0
                l1_loss = 0
                cls_lossS = 0
                cls_lossS_MP = 0

                # homo_outS = self.sharedNet(xS, adjsS, samplesS)
                # # new_hsS = torch.cat([self.linear_block(homo_outS[i]).view(1,nnode,-1) for i in range(self.nchannel)], dim=0)
                # new_hsS = torch.cat([self.linear_block(homo_outS[i]).view(1,len(samplesS),-1) for i in range(self.nchannel)], dim=0)
                # clabel_predS1 = self.cls_fc1(new_hsS[0])
                # clabel_predS2 = self.cls_fc2(new_hsS[1])
                # clabel_predS3 = self.cls_fc3(new_hsS[2])

                # loss_mmd += mmd.lmmd(new_hsS[mark], new_hsT[mark], S_label, torch.nn.functional.softmax(clabel_predT1, dim=1))
                # clabel_predT2 = self.cls_fc1(new_hsT[mark])

                # l1_loss = torch.abs(torch.nn.functional.softmax(clabel_predT1, dim=1) - torch.nn.functional.softmax(clabel_predT2, dim=1))
                # l1_loss = torch.mean(l1_loss)

                # cls_loss = F.nll_loss(F.log_softmax(clabel_predS, dim=1), S_label)
            
            target_probs1 = F.softmax(clabel_predT1, dim=-1)
            target_probs1 = torch.clamp(target_probs1, min=1e-9, max=1.0)
            target_label1 = target_probs1.data.max(1)[1]

            target_probs2 = F.softmax(clabel_predT2, dim=-1)
            target_probs2 = torch.clamp(target_probs2, min=1e-9, max=1.0)
            target_label2 = target_probs2.data.max(1)[1]


            choose = (target_label1==target_label2).cpu().numpy().tolist()
            # print('===---------choose---------',sum(choose))
            twore = torch.cat((clabel_predT1.view(1,len(samplesT),-1),clabel_predT2.view(1,len(samplesT),-1)), dim=0)
            clabel_predF = self.HeteroAttLayerT(twore, len(samplesT))
            # clabel_predF = self.cls_fcF(feadataT)

            # print('clabel_predF',clabel_predF.size())
            # print('target_label1',target_label1)

            # cls_lossT = F.nll_loss(F.log_softmax(clabel_predF[choose], dim=1), target_label1[choose].long())
            # cls_lossS=0
            target_probs = F.softmax(clabel_predF, dim=-1)
            target_probs = torch.clamp(target_probs, min=1e-9, max=1.0)

            cls_lossT = torch.mean(torch.sum(-target_probs * torch.log(target_probs), dim=-1))
            
            # clabel_predF = self.cls_fcF(feadataT)

            # cls_lossTS = F.nll_loss(F.log_softmax(clabel_predFS[choose], dim=1), target_label1[choose].long())
            # print('clabel_predS',clabel_predS.size())
            # print('S_label',S_label)

            
            # print('homo_outS[i]',homo_outS[i].size())
            # print('homo_outT[i]',homo_outS[i].size())
        # print('S_label',S_label.size())
        # print('clabel_predT',clabel_predT.size())
        # aa  = torch.nn.functional.softmax(clabel_predT, dim=1)
        # print('aa',aa.size())
        # exit()

        return cls_lossS, cls_lossT, mmd_loss, l1_loss, clabel_predF, cls_lossS_MP

def resnet50(nchannel, nfeat, nhid, nlabel, nlayer, nheads, neigh_por, dropout, alpha, device,pretrained=False):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = HANModel(nchannel, nfeat, nhid, nlabel, nlayer, nheads, neigh_por, dropout, alpha, device)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        pretrained_dict = torch.load('/home/zyx/Fighting/Model/HAN/src/pthacm/ACMrevpth30_45.pth')
        model.load_state_dict(pretrained_dict, strict=False)
    return model