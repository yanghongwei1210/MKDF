import gc
import random
import numpy as np
from collections import defaultdict

import torch


def convert(posi, nega, posi_size, nega_size, batch_size):
    
    posi = posi[np.random.randint(posi_size, size=batch_size), :]
    nega = nega[np.random.randint(nega_size, size=batch_size), :]
    
    seeds = set()
    for each in posi.flatten():
        seeds.add(each)
    for each in nega.flatten():
        seeds.add(each)
    seeds = np.sort(list(seeds))
    
    index_dict = {k:v for v,k in enumerate(seeds)}
    indices = np.array([index_dict[k] for k in seeds])
    
    new_posi, new_nega = [], []
    for (pleft, pright), (nleft, nright) in zip(posi, nega):
        new_posi.append([index_dict[pleft], index_dict[pright]])
        new_nega.append([index_dict[nleft], index_dict[nright]])
    
    return seeds, indices, np.array(new_posi), np.array(new_nega)
        

def set_seed(seed, device):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device=="cuda":
        torch.cuda.manual_seed(seed)


def sample(target_pool, positive_edges):
    
    positive_pool = set()
    for edge in positive_edges:
        positive_pool.add(tuple(sorted(edge)))
        
    negative_edges = set()
    positive_count, negative_count = positive_edges.shape[0], 0
    while negative_count < positive_count:
        nega_left, nega_right = np.random.choice(list(target_pool), size=positive_count-int(negative_count/2), replace=True), np.random.choice(list(target_pool), size=positive_count-int(negative_count/2), replace=True)
        for each_left, each_right in zip(nega_left, nega_right):
            if each_left==each_right: continue
            if tuple(sorted([each_left, each_right])) in positive_pool: continue
            if (each_left, each_right) in negative_edges: continue
            negative_edges.add((each_left, each_right))
            negative_count += 1
            if negative_count >= positive_count: break
                
    negative_edges = np.array(list(negative_edges)).astype(np.int32)
    return negative_edges    


def load_data_unsupervised(args, node, edge, config, meta):
    print('check 0', flush=True)
    lines = open(config, 'r').readlines()
    target, positive_type = int(lines[0][:-1]), int(lines[1][:-1])
    useful_types, positive_same, positive_edges = set(), False, []
    for each in lines[2].split('\t'):
        start, end, ltype = each.split(',')
        start, end, ltype = int(start), int(end), int(ltype)
        if ltype in meta:
            useful_types.add(ltype)
        if ltype==positive_type and start==target and end==target:
            positive_same = True
    print('check 1', flush=True)
    id_inc, id_name, name_id, name_attr = 0, {}, {}, {}
    with open(node, 'r') as file:
        for line in file:
            if args.attributed=='True': nid, ntype, attr = line[:-1].split('\t')
            elif args.attributed=='False': nid, ntype = line[:-1].split('\t')
            nid, ntype = int(nid), int(ntype)
            if ntype==target:
                name_id[nid] = id_inc
                id_name[id_inc] = nid
                if args.attributed=='True': name_attr[nid] = np.array(attr.split(',')).astype(np.float32)
                id_inc += 1
    print('check 2', flush=True)
    type_corners = {ltype:defaultdict(set) for ltype in useful_types}
    with open(edge, 'r') as file:
        for line in file:
            start, end, ltype = line[:-1].split('\t')
            start, end, ltype = int(start), int(end), int(ltype)
            if ltype in useful_types:
                if start in name_id:
                    type_corners[ltype][end].add(name_id[start])
                if end in name_id:
                    type_corners[ltype][start].add(name_id[end])
            if ltype==positive_type and positive_same:
                positive_edges.append([name_id[start], name_id[end]])
        if positive_same:
            positive_edges = np.array(positive_edges).astype(np.int32)
    
    print('check 3', flush=True)            
    adjs = []
    for ltype in useful_types:
        corners = type_corners[ltype]
        two_hops = defaultdict(set)
        for _, neighbors in corners.items():
            for snode in neighbors:
                for enode in neighbors:
                    if snode!=enode:
                        two_hops[snode].add(enode)
        print('check 3.1', flush=True)
        rights, counts = [], np.zeros(id_inc).astype(int)
        for i in range(id_inc):
            if i in two_hops:
                current = np.sort(list(two_hops[i]))
                rights.append(current)
                counts[i] = len(current)
        adjs.append((np.concatenate(rights), counts))  
        print('check 3.2', flush=True)
        if ltype==positive_type and not positive_same:
            for _, neighbors in corners.items():
                for snode in neighbors:
                    for enode in neighbors:
                        positive_edges.append([snode, enode])
            positive_edges = np.array(positive_edges).astype(np.int32)
        del two_hops, rights, counts, type_corners[ltype]
        gc.collect()
        print('check 3.3', flush=True)
    print('check 4', flush=True)
    
    if args.attributed=="True": name_attr = np.array([name_attr[id_name[i]] for i in range(len(id_name))]).astype(np.float32)
    
    return adjs, id_name, set(range(id_inc)), positive_edges, name_attr


def load_label(label_path, id_name):
    
    name_id, id_label, all_labels = {v:k for k,v in id_name.items()}, {}, set()
    # print(name_id)#???node??????????????????node1????????????
    
    train_set, multi = set(), False
    with open(label_path, 'r') as file:
        for line in file:
            node, label = line[:-1].split('\t')
            train_set.add(name_id[int(node)])#?????????????????????node1????????????
            if multi or ',' in label:
                multi = True
                label_array = np.array(label.split(',')).astype(int)
                for each in label_array:
                    all_labels.add(each)
                id_label[name_id[int(node)]] = label_array#?????????????????????node1?????????????????????
            else:
                all_labels.add(int(label))
                id_label[name_id[int(node)]] = int(label)            
    train_pool = np.sort(list(train_set))
    
    train_label = []
    for k in train_pool:
        if multi:
            curr_label = np.zeros(len(all_labels)).astype(int)
            curr_label[id_label[k]] = 1
            train_label.append(curr_label)
        else:
            train_label.append(id_label[k])
    train_label = np.array(train_label)
##?????????????????????node1???????????????????????????????????????????????????
    return train_pool, train_label, len(all_labels), multi


def load_data_semisupervised(args, node, edge, config, meta):
    
    print('check 0', flush=True)
    lines = open(config, 'r').readlines()
    target, useful_types = int(lines[0][:-1]), set()

    for each in lines[2].split('\t'):
        start, end, ltype = each.split(',')
        start, end, ltype = int(start), int(end), int(ltype)
        if ltype in meta:
            useful_types.add(ltype)
    # print(useful_types)
    # exit()
    print('check 1', flush=True)
    id_inc, id_name, name_id, name_attr = 0, {}, {}, {}
    with open(node, 'r') as file:
        for line in file:
            nid, ntype, attr = line[:-1].split('\t')
            nid, ntype = int(nid), int(ntype)
            if ntype==target:
                name_id[nid] = id_inc
                id_name[id_inc] = nid
                if args.attributed=='True': name_attr[nid] = np.array(attr.split(',')).astype(np.float32)
                id_inc += 1
                
    print('check 2', flush=True)
    type_corners = {ltype:defaultdict(set) for ltype in useful_types}
    # iii=0
    with open(edge, 'r') as file:
        for line in file:
            # iii=iii+1
            start, end, ltype = line[:-1].split('\t')
            start, end, ltype = int(start), int(end), int(ltype)
            if ltype in useful_types:
                if start in name_id: # name_id:{9:0,10:1,...:...}
                    type_corners[ltype][end].add(name_id[start])
                    # print('1,,,',type_corners)
                if end in name_id: 
                    type_corners[ltype][start].add(name_id[end])
                    # print('\n2,,,',type_corners)
    # print(type_corners)
    # exit()
    
    print('check 3', flush=True)            
    adjs = []
    for ltype in useful_types:
        corners = type_corners[ltype]
        two_hops = defaultdict(set)
        # for _, neighbors in corners.items():
        for yyy, neighbors in corners.items():
            # print('yyy',yyy)
            # print('neighbors',neighbors)
            for snode in neighbors:
                # print('snode',snode)
                for enode in neighbors:
                    # print('enode',enode)
                    if snode!=enode:
                        two_hops[snode].add(enode)
        # print('4174',two_hops[4174])
        # print('4175',two_hops[4175])
        # print('check 3.1', flush=True)
        rights, counts = [], np.zeros(id_inc).astype(int)
        for i in range(id_inc):
            if i in two_hops:
                current = np.sort(list(two_hops[i]))
                # print('current',current)
                rights.append(current)
                # print('rights',rights)
                counts[i] = len(current)
                # print('counts',counts)
        adjs.append((np.concatenate(rights), counts))
        # print('check 3.2', flush=True)
        del two_hops, rights, counts, type_corners[ltype]
        gc.collect()
        # print('check 3.3', flush=True)
    print('check 4', flush=True)
    # print('adjs',adjs)
    # print('adjs[0]',adjs[0])
    # print('adjs[0][0]',adjs[0][0])
    # exit()
    if args.attributed=="True": name_attr = np.array([name_attr[id_name[i]] for i in range(len(id_name))]).astype(np.float32)
    return adjs, id_name, name_attr
#adjs:[[node1???????????????????????????][node1????????????????????????????????????],[??????????????????][]]
    # id_name:{0:9,1:10,2:...}



def load_data_distill(isattributed, node, edge, config, meta,distillRate):
    
    
    lines = open(config, 'r').readlines()
    target, useful_types = int(lines[0][:-1]), set()

    for each in lines[2].split('\t'):
        start, end, ltype = each.split(',')
        start, end, ltype = int(start), int(end), int(ltype)
        if ltype in meta:
            useful_types.add(ltype)
    # print(useful_types)
    # exit()
    id_inc, id_name, name_id, name_attr = 0, {}, {}, {}
    with open(node, 'r') as file:
        for line in file:
            nid, ntype, attr = line[:-1].split('\t')
#             if nid in distillNode:
#                 continue
            nid, ntype = int(nid), int(ntype)
            if ntype==target:
                name_id[nid] = id_inc
                id_name[id_inc] = nid
                name_attr[nid] = np.array(attr.split(',')).astype(np.float32)
                id_inc += 1
                
    type_corners = {ltype:defaultdict(set) for ltype in useful_types}
    # iii=0
    with open(edge, 'r') as file:
        for line in file:
            # iii=iii+1
            start, end, ltype = line[:-1].split('\t')
            start, end, ltype = int(start), int(end), int(ltype)
            if ltype in useful_types:
                if start in name_id: # name_id:{9:0,10:1,...:...}
                    type_corners[ltype][end].add(name_id[start])
                    # print('1,,,',type_corners)
                if end in name_id: 
                    type_corners[ltype][start].add(name_id[end])
                    # print('\n2,,,',type_corners)
    # print(type_corners)
    # exit()
    
    rnode = []
    # Read the sorted node file
    with open ('/home/user/??????/Ljx/MKDF/Model/HAN/data/node-label/DBLP/ABtoC/listrank_ABtoC_DBLP_a.dat','r') as file:
        for line in file:
            rnode.append(int(line))
        # Select a percentage of nodes to be distilled
    r = distillRate # Indicates that the lower 0.1 of the ranking is removed
    rnode = rnode[::-1]
    distillNode = rnode[0:int(len(rnode)*r)]    
    
    
    adjs = []
    for ltype in useful_types:
        corners = type_corners[ltype]
        two_hops = defaultdict(set)
        # for _, neighbors in corners.items():
        for yyy, neighbors in corners.items():
            # print('yyy',yyy)
            # print('neighbors',neighbors)
            for snode in neighbors:
                # print('snode',snode)
                for enode in neighbors:
                    # print('enode',enode)
                    if snode!=enode:
                        two_hops[snode].add(enode)
                        
                        
        # print('4174',two_hops[4174])
        # print('4175',two_hops[4175])
        rights, counts = [], np.zeros(id_inc).astype(int)
        for i in range(id_inc):
            if i in two_hops:
                current = np.sort(list(two_hops[i]))
                # print('current',current)
                rights.append(current)
                # print('rights',rights)
                counts[i] = len(current)
                # print('counts',counts)
        
        # concat_rights = np.concatenate(rights)
        # print("concat_rights:",len(concat_rights))
        # start = 0
        # for i in range(len(counts)):
        #     end = start + counts[i]
        #     # print("start:" + str(start) + ", end:" +str(end))
        #     # print("len concat_rights:",len(concat_rights))
        #     deleteNode = []
        #     for j in range(start,end):
        #         if id_name[concat_rights[j]] in distillNode:
        #             
        #             deleteNode.append(j)
        #     for k in range(len(deleteNode)):
        #         # if len(concat_rights) < deleteNode[k]:
        #             # print("len concat_rights:",len(concat_rights))
        #             # print("deleteNode[" +str(k) +"] :" + str(deleteNode[k]))
        #         concat_rights = np.delete(concat_rights,deleteNode[k])
        #         
        #         for l in range(k,len(deleteNode)):
        #             deleteNode[l] = deleteNode[l] - 1
        #         counts[i] = counts[i] - 1
        #     start = start + counts[i]
            
        # In the above algorithm, as long as one side of the path meets the requirements, it will be deleted. In the following algorithm, it will be deleted only if both sides meet the requirements.
        concat_rights = np.concatenate(rights)
        start = 0
        for i in range(len(counts)):
            end = start + counts[i]
            # print("start:" + str(start) + ", end:" +str(end))
            # print("len concat_rights:",len(concat_rights))
            deleteNode = []
            # Determine whether the i-th node belongs to the node to be distilled
            if id_name[i] in distillNode:
                for j in range(start,end):
                    if id_name[concat_rights[j]] in distillNode:
                        # np.delete is deleted according to the location, so the location is stored here
                        deleteNode.append(j)
                for k in range(len(deleteNode)):
                    # if len(concat_rights) < deleteNode[k]:
                        # print("len concat_rights:",len(concat_rights))
                        # print("deleteNode[" +str(k) +"] :" + str(deleteNode[k]))
                    concat_rights = np.delete(concat_rights,deleteNode[k])
                    
                    for l in range(k,len(deleteNode)):
                        deleteNode[l] = deleteNode[l] - 1
                    counts[i] = counts[i] - 1
            start = start + counts[i]
            
        print("len concat_rights:",len(concat_rights))                
#         start = 0
#         for i in range(len(counts)):
#             if id_name[i] in distillNode:
#                 end = start +counts[i]
#                 deleteNode = []
#                 for j in range(start,end):
#                     if concat_rights[j] in distillNode:
#                         deleteNode.append(j)
# #                         print("delete:",concat_rights[j])
#                 for k in range(len(deleteNode)):
#                     concat_rights = np.delete(concat_rights,deleteNode[k])
#                     counts[i] = counts[i] - 1
#                 start = start + counts[i]
        
        adjs.append((concat_rights, counts))
        del two_hops, rights, counts, type_corners[ltype]
        gc.collect()
    # print('adjs',adjs)
    # print('adjs[0]',adjs[0])
    # print('adjs[0][0]',adjs[0][0])
    # exit()
    name_attr = np.array([name_attr[id_name[i]] for i in range(len(id_name))]).astype(np.float32)
    return adjs, id_name, name_attr
