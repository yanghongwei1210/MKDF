#!/bin/bash
# dataset = "DBLP" 

datasetS="DBLP_a"  # Source Domain
folderS="data/${datasetS}/"
node_fileS="${folderS}node.dat"
config_fileS="${folderS}config.dat"
link_fileS="${folderS}link.dat"
label_fileS="${folderS}labelall.dat"
emb_fileS="${folderS}emb.dat"

datasetT="DBLP_c"  # Target Domain
folderT="data/${datasetT}/"
node_fileT="${folderT}node.dat"
config_fileT="${folderT}config.dat"
link_fileT="${folderT}link.dat"
label_fileT="${folderT}labelall.dat"
emb_fileT="${folderT}emb.dat"

# meta="1,2,4,8" # Choose the meta-paths used for training. Suppose the targeting node type is 1 and link type 1 is between node type 0 and 1, then meta="1" means that we use meta-paths "101".
metaS="1,2,3"
metaT="1,2,3"

size=64
nhead="8"
dropout=0.4
neigh_por=0.6
lr=0.01
weight_decay=0.0005
batch_size=256
epochs=50
device="cuda"

attributed="True"
supervised="True"

model='HAN' # choose from 'metapath2vec-ESim', 'PTE', 'HIN2Vec', 'AspEm', 'HEER', 'R-GCN', 'HAN', 'HGT', 'TransE', 'DistMult', and 'ConvE'
task='nc' # choose 'nc' for node classification, 'lp' for link prediction, or 'both' for both tasks
# attributed='True' # choose 'True' or 'False'
# supervised='True' # choose 'True' or 'False'

python3 src/mainDistill.py --nodeS=${node_fileS} --linkS=${link_fileS} --configS=${config_fileS} --labelS=${label_fileS} --outputS=${emb_fileS} --nodeT=${node_fileT} --linkT=${link_fileT} --configT=${config_fileT} --labelT=${label_fileT} --outputT=${emb_fileT} --device=${device} --metaS=${metaS} --metaT=${metaT} --size=${size} --nhead=${nhead} --dropout=${dropout} --neigh-por=${neigh_por} --lr=${lr} --weight-decay=${weight_decay} --batch-size=${batch_size} --epochs=${epochs} --attributed=${attributed} --supervised=${supervised} --datasetS ${datasetS} --datasetT ${datasetT} --model ${model} --task ${task}