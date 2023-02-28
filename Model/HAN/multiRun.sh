#!/bin/bash
# dataset = "DBLP" 

datasetS_1="DBLP_a"  # Source Domain 1
# folderS_1="data/ABtoC/ACM_0.1/${datasetS_1}/" 
folderS_1="data/${datasetS_1}/"
node_fileS_1="${folderS_1}node.dat"
config_fileS_1="${folderS_1}config.dat"
link_fileS_1="${folderS_1}link.dat"
label_fileS_1="${folderS_1}labelall.dat"
emb_fileS_1="${folderS_1}emb.dat"

datasetS_2="DBLP_b"  # Source Domain 2
# folderS_2="data/ABtoC/ACM_0.1/${datasetS_2}/" 
folderS_2="data/${datasetS_2}/"
node_fileS_2="${folderS_2}node.dat"
config_fileS_2="${folderS_2}config.dat"
link_fileS_2="${folderS_2}link.dat"
label_fileS_2="${folderS_2}labelall.dat"
emb_fileS_2="${folderS_2}emb.dat"


datasetT="DBLP_c"  # Target Domain 
folderT="data/${datasetT}/"
node_fileT="${folderT}node.dat"
config_fileT="${folderT}config.dat"
link_fileT="${folderT}link.dat"
label_fileT="${folderT}labelall.dat"
emb_fileT="${folderT}emb.dat"

# meta="1,2,4,8" # Choose the meta-paths used for training. Suppose the targeting node type is 1 and link type 1 is between node type 0 and 1, then meta="1" means that we use meta-paths "101".
metaS_1="1,2,3"
metaS_2="1,2,3"
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

python3 src/evaluateMulti.py --nodeS_1=${node_fileS_1} --linkS_1=${link_fileS_1} --configS_1=${config_fileS_1} --labelS_1=${label_fileS_1} --outputS_1=${emb_fileS_1} --nodeS_2=${node_fileS_2} --linkS_2=${link_fileS_2} --configS_2=${config_fileS_2} --labelS_2=${label_fileS_2} --outputS_2=${emb_fileS_2} --nodeT=${node_fileT} --linkT=${link_fileT} --configT=${config_fileT} --labelT=${label_fileT} --outputT=${emb_fileT} --device=${device} --metaS_1=${metaS_1} --metaS_2=${metaS_2} --metaT=${metaT} --size=${size} --nhead=${nhead} --dropout=${dropout} --neigh-por=${neigh_por} --lr=${lr} --weight-decay=${weight_decay} --batch-size=${batch_size} --epochs=${epochs} --attributed=${attributed} --supervised=${supervised} --datasetS_1 ${datasetS_1} --datasetS_2 ${datasetS_2} --datasetT ${datasetT} --model ${model} --task ${task}