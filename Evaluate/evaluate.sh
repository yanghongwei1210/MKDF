#!/bin/bash

# Note: Only 'R-GCN', 'HAN', and 'HGT' support attributed='True' or supervised='True'
# Note: Only 'DBLP' and 'PubMed' support attributed='True'

dataset='ACM' # choose from 'DBLP', 'Yelp', 'Freebase', and 'PubMed'
model='HAN' # choose from 'metapath2vec-ESim', 'PTE', 'HIN2Vec', 'AspEm', 'HEER', 'R-GCN', 'HAN', 'HGT', 'TransE', 'DistMult', and 'ConvE'
task='nc' # choose 'nc' for node classification, 'lp' for link prediction, or 'both' for both tasks
attributed='True' # choose 'True' or 'False'
supervised='True' # choose 'True' or 'False'

python evaluate.py -dataset ${dataset} -model ${model} -task ${task} -attributed ${attributed} -supervised ${supervised}