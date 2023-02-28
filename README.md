# MKDN

Code for 


### DataSets

*DataSets* prepares 3 input files stored in ```Model/HAN/data/```:
For ACM and DBlP datasets:
- ```node.dat```: For attributed training, each line is formatted as ```{node_id}\t{node_type}\t{node_attributes}``` where entries in ```{node_attributes}``` are separated by ```,```. For unattributed training, each line is formatted as ```{node_id}\t{node_type}```.
- ```link.dat```: Each line is formatted as ```{head_node_id}\t{tail_node_id}\t{link_type}```.
- ```config.dat```: The first line specifies the targeting node type. The second line specifies the targeting link type. The third line specifies the information related to each link type, e.g., ```{head_node_type}\t{tail_node_type}\t{link_type}```.
- ```labelall.dat```:  Each line is formatted as ```{node_id}\t{node_label}```.
For IMDB dataset:
- ```feature_x.npy```: Feature matrix
- ```MAM_x.npy/MDM_x.npy```: Meta-path adjacency matrix
- ```label_x.dat```: Each line is formatted as ```{node_id}\t{node_label}```

### Run
*Step 1:pre-tarin*
Users need to specify the targeting dataset 
and the set of training parameters in ```Model/HAN/run.sh```. <br /> 
Run ```bash run.sh``` to start pre-training.
In addation, users are supposed to create a folder to save the pretrained models
The pretrained model will be saved in e.g.```Model/HAN/DBLP_ABtoC```.

*Step 2:Distill*
For Node Distillation:
Users need to specify the targeting dataset 
and the set of training parameters in ```Model/HAN/LoopmainDistill.sh```. <br /> 
Run ```bash LoopmainDistill.sh``` to start distilling.
The distilled model will be saved in ```Model/HAN/...```.

For Meta-path Distillation:
Users need to specify the targeting dataset 
and the set of training parameters in ```Model/HAN/path_distill.sh```. <br /> 
In addation, users are supposed to specify listrank_x.dat in ```Model/HAN/src/utils.py``` 
corresponding to training parameters.
Run ```bash path_distill.sh``` to start distilling.
The distilled model will be saved in ```Model/HAN/...```.

*Step 3:Aggregate output*
Users need to specify the targeting dataset 
and the set of training parameters in ```Model/HAN/multiRun.sh```. <br /> 
Run ```bash multiRun.sh``` to start aggregating.