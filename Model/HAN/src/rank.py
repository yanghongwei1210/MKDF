import os
import numpy as np
from scipy.stats import wasserstein_distance

# Purpose: To sort the nodes of S
# Load Source
id_namesS = []
featuresS = []
with open('/Users/donvictor/Desktop/tri/DBLP/DBLP_c/node_c.dat','r') as file:
	for line in file:
		nid, _, feature = line[:-1].split('\t')
		id_namesS.append(nid)
		featuresS.append(np.array(feature.split(',')).astype(np.float32))
file.close()
# print("featuresS:",featuresS)
# Load target
id_namesT = []
featuresT = []
with open('/Users/donvictor/Desktop/tri/DBLP/DBLP_a/node_a.dat','r') as file:
	for line in file:
		nid, _, feature = line[:-1].split('\t')
		id_namesT.append(nid)
		featuresT.append(np.array(feature.split(',')).astype(np.float32))
file.close()
print("featuresT:",featuresT)

# Calculate the distance from each node of S to T
emd = np.zeros(len(featuresS))

concat_featuresT = np.concatenate(featuresT)
for i in range(len(featuresS)):
# for i in range(10):
	print(str(i) + "th LOOP")
	emd[i] = wasserstein_distance(featuresS[i],concat_featuresT)
	
emd_dict = {}
for i in range(len(featuresS)):
	emd_dict[id_namesS[i]] = emd[i]
print(type(emd_dict))
sorted_dict = dict(sorted(emd_dict.items(),key=lambda x:x[1]))
print(type(sorted_dict))
with open('/Users/donvictor/Desktop/tri/DBLP/BCtoA/listrank_BCtoA_DBLP_c.dat','w') as file:
	for k,v in sorted_dict.items():
		file.write(k + "\n")
file.close()
print(emd_dict)






