import numpy as np

def euclidean_dist(v1, v2):
    return np.sqrt( np.sum( np.power((v1-v2),2)  ) )

def  manhattten_dist(v1, v2):
    return np.sum(np.abs(v1-v2))


#===============================================================================
# simple test
# x = np.matrix((1,1,1,1))
# y = np.matrix((3,3,3,3))
# 
# print(euclidean_dist(x, y))
#===============================================================================