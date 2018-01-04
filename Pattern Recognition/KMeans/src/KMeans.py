import numpy as np
import metrices
import time

# 1. Load initial clusters
# 2. set point label to nearest cluster
# 3. set cluster in the mean of assigned clusters.
# to 2.

def kmeans(k, iterations, train):
    #test = np.asmatrix(np.loadtxt(open("test.csv", "rb"), delimiter=",", skiprows=1))
    start_time = time.clock()

    train = train[:1000,:]#debug
    points = np.zeros((train.shape[0], train.shape[1] + 1))
    points[:,:train.shape[1]] = train
    
    centroids = points.copy()
    np.random.shuffle(centroids)
    centroids =  centroids[:k, 1:centroids.shape[1]-1]
    #print(centroids.shape)
    #print("c itnit ", time.clock() - start_time, "seconds")
    for i in range(iterations):
        #print(points[:,1:points.shape[1]-1].shape)
        #print(centroids.shape)
        points[:,points.shape[1]-1] = np.transpose(np.argmin(np.sqrt(((points[:,1:points.shape[1]-1] - centroids[:, np.newaxis])**2).sum(axis=2)), axis=0))   
        centroids = np.array([points[points[:,points.shape[1]-1]==c].mean(axis=0) for c in range(centroids.shape[0])])
        centroids = centroids[:, 1: centroids.shape[1]-1]
        
    print("finished k means, calc scores after: ", time.clock() - start_time, "seconds")
    start_time = time.clock()
    
    calc_scores(points, k)
    print("Score calc time: " , time.clock() - start_time)
    return (points)        

def calc_scores(points, k):
    print( BDIndex(points, k))
    print(cIndex(points, k))

def cIndex(points,k):
    
    g = 0
    a = 0
    d = list()
    for i in range(1,points.shape[0]-1):
        for j in range(2,points.shape[0]-1):
            j = i + 1
            dt = metrices.euclidean_dist(points[i, 1:points.shape[1]-1], points[j, 1:points.shape[1]-1])
            d.append(dt)
            
            g = g + dt * c(points[i], points[i+1])
            a = a + c(points[i], points[i+1])
            
    
    d =  np.sort(np.array(d))
    mi = sum(d[:a])
    ma = sum(d[d.shape[0]-a:])
    return (g-mi)/(ma-mi)





def c(p1, p2):
    if p2[p2.shape[0]-1] == p1[p1.shape[0]-1]:
        return 1
    else:
        return 0
    
def BDIndex(points, k):
    m = np.zeros((k, points.shape[1]))
    d = np.zeros(k)

    for i in range(k):
        m[i] = points[points[:,points.shape[1]-1]==i].mean(axis=0)

    for c in range(k):
        x = points[points[:,points.shape[1]-1] == c]
        d[c] = np.sqrt(np.sum((x-m[c, np.newaxis])**2, axis=1)).mean(axis=0)
    r = np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            if i is not j:
                r[i,j] = d[i]+d[j]/metrices.euclidean_dist(d[i], d[j])
                    
    ri = np.zeros(k)
    for i in range(k):
        ri[i] = np.max(r[i,:]) #j =i cannot be max because made zero.
                
    return ri.mean()

train = np.asmatrix(np.loadtxt(open("train.csv", "rb"), delimiter=",", skiprows=1))

l = np.array((5,7,9,10,12,15))
for i in range(l.shape[0]):
    kmeans(l[i], 100, train)

#===============================================================================
# 
# c = np.zeros(5)
#     
# mat = np.zeros((8,8))
# a = np.array((4,2,3,4,4,4,7,8))
# b = np.array((1,2,3,4,4,4,7))
# b_c = b[:, np.newaxis]
# print(b_c.shape)
# print(b.shape)
# mat[0,:7] = b
# a = np.transpose(a)
# mat[:,7] = a
# i = 7
# label = 4
# z = np.mean(mat[np.where(mat[:,7] == label),:7], axis=1)#per centroid.
# print(b_c)
# print(b)
#===============================================================================