import numpy as np
#from array import array
def euclidean_dist(v1, v2):
    return np.sqrt( np.sum( np.power((v1-v2),2)  ) )

def  manhattten_dist(v1, v2):
    return np.sum(np.abs(v1-v2))

def knn(k, metric):
    train = np.asmatrix(np.loadtxt(open("train.csv", "rb"), delimiter=",", skiprows=1))
    test = np.asmatrix(np.loadtxt(open("test.csv", "rb"), delimiter=",", skiprows=1))
    positiv=0
    
    #===========================================================================
    # for faster debuging
    # train = train[:1000,:]
    # test = test[:1000,:]
    #===========================================================================
    
    print ("loaded")
    
    
    for i in range(0, test.shape[0]):
        distances = list()
        print("number of test feature: ", i )
        #print("starting distances of column", i)
        for j in range(0, train.shape[0]):
            #print("distance calc at" ,i, "progress:",j)
            if(metric == 0):
                distances.append(euclidean_dist(test[i,1:], train[j,1:]))
            else: 
                distances.append(manhattten_dist(test[i,1:], train[j,1:]))
        print(distances)    
            
        #get neighbor class labels
        a = np.argsort(distances) #indices sorted according to smallest distances
        a = a[:k] #first k indices of the smallest distances
        trainingclasses = train[:,0]
        
        #voting max of nearest neighbors
        label = list()
        votes = list()
        #decide class label based on neighbors
        for q in range(k):
            x = trainingclasses[a[q],0]
            if(x not in label):
                label.append(trainingclasses[a[q],0])
                votes.append(1)
               
            else:
                votes[label.index(x)] += 1
        print(votes)
        print(label)
        indices = np.argsort(np.multiply(votes,-1)) #multiply with -1 to make get best not smallest vote
        prediction = label[indices[0]]
        
        
        print(prediction)
        print(test[i,0])
        if prediction == test[i,0]:
            positiv += 1
            print("positiv")
        else:
            print("negativ")
        
    return (positiv/ test.shape[0])

 
li = list()
li.append(knn(3,0))
li.append(knn(3,1))
metric = ""
result = list()
if li[0] > li[1]:
    metric = "eucledean"
    result.append(li[0])
    result.append(knn(1, 0))
    result.append(knn(5, 0))
    result.append(knn(10, 0))
    result.append(knn(15, 0))
    print(metric)
else:
    metric = "manhattan"
    result.append(li[1])
    result.append(knn(1, 1))
    result.append(knn(5, 1))
    result.append(knn(10, 1))
    result.append(knn(15, 1))
     
print(result)
#===============================================================================
# #debuging
# x = np.matrix( ((1,2,3,4,5),
#                (2,3,4,5,6),
#                (3,4,5,6,7)))
#    
# for i in range(0,x.shape[0]):
#     print(x[:,:1])
#   
# abs = [2,2,1,3]
# #print(abs)
# print(np.argsort(abs))
#      
#===============================================================================


