import os
import pandas as pd;
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist,euclidean
import sys
sys.path.append(os.path.join(os.getcwd(),"../utils"))
import constants
import classifier_preprocessing, plotting_utils
from sklearn import preprocessing

dF=pd.read_table('../data/processed_training_data.txt',encoding='utf-8') #read data

trainData=dF[dF['Type']=='Train'] #filter out training data
testData=dF[dF['Type']=='Test'] #filter out test data
trainRevenue=np.array(trainData['Weekend_Revenue']) #revenue
trainRevenue=np.log10(trainRevenue.reshape((len(trainRevenue),1)))
testRevenue=np.log10(np.array(testData['Weekend_Revenue'])) #revenue
testRevenue=testRevenue.reshape((len(testRevenue),1))

def avgClusterCenterDist(ccenter,labels,imgVectors):
    '''
    Function to find the average distance of the points of clusters to their respective centroids
    Args:
    @param ccenter: The cluster centroids
    @param labels: The labels assigned to data samples
    @param imgVectors: The data samples
    @return dist: The average distance of the points of clusters to their respective centroids
    '''
    clusterLabels=np.unique(labels);
    dist=np.zeros(len(clusterLabels));
    for cl in clusterLabels:
        imgVct=imgVectors[labels==cl]
        center=ccenter[cl];
        for v in imgVct:
            dist[cl]+=euclidean(v,center)
        dist[cl]/=len(imgVct)
    return dist
    

def daviesBouldinIndex(ccenter,labels,imgVectors):
    '''
    Function to compute the Davies-Bouldin Index to measure clustering quality. The lower the value of Davies-Bouldin Index, the better is the cluster quality
    Args:
    @param ccenter: The cluster centroids
    @param labels: The labels assigned to data samples
    @param imgVectors: The data samples
    @return The Davies Bouldin Index
    '''
    clusters=np.unique(labels)
    nClusters=len(clusters);
    avgDist=avgClusterCenterDist(ccenter,labels,imgVectors); #Compute average distances from the centroid
    dbIndex=0.0;
    for i in np.arange(0,nClusters): #for all the clusters
        maxDij=0.0;
        for j in np.arange(0,nClusters): #for every other cluster
            if i==j:
                continue;
            dij=(avgDist[i]+avgDist[j])/euclidean(ccenter[i],ccenter[j]) 
            if dij>maxDij:
                maxDij=dij;
        dbIndex+=maxDij;
    return dbIndex/nClusters;


minDB=float("Inf");
optimalK=0;
optimalKmeans=KMeans(n_clusters=2,n_jobs=-1);
for k in np.arange(6,7):
    kmeans=KMeans(n_clusters=k,n_jobs=-1).fit(trainRevenue);
    dbIndex=daviesBouldinIndex(kmeans.cluster_centers_,kmeans.labels_,trainRevenue)
    if(dbIndex<minDB):
        minDB=dbIndex;
        optimalK=k;
        optimalKmeans=kmeans;
        print"Update K to ",k, "score = ",minDB;
    if k%50==0:
        print "k=",k,"Optimal intermediate = ",optimalK
print "optimal k = ",optimalK, "score = ",minDB
print "cluster centers = ",optimalKmeans.cluster_centers_



# maxSilScore=0.0;
# optimalK=0;
# optimalKmeans=KMeans(n_clusters=2,n_jobs=-1);
# for k in np.arange(2,len(trainRevenue)):
#     kmeans=KMeans(n_clusters=k,n_jobs=-1).fit(trainRevenue);
#     silScore=silhouette_score(trainRevenue,kmeans.labels_) #Compute the Silhouette Score
#     if(silScore>maxSilScore):
#         maxSilScore=silScore;
#         optimalK=k;
#         optimalKmeans=kmeans;
#     if k%50==0:
#         print "k=",k,"Optimal intermediate = ",optimalK
# print "optimal k = ",optimalK, "score = ",maxSilScore
# print "cluster centers = ",optimalKmeans.cluster_centers_


### HAC

# maxClusters=len(trainRevenue);
# hac=AgglomerativeClustering(n_clusters=1);
# minDbIndex=float("Inf")
# optimalK=maxClusters;
# for k in np.arange(1,maxClusters+1):
#     hac.set_params(n_clusters=k);
#     hac.fit(trainRevenue)
#     ccenter=np.zeros((k,1))
#     labels=hac.labels_
#     for l in np.unique(labels):
#         ccenter[l,:]=np.mean(trainRevenue[labels==l,:],axis=0)
#     dbIndex=daviesBouldinIndex(ccenter,hac.labels_,trainRevenue)
#     if(dbIndex<minDbIndex):
#         minDbIndex=dbIndex;
#         optimalK=k
# print("Optimal k = ",optimalK)



###########################################################################################
###########################################################################################
#                                  CLASSIFICATION                                         #
###########################################################################################
###########################################################################################

#First drop moviename, type and revenue from dat which is to be given to classifier as input

trDInput=trainData.drop(['Title','Type','Weekend_Revenue'],1)
trDIArray=np.array(trDInput) #This needs to be given as input to the classifier
trDOutput=optimalKmeans.labels_

teDInput=testData.drop(['Title','Type','Weekend_Revenue'],1)
teDIArray=np.array(teDInput)
teDOutput=optimalKmeans.predict(testRevenue)
print "Testing Classifiers"

if __name__ == '__main__':
    data_folder_path = os.path.join(os.getcwd(),"../Data")
    #Running all the classifiers on a sample of the data to report basic accuracies
    features_train=trDIArray;
    features_test=teDIArray;
    y_train=trDOutput;
    y_test=teDOutput;
    #Get Predictive accuracy of each model with the data
    features_train_normalized =preprocessing.MinMaxScaler().fit_transform(features_train)
    print "Train Data Shape : ",features_train.shape
    print "Test Data Shape : ", features_test.shape
    accuracies = map(lambda classifier_type: classifier_preprocessing.get_raw_classifier_accuracy(classifier_type, features_train, y_train), constants.INITIAL_CLASSIFIER_LIST)
    plotting_utils.create_bar_plot(constants.INITIAL_CLASSIFIER_LIST, range(len(constants.INITIAL_CLASSIFIER_LIST)), accuracies, "Prediction Accuracy", "Classifier Type", "CV-Accuracy of the Data on Different Classfiers" )
    plotting_utils.create_pv_revenue_lr_plots(trainData)