import pandas as pd;
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist,euclidean
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import RFE,SelectKBest,f_regression,RFECV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.pipeline import Pipeline;
import pickle


dF=pd.read_table('../data/processed_training_data.txt',encoding='utf-8') #read data

trainData=dF[dF['Type']=='Train'] #filter out training data
testData=dF[dF['Type']=='Test'] #filter out test data
trainRevenue=np.array(trainData['Weekend_Revenue']) #revenue
trainRevenue=trainRevenue.reshape((len(trainRevenue),1))
trainRevenue=np.log10(trainRevenue)
testRevenue=np.array(testData['Weekend_Revenue']) #revenue
testRevenue=testRevenue.reshape((len(testRevenue),1))
testRevenue=np.log10(testRevenue)

def plotBarChart(title,xlabel,ylabel,xtickLabel,values):
    """
    Function to plot a bar chart
    Args:
    title: The title of the figure
    ylabel: the label of y axis
    xlabel: the label of x axis
    xticklabel: the labels of ticks on x axis
    values: the values that have to be mapped to xticklabels
    """
    inds=np.arange(len(xtickLabel))
    plt.figure()  #6x4 is the aspect ratio for the plot
    plt.bar(inds,values,align='center')
    plt.xlabel(xlabel);
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(inds,xtickLabel)
    plt.tight_layout()
    plt.savefig("../Figures/\'"+title+"\'.png")
    # plt.show()
    return

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

def groupRevenue(revenue,labels):
    count=np.zeros(len(np.unique(labels)))
    minMax=np.zeros((len(np.unique(labels)),2))
    for l in np.unique(labels):
        lRevenue=revenue[labels==l]
        count[l]=len(lRevenue)
        minMax[l,0]=np.min(lRevenue)
        minMax[l,1]=np.max(lRevenue)
    return count,minMax


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


trDInput=trainData.drop(['Title','Type','Weekend_Revenue','Year_2007','Year_2008','Year_2009','Year_2010','Year_2011','Year_2012','Year_2013','Year_2014','Year_2015','Year_2016'],1)
trDIArray=np.array(trDInput) #This needs to be given as input to the classifier
trDOutput=optimalKmeans.labels_
#trDOutput=np.floor(trainRevenue).astype(int);

teDInput=testData.drop(['Title','Type','Weekend_Revenue','Year_2007','Year_2008','Year_2009','Year_2010','Year_2011','Year_2012','Year_2013','Year_2014','Year_2015','Year_2016'],1)
teDIArray=np.array(teDInput)
teDOutput=optimalKmeans.predict(testRevenue)
#teDOutput=np.floor(testRevenue).astype(int)


print np.unique(trDOutput)
print np.unique(teDOutput)
print trDOutput;
trCount,trRange=groupRevenue(trainRevenue,trDOutput)
teCount,teRange=groupRevenue(testRevenue,teDOutput)



trLabel=[""]*len(trCount)
for i in range(len(trRange)):
    trLabel[i]=str(trRange[i,0])+"-"+str(trRange[i,1])

teLabel=[""]*len(teCount)
for i in range(len(teRange)):
    teLabel[i]=str(teRange[i,0])+"-"+str(teRange[i,1])

plotBarChart("Cluster frequency for train data","cluster label","Frequency",range(0,len(trCount)),trCount)
plotBarChart("Cluster frequency for test data","cluster label","Frequency",range(0,len(teCount)),teCount)

k = SelectKBest(chi2, k=2400).fit(trDIArray,trDOutput)
X_new=k.transform(trDIArray)
clf = RandomForestClassifier(n_estimators=100,criterion='entropy',oob_score=True, n_jobs=2,verbose=0)

paramGrid={
    'max_features':['sqrt','log2'],
    'min_samples_split':np.arange(10,40,4),
    'min_samples_leaf':np.arange(3,10,3)
}

gS=GridSearchCV(clf,param_grid=paramGrid,n_jobs=4)
gS.fit(X_new,trDOutput)
with open('cval.pkl','wb') as op:
    pickle.dump(gS,op,pickle.HIGHEST_PROTOCOL)

teNew=k.transform(teDIArray)
print gS.best_estimator_
print "Training score = ",gS.best_estimator_.score(X_new,trDOutput)
print "Score = ",gS.best_estimator_.score(teNew,teDOutput)
# print trDIArray.shape
# print teDIArray.shape

# for  i in range(2000,2500,100):
#      X_new = SelectKBest(chi2, k=i).fit_transform(trDIArray,trDOutput)
#      score=cross_val_score(clf,X_new,trDOutput,cv=5)
#      print "k = ",i," score = ",np.mean(score)
        
# score=cross_val_score(clf,trDIArray,trDOutput,cv=5)
# print "k = all score = ",np.mean(score)


# print "Optimal # features = ",rfecv.n_features_
# print "mask = ",rfecv.support_

# with open('rfecv_results.pkl','wb') as op:
#     pickle.dump(rfecv,op,pickle.HIGHEST_PROTOCOL)





# gridSearch=GridSearchCV(clf,param_grid=paramGrid,n_jobs=4)
# gridSearch.fit(trDInput,trDOutput)

# bestEstimator=gridSearch.best_estimator_

# tOut=bestEstimator.predict(teDInput);
# print tOut
# print teDOutput
# print bestEstimator.score(teDInput,teDOutput)



# clf=LogisticRegression(n_jobs=2);
# ovr=OneVsRestClassifier(clf,n_jobs=2)
# ovr.fit(trDInput,trDOutput)
# print ovr.score(teDInput,teDOutput)



# REGRESSOR

# reg=RandomForestRegressor(n_estimators=100,oob_score=True,n_jobs=-3,verbose=1)
# reg.fit(trDInput,trDOutput)

# print reg.score(teDInput,teDOutput);





# clf=GradientBoostingClassifier(verbose=1)
# paramGrid={
#     'loss':['deviance','exponential'],
#     'max_depth':np.arange(3,18,3),
#     'min_samples_split':np.arange(2,10,2),
#     'min_samples_leaf':np.arange(1,10,2)
# }

# gS=GridSearchCV(clf,paramGrid,n_jobs=-3)
# gS.fit(trDInput,trDOutput)
# gS.cv_results_
# with open('cv_results.pkl','wb') as op:
#     pickle.dump(gS,op,pickle.HIGHEST_PROTOCOL)

# bestEstimator=gS.best_estimator_
# print bestEstimator.score(teDInput,teDOutput)
