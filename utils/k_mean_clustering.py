import sklearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import plotting_utils
import sys
import os
sys.path.append(os.path.join(os.getcwd(),"../data"))
import pandas as pd
import re

def calculate_k_means_variance(df,num_clusters):
    kmeans = KMeans(n_clusters=num_clusters).fit(df)
    labels=kmeans.labels_
    cluster_centers=kmeans.cluster_centers_
    return get_sum_squared_error(df,labels,cluster_centers)


def get_sum_squared_error(df,labels,cluster_centers):
    cluster_var=0.0
    for label in set(labels):
        cluster_var += np.sum(np.square(df[labels==label]- cluster_centers[labels][labels==label]))
    return cluster_var

def plot_variance(df):
    x_array=range(2,26)
    y_array=[]
    for num_clusters in range(2,26):
        y_array.append(calculate_k_means_variance(df,num_clusters))
    x_labels = map(lambda x : str(x),x_array)
    plotting_utils.create_line_plot(labels=x_labels,x_values=x_array,y_values=y_array,y_label="Total Within Cluster Variance (Log Scale)",x_label="Number of clusters",title="Within Cluster Variance for different clusters")

def get_range(label,labels,df):
    min_value=np.min(df[labels==label])
    max_value = np.max(df[labels==label])
    return (min_value,max_value)


if __name__ == '__main__':
    data_folder=os.path.join(os.getcwd(),"../data")
    file_revenue = pd.read_csv(os.path.join(data_folder,"processed_training_data.txt"), sep="\t")
    train_data = file_revenue[file_revenue['Type'] == "Train"]
    revenue_column = train_data['Weekend_Revenue']
    log_revenue = np.log10(revenue_column[:, np.newaxis])
    plot_variance(log_revenue)
    number_of_cluster_chosen = 6
    kmeans = KMeans(n_clusters=number_of_cluster_chosen).fit(log_revenue)
    labels = kmeans.labels_
    plt.close('all')
    plt.hist(labels,stacked=True,bins=range(7))
    plt.gca().set_xticks(np.arange(7)+0.4)
    plt.gca().set_xticklabels(np.arange(6))
    plt.ylabel("Frequency")  # Y-axis label
    plt.xlabel("Cluster label")  # X-axis label
    plt.title("Frequency of different Clusters")
    plt.savefig("../Figures/bucket_distribution.jpg")
    dict_range={}
    for label in set(labels):
        dict_range[label] = get_range(label,labels,revenue_column[:,np.newaxis])
    sorted_range=sorted(dict_range.iteritems(),key=lambda x : x[1][0])
    labels=map(lambda x: x[0],sorted_range)
    ranges = map(lambda x: x[1], sorted_range)
    list_ranges =zip(*ranges)
    min_range = np.log10(np.array(list_ranges[0]))
    max_range = np.log10(np.array(list_ranges[1]))
    plotting_utils.create_bar_plot_clustered(min_range,max_range,labels,"Cluster label","Log Revenue","Log(Revenue) vs Cluster label")








