#!/usr/bin/python

#run for 30 epochs and plot at 10 epochs each
#this run will clarify the effect of increasing number of filters,kernel and epochs. Increasing filter is decreasing accuracy for 10 epoch which should not be case. A possible justification is that the epochs are less. If even after increasing epochs the things don't improve, it is a discovery that for one layer optimal number of filters are 2.
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pickle

i=0;
#labels=["2_32","4_16","16_4","32_8"]
labelarr=[]

for k_sz in [2,4,8,16,32,64]:
    templabel=[]
    for f_sz in [2,4,8,16,32,64,128,256]:
        templabel+=[str(f_sz)+"_"+str(k_sz)]
    labelarr+=[templabel]

print labelarr

for j in range(0,len(labelarr)):
    labels = labelarr[j]
    for i in range(0,len(labels)):
        epoch_arr = np.arange(0,30,1)
        with open("./"+"onelayer_"+labels[i]+".p",'r') as fp:
	    res=pickle.load(fp)
            acc_arr = res['val_acc']
            print len(acc_arr)
            plt.plot(epoch_arr,acc_arr,label=labels[i]) 
#	j= i%8
        
    i+=1
    plt.legend()
    plt.suptitle("Model accuracy vs Epochs")
    plt.title("kernel size:"+labels[0].split("_")[1]) 
    plt.xlabel("n_epochs")
    plt.ylabel("accuracy")
    plt.show()
    plt.gcf().clear()



