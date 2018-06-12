#!/usr/bin/python

#run for 30 epochs and plot at 10 epochs each
#this run will clarify the effect of increasing number of filters,kernel and epochs. Increasing filter is decreasing accuracy for 10 epoch which should not be case. A possible justification is that the epochs are less. If even after increasing epochs the things don't improve, it is a discovery that for one layer optimal number of filters are 2.
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pickle

filter_sz=[2,4,8,16,32,64,128,256]
kernel_sz=[2,4,8,16,32,64]
cc=['r','b','g','m','y','k']
x_arr=[]
y_arr=[]
z_arr=[]
filename='onelayer'
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
i=0;
for k_sz in kernel_sz:
    x_arr=[]
    y_arr=[]
    z_arr=[]
    for f_sz in filter_sz:
        with open(filename+"_"+str(f_sz)+"_"+str(k_sz)+".p",'r') as fp:
	    res=pickle.load(fp)
            max_accuracy=max(res['acc'])
            x_arr += [f_sz]
            y_arr += [k_sz]
            z_arr += [max_accuracy]
            ax.scatter(np.array(x_arr),np.array(y_arr),np.array(z_arr),c=cc[i])
            print str(f_sz)+":"+str(k_sz)+":"+str(max_accuracy)
    i+=1



ax.set_xlabel('number of filters')
ax.set_ylabel('kernel size')
ax.set_zlabel('accuracy')
plt.title("Variation of accuracy with change in k_sz and f_sz")
plt.show()

