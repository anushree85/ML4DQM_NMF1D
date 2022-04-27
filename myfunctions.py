import numpy as np
import json
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 
def PlotHist(df,run,ls, max,min,bin):

    Xmax=max#30.0 #80000.0
    Xmin=min
    Xbins=bin #30 #100
    hname="size_PXLayer_2"

    ahisto=df['histo'][run][ls]
    x= np.linspace(Xmin,Xmax,Xbins)
    plt.xlim(Xmin,Xmax)

    plt.step(x, ahisto, where='mid', c='black',label=(" LS " + str(df.fromlumi[run][ls]) + " Run " + str(df.fromrun[run][ls]) ))
    plt.xlabel("Cluster charge(electrons)",loc="center")
    plt.ylabel("Arbritrary units",loc="center")
    plt.legend()



def checkLS(run,ls, jsondata):
    isok=False

    if str(run) in jsondata.keys():
        for i in jsondata[str(run)]:
            if (ls>=i[0] and ls <=i[1]):
                isok=True
                return isok
        return isok



def checkCustomLS(run,ls ,jsondata):
    isok=False

    if str(run) in jsondata.keys():
        for i in jsondata[str(run)]:
           # print("i[0]",i[0])
           # print("i[1]",i[1])
            if (ls>=i[0] and ls <=i[1]):
                isok=True
                return isok
        return isok


def find_IOV(run, gain_array):
    for i in gain_array:
        if (run<i):
            index=np.where(gain_array==i)
            index=int(np.array(index).T)
            #print(run, i, index)
            k=index -1
            #print(run, i,k)
            if(index==0):
                k=0
            return k




def cluster_byrun(dfl, run_all,runlist):
    data_byrun=[]
    k=0
    print(dfl.shape[0])
    for i in range(dfl.shape[0]):
        #pass
        data = np.stack(dfl[i], axis=0)
        #if(len(data_byrun_nj)<1):
        #    data_byrun_nj.append(data)
        #    continue
        if(run_all[i]==runlist[k]):        
            data_byrun.append(data)
        if(run_all[i]!=runlist[k]):
            yield data_byrun
            print(data_byrun.shape)
            data_byrun[:]=[]
            k=k+1
            data_byrun.append(data)
    return data_byrun



import math
from keras import backend as K
import tensorflow.compat.v1 as tf
def mseTop10(y_true, y_pred):
    top_values, _ = tf.nn.top_k(K.square(y_pred - y_true), k=10, sorted=True)
    mean=K.mean(top_values, axis=-1)
    return K.sqrt(mean)

def MSE_dist(data,tot):
    #print(tot.shape)
    MSEs=K.eval(mseTop10(tot,data))
    return MSEs




import scipy as stats
from scipy.spatial import distance
from sklearn.preprocessing import normalize

def distance_mes(DT, DR):
    dist_mat=[]
    dist_mat_cor=[]
    size=np.int(DT.shape[0])
    #print(size)
    for i in range(0,size):
        Eucl_dist=distance.euclidean(DT[i], DR[i])
        correlation_dist=distance.correlation(DT[i], DR[i])

        dist_mat.append(Eucl_dist)
        dist_mat_cor.append(correlation_dist)

    return dist_mat, dist_mat_cor




## poor man's grouping
def group_data(dfl, run_all, run):
    indices=np.where(run_all==run)
    #print(indices)
    #print(type(indices), run)
    data=[]
    indices=np.array(indices)
    for x in indices.T:
        ind=int(x)
        data.append(dfl[ind])
    return data




def group_ind(dfl, indices):
    data=[]
    indices=np.array(indices)
    #print("indices",indices)
    for x in indices.T:
        ind=int(x)
        #print("x",x)
        data.append(dfl[ind])
    return data




def plotComponents(comp,suffx):
    #colors = ["blue", "orange" , "green", "red", "purple"]
    size=len(comp)
    #color=iter(cm.rainbow(np.linspace(0,1,size)))
    color=['blue','green','magenta','cyan','orange','yellow'];
    color=iter(color)
    #x= np.linspace(0,8000.0,100)
    shortrange=True
    if(shortrange):
        x= np.linspace(0,62,62)
    #print(np.shape(x))
    #print("x",x)
    for i in range(0,size):
        #print(len(comp[i]))
        #print(comp[i].max())
        #print(comp[i])
        #plt.hist(comp[i],bins="auto")
        plt.step(x,comp[i], where='mid', color=next(color),label="basis element " + str(i+1))
        #plt.step(x,comp[i]/comp[i].sum(), where='mid', label="component " + str(i+1))
    plt.legend()
    plt.title("distribution of basis elements")
    #plt.ylable("Number of bins ")
    #plt.text(0,0.45,"297178 with good LSs")
    #plt.savefig("TOTgolden_ZeroBias_UL2018_DataFrame_1D_size_PXLayer_3_plotcomponents_"+suffx+".png")
    plt.show()



def plotDecomposition_notallcomp(i, truedata, W,H, suffix, min3=0, min5=0):
    #x= np.linspace(0,30,30)
    x= np.linspace(0,80000.0,62)
    #if(shortrange):
    #    x= np.linspace(0,30,13)
    plt.step(x,truedata[i], where='mid', label="original",color='black')
    dl=np.array([3,4,5])
    delt=False
    if(delt):
        W=np.delete(W,dl, axis=0)
        H=np.delete(H,dl, axis=1)
    size=len(W)
    print(size)
    #print(H.shape)
    #color=iter(cm.rainbow(np.linspace(0,1,size)))
    color=['blue','green','magenta','cyan','orange','yellow'];
    color=iter(color)
    for j in range(0,size):
        sums=0
        sums=np.sum(W[j]*H[i][j])
        sums=sums*100
        sums='{0:.2f}'.format(sums) 
        print(sums)
        plt.step(x,W[j]*H[i][j], where='mid', color=next(color), label="component" + str(j+1)+ " "+str(sums))
        print(np.sum(W[j]*H[i][j]), j)
    tot=np.matmul(H[i],W)
    print("tottttttttt  ",np.sum(tot))
    print(np.sum(truedata[i]))
    print(tot.shape)
    err=K.eval(mseTop10(tot,truedata[i]))
    print("err",err)
    plt.step(x,tot, where='mid', label="Reco", linestyle='--', c="red" )#, mse10="+str(err), linestyle='--',c="black")
    plt.xlabel("Cluster charge(electrons)",loc="center")
    plt.ylabel("normalized number of entries",loc="center")
    plt.title("Pixel Cluster Charge Barrel Layer 1")
    #plt.text(0.5,0.5,str(err))
    #plt.yscale("log")
    #plt.set_size_inches(5,5)
    #if(H[i][2] > min3 or H[i][4]> min5):
    #    print(H[i][2])
    plt.legend()
    plt.show()



def plotDecomposition_average(truedata, W,H, suffix, min3=0, min5=0):
    x= np.linspace(0,80000.0,62)
    #if(shortrange):
    #    x= np.linspace(0,8000,62)
    tot=np.zeros(np.shape(truedata))
    truedata=np.mean(truedata,axis=0)    
    print("tot shape", tot.shape)
    print("truedata.shape",np.shape(truedata))
    plt.step(x,truedata, where='mid', label="original",c="black")
    size=len(W)
    print(size)
    print(np.shape(H))
    tot=np.matmul(H,W)
    print("tot shape after multiplication before mean=", np.shape(tot))
    tot=np.mean(tot,axis=0)
    print("tot shape after multiplication=", np.shape(tot))
    Hav=np.zeros(W.shape)
    Hav=np.mean(H,axis=0)
    print("tot shape of Haverage=", Hav.shape)
    #color=iter(cm.rainbow(np.linspace(0,1,size)))
    color=['blue','green','magenta','cyan','orange','yellow'];
    color=iter(color)
    for j in range(0,size):
        sums=0
        sums=np.sum(W[j]*Hav[j])
        sums=sums*100
        sums='{0:.2f}'.format(sums) 
        print(sums)
        plt.step(x,W[j]*Hav[j], where='mid', color=next(color), label="component" + str(j+1) + " "+str(sums))
    err=K.eval(mseTop10(tot,truedata))
    print("err",err)
    plt.step(x,tot, where='mid', label="Reco", linestyle='--', c="red" )#, mse10="+str(err), linestyle='--',c="black")
    plt.xlabel("Cluster charge(electrons)",loc="center")
    plt.ylabel("normalized number of entries",loc="center")
    plt.title("Pixel Cluster Charge Barrel Layer 1")
    #plt.text(0.5,0.5,str(err))
    #plt.yscale("log")
    #plt.set_size_inches(5,5)
    #if(H[i][2] > min3 or H[i][4]> min5):
    #    print(H[i][2])
    plt.legend()
       # plt.title("original vs reco vs plot components")
       # plt.savefig("GOLDEN_ZeroBias_UL2017_DataFrame_1D_chargeInner_PXLayer_1_vs_reco_vs_plotcomponents.png")
    plt.show()


def comp_dist(i,W,H):
    sums=0
    #print(H[0][1])
    comp_fact=[]
    for j in range(6):
        #print(H[0][1])
        fact=(np.sum(W[j]*H[i][j]))
        sums=sums+fact
        #print(fact)
        comp_fact.append(fact)
        #print(sums)
    return comp_fact




def plot_compCont(comp_arr,run):

    color=['blue','green','magenta','cyan','orange','yellow']
    color=iter(color)
    for j in range(6):
        plt.plot(comp_arr[j],color=next(color),label="component " + str(j+1))
    plt.legend()
    plt.title("Run: " + str(run)+ " Pixel Cluster Charge Barrel Layer 1") 
    plt.xlabel("Number of LSs",loc="center")
    plt.ylabel("contribution of component",loc="center")




def plot_compConttwo(comp_arr,run,dqltest):
    fig, axs = plt.subplots(2,1,figsize=(10,10), gridspec_kw={'height_ratios': [3, 1]})
    #refshape=np.shape(lumi)
    #refshape=int(''.join(map(str, refshape)))
    #x=np.linspace(lumi,0,refshape)
    color=['blue','green','magenta','cyan','orange','yellow']
    color=iter(color)
    for j in range(6):
        #plt.plot(comp_arr[j],color=next(color),label="component " + str(j+1))
        #ax[0].step(x,comp_arr[j],where='mid',color=next(color),label="component " + str(j+1))
        axs[0].plot(comp_arr[j],color=next(color),label="component " + str(j+1))
    axs[0].text(0,1.01,"Run: " + str(run)+ " Pixel Cluster Charge Barrel Layer 1",fontsize=15)
    axs[0].legend(fontsize=15)
    axs[0].set_ylim(0,1)
    axs[0].set_ylabel("contribution of components")
    #axs[0].set_xlabel("Number of LSs")
    axs[1].plot(dqltest)
    axs[1].set_xlabel("Number of LSs")
    axs[1].set_ylabel("tag of the LSs using golden json")




def component(i, W,H, comp,min3=0):
    #size=len(W)
    fact=(np.sum(W[comp]*H[i][comp]))
    if(fact>min3):
        return True




def firstcomponent(i, W,H, comp,min3=0):
    #size=len(W)
    fact=(np.sum(W[comp]*H[i][comp]))
    if(fact<min3):
        return True





