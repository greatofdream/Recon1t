import numpy as np 
import scipy, h5py
import tables
import sys
from scipy.optimize import minimize
from numpy.polynomial import legendre as LG
import matplotlib.pyplot as plt
import os
import argparse, uproot
#os.chdir()
#print(os.getcwd())

def Calib(theta, *args):
    ChannelID, flight_time, PMT_pos, cut = args
    y = flight_time
    # fixed axis
    x = Legendre_coeff(PMT_pos, cut)
    Legend_coeff = x[ChannelID,:]
    # quantile regression
    T_i = np.dot(Legend_coeff, theta)
    L = Likelihood_quantile(y, T_i, 0.01, 0.3)
    # L = np.log(np.sum((np.transpose(np.dot(Legend_coeff, theta))-y)**2))
    # print(L)
    return L

def Likelihood_quantile(y, T_i, tau, ts):
    less = T_i[y<T_i] - y[y<T_i]
    more = y[y>=T_i] - T_i[y>=T_i]
    
    R = (1-tau)*np.sum(less) + tau*np.sum(more)
    #log_Likelihood = exp
    return R

def quantile_check(y, T_i, tau, ts):
    less = T_i - y[y<T_i]
    more = y[y>=T_i] - T_i
    R = (1-tau)*np.sum(less) + tau*np.sum(more)
    return R

def Legendre_coeff(PMT_pos, cut):
    # vertex = np.array([0,0,2,10])
    vertex = np.array([0,0,0,1])
    cos_theta = np.sum(vertex[1:4]*PMT_pos,axis=1) \
        /np.sqrt(np.sum(vertex[1:4]**2)*np.sum(PMT_pos**2,axis=1))
    # accurancy and nan value
    cos_theta = np.nan_to_num(cos_theta)
    cos_theta[cos_theta>1] = 1
    cos_theta[cos_theta<-1] =-1
    size = np.size(PMT_pos[:,0])
    x = np.zeros((size, cut))
    # legendre coeff
    for i in np.arange(0, cut):
        c = np.zeros(cut)
        c[i] = 1
        x[:,i] = LG.legval(cos_theta,c)
    return x  

def rosen_hess(x, *args):
    x = np.asarray(x)
    H = np.diag(-400*x[:-1],1) - np.diag(400*x[:-1],-1)
    diagonal = np.zeros_like(x)
    diagonal[0] = 1200*x[0]**2-400*x[1]+2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200*x[1:-1]**2 - 400*x[2:]
    H = H + np.diag(diagonal)
    return H

def rosen_der(x, *args):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der

def hessian(x, *args):
    ChannelID, PETime, PMT_pos, cut = args
    H = np.zeros((len(x),len(x)))
    h = 1e-3
    k = 1e-3
    for i in np.arange(len(x)):
        for j in np.arange(len(x)):
            if (i != j):
                delta1 = np.zeros(len(x))
                delta1[i] = h
                delta1[j] = k
                delta2 = np.zeros(len(x))
                delta2[i] = -h
                delta2[j] = k


                L1 = - Calib(x + delta1, *(ChannelID, PETime, PMT_pos, cut))
                L2 = - Calib(x - delta1, *(ChannelID, PETime, PMT_pos, cut))
                L3 = - Calib(x + delta2, *(ChannelID, PETime, PMT_pos, cut))
                L4 = - Calib(x - delta2, *(ChannelID, PETime, PMT_pos, cut))
                H[i,j] = (L1+L2-L3-L4)/(4*h*k)
            else:
                delta = np.zeros(len(x))
                delta[i] = h
                L1 = - Calib(x + delta, *(ChannelID, PETime, PMT_pos, cut))
                L2 = - Calib(x - delta, *(ChannelID, PETime, PMT_pos, cut))
                L3 = - Calib(x, *(ChannelID, PETime, PMT_pos, cut))
                H[i,j] = (L1+L2-2*L3)/h**2                
    return H

def getTimeData(filename):
    print(filename)
    if filename.endswith('.h5'):
        # read files by table
        h1 = tables.open_file(filename,'r')
        truthtable = h1.root.GroundTruth
        EventID = truthtable['EventID'][:]
        ChannelID = truthtable['ChannelID'][:]
        PETime = truthtable[:]['PETime']
        h1.close()
        
    elif filename.endswith('.root'):
        f = uproot.open(filename)
        e = f['evt']
        # ! 1e7 is estimate value,need to get from file
        ChannelID = np.zeros(shape=(np.uint(1e7),),dtype=np.uint16)
        PETime = np.zeros(shape=(np.uint(1e7),))
        cursor = 0
        for eid, chl, ht in zip(e.array('evtID'), e.array('pmtID'), e.array('hitTime')):
            print('pmtID:{},hittime:{}'.format(chl.shape,ht.shape))
            # select Large pmt 
            chlLarge = chl[(chl<largeidUp)&(ht<3000)]
            htLarge = ht[(chl<largeidUp)&(ht<3000)]
            cursorNext = cursor + chlLarge.shape[0]
            ChannelID[cursor:cursorNext] = chlLarge
            PETime[cursor:cursorNext] = htLarge
            cursor = cursorNext
            print('processed:{}'.format(eid))
        ChannelID = np.array(ChannelID[0:cursor])
        PETime = np.array(PETime[0:cursor])
    else:
        print('the format of input file:{} is not correct'.format(filename))
        exit(1)
    return ChannelID, PETime
def main_Calib(radius, path, fout):
    #filename = '/mnt/stage/douwei/Simulation/1t_root/1.5MeV_015/1t_' + radius + '.h5'
    filename = path
    ChannelID, PETime = getTimeData(filename)
    print('pmtID:{},hittime:{}'.format(ChannelID.shape,PETime.shape))

    # flight_time = PulseTime - dETime
    flight_time = PETime
    '''
    ChannelID = ChannelID[~(flight_time==0)]
    flight_time = flight_time[~(flight_time==0)]
    '''
    
    with h5py.File(fout,'w') as out:
        for cut in np.arange(5,35,5):
            theta0 = np.zeros(cut) # initial value
            theta0[0] = np.mean(flight_time) - 26
            result = minimize(Calib,theta0, method='SLSQP',args = (ChannelID, flight_time, PMT_pos, cut))  
            record = np.array(result.x, dtype=float)
            hess = hessian(result.x, *(ChannelID, flight_time, PMT_pos, cut))
            hess_inv = np.linalg.pinv(np.matrix(hess))
            print(result.x)
            
            x = Legendre_coeff(PMT_pos, cut)
            predict = []
            predict.append(np.dot(x, result.x))

            out.create_dataset('coeff' + str(cut), data = record)
            out.create_dataset('ft' + str(cut), data = flight_time)
            out.create_dataset('ch' + str(cut), data = ChannelID)
            out.create_dataset('predict' + str(cut), data = predict)
            out.create_dataset('hinv'+str(cut), data=hess_inv)
psr = argparse.ArgumentParser()
psr.add_argument("-o", dest='opt', help='output help')
psr.add_argument('-g', dest='geo', help='geometry')
psr.add_argument('-r', dest='radius', help='radius')
psr.add_argument('ipt', help='input')

args = psr.parse_args()
print(args)

f = open(args.geo)
line = f.readline()
data_list = []
while line:
    num = list(map(float,line.split()))
    data_list.append(num)
    line = f.readline()
f.close()
largeidUp = 17612
PMT_pos = np.array(data_list)[:,1:4]

radius = np.int(args.radius)
#cut = 6 # Legend order
main_Calib(radius, args.ipt, args.opt)
