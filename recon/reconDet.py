# recon range: [-1,1], need * detector radius

import numpy as np
import scipy, h5py
import scipy.stats as stats
import os,sys
import tables
import scipy.io as scio
import matplotlib.pyplot as plt
import uproot, argparse
from scipy.optimize import minimize
from scipy import interpolate
from numpy.polynomial import legendre as LG
from scipy import special
from scipy.linalg import norm
import warnings
import argparse

warnings.filterwarnings('ignore')

# physical constant (if need)
Light_yield = 4285*0.88 # light yield
Att_LS = 18 # attenuation length of LS
Att_Wtr = 300 # attenuation length of water
tau_r = 1.6 # fast time constant
TTS = 5.5/2.355
QE = 0.20
PMT_radius = 0.254
c = 2.99792e8
n = 1.48

# boundaries
detRadius = 17700

def load_coeff(peOrderFile, timeOrderFile, order):
    # h = tables.open_file('../calib/PE_coeff_1t' + order + '.h5','r')
    h = tables.open_file(peOrderFile, 'r')
    coeff_pe = h.root['/{}/poly'.format(order)][:]
    cut_pe, fitcut_pe = coeff_pe.shape

    # h = tables.open_file('../calib/Time_coeff_1t' + order + '.h5','r')
    h = tables.open_file(timeOrderFile, 'r')
    coeff_time = h.root['/{}/poly'.format(order)][:]
    h.close()
    cut_time, fitcut_time = coeff_time.shape
    return coeff_pe, coeff_time, cut_pe, fitcut_pe, cut_time, fitcut_time

def r2c(c):
    # radius to Cartesian
    v = np.zeros(3)
    v[2] = c[0] * np.cos(c[1]) #z
    rho = c[0] * np.sin(c[1])
    v[0] = rho * np.cos(c[2]) #x
    v[1] = rho * np.sin(c[2]) #y
    return v

def c2r(c):
    v = np.zeros(3)
    v[0] = norm(c)
    v[1] = np.arccos(c[2]/(v[0]+1e-6))
    v[2] = np.arctan(c[1]/(c[0]+1e-6)) + (c[0]<0)*np.pi
    return v

def Likelihood(vertex, *args):
    '''
    vertex[1]: r
    vertex[2]: theta
    vertex[3]: phi
    '''
    coeff_time, coeff_pe, PMT_pos, fired_PMT, time_array, pe_array, cut_time, cut_pe = args
    '''
    print('coeff_time:{},coeff_pe:{}'.format(coeff_time,coeff_pe))
    print('pmtPos:{}'.format(PMT_pos))
    print('pmtID:{}'.format(fired_PMT))
    print('time:{}'.format(time_array))
    '''
    L1 = Likelihood_PE(vertex, *(coeff_pe, PMT_pos, pe_array, cut_pe))
    # L2 = Likelihood_Time(vertex, *(coeff_time, PMT_pos, fired_PMT, time_array, cut_time))
    L2 = Likelihood_Time(vertex, *(coeff_time, PMT_pos, fired_PMT, time_array, cut_time ))
    # L2 = 0
    return L1+L2
                         
def Likelihood_PE(vertex, *args):
    coeff, PMT_pos, event_pe, cut = args
    y = event_pe
    
    z = abs(vertex[1])
    if z > 1:
        z = np.sign(z)-1e-6
 
    if z<1e-3:
        # assume (0,0,1)
        # cos_theta = PMT_pos[:,2] / norm(PMT_pos,axis=1)
        vertex[1] = 1e-3
        z = 1e-3
        v = r2c(vertex[1:4])
        cos_theta = np.dot(v,PMT_pos.T) / (z*norm(PMT_pos,axis=1))
    else:
        v = r2c(vertex[1:4])
        cos_theta = np.dot(v,PMT_pos.T) / (z*norm(PMT_pos,axis=1))
    
    size = np.size(PMT_pos[:,0])
    x = np.zeros((size, cut))
    # legendre theta of PMTs
    for i in np.arange(0,cut):
        c = np.zeros(cut)
        c[i] = 1
        x[:,i] = LG.legval(cos_theta,c)
    # legendre coeff by polynomials
    k = np.zeros(cut)
    for i in np.arange(cut):
        # Legendre fit
        # k[i] = np.sum(np.polynomial.legendre.legval(z,coeff[i,:]))
        # polynomial fit
        
        if(i % 2 == 0):
            # k[i] = coeff[i,0] + coeff[i,1] * z ** 2 + coeff[i,2] * z ** 4 + coeff[i,3] * z ** 6 + coeff[i,4] * z ** 8 + \
            k[i] = np.sum(coeff[i,:] * z ** np.arange(0,2*np.size(coeff[i,:]),2))
            
        elif(i % 2 == 1):
            # k[i] = coeff[i,0] * z + coeff[i,1] * z ** 3 + coeff[i,2] * z ** 5 + coeff[i,3] * z ** 7 + coeff[i,4] * z ** 9
            k[i] = np.sum(coeff[i,:] * z ** np.arange(1,2*np.size(coeff[i,:])+1,2))

    k[0] = vertex[0]
    expect = np.exp(np.dot(x,k))
    '''
    print('z:{};expect:{}'.format(z, expect))
    print('k:{}'.format(k))
    '''
    L = -np.sum(y*np.log(expect)-expect)
    if(np.isnan(L)):
        print(z, expect)
        print(vertex)
        print('costheta:{}'.format(cos_theta))
        print('x:{}'.format(x))
        print('k:{}'.format(k))
        print(np.dot(x,k))
        #print(np.exp(-expect))
        #print(expect**y)
        #print((expect**y)*np.exp(-expect))
        exit()
    return L

def Likelihood_Time(vertex, *args):
    coeff, PMT_pos, fired, time, cut = args
    y = time
    # fixed axis
    z = abs(vertex[1])
    if z > 1:
        z = np.sign(z)-1e-6

    if z<1e-3:
        # assume (0,0,1)
        # cos_theta = PMT_pos[:,2] / norm(PMT_pos,axis=1)
        vertex[1] = 1e-3
        z = 1e-3
        v = r2c(vertex[1:4])
        cos_theta = np.dot(v,PMT_pos.T) / (z*norm(PMT_pos,axis=1))
    else:
        v = r2c(vertex[1:4])
        cos_theta = np.dot(v,PMT_pos.T) / (z*norm(PMT_pos,axis=1))
    # accurancy and nan value
    cos_theta = np.nan_to_num(cos_theta)
    cos_theta[cos_theta>1] = 1
    cos_theta[cos_theta<-1] =-1

    cos_total = cos_theta[fired]
    
    size = np.size(cos_total)
    x = np.zeros((size, cut))
    # legendre theta of PMTs
    for i in np.arange(0,cut):
        c = np.zeros(cut)
        c[i] = 1
        x[:,i] = LG.legval(cos_total,c)
        
    # legendre coeff by polynomials
    k = np.zeros((1,cut))
    for i in np.arange(cut):
        # Legendre fit
        # k[i] = np.sum(np.polynomial.legendre.legval(z,coeff[i,:]))
        # polynomial fit
        if(i % 2 == 0):
            #k[0,i] = coeff[i,0] + coeff[i,1] * z ** 2 + coeff[i,2] * z ** 4 + coeff[i,3] * z ** 6 + coeff[i,4] * z ** 8
            k[0,i] = np.sum(coeff[i,:] * z ** np.arange(0,2*np.size(coeff[i,:]),2))
        elif(i % 2 == 1):
            #k[0,i] = coeff[i,0] * z + coeff[i,1] * z ** 3 + coeff[i,2] * z ** 5 + coeff[i,3] * z ** 7 + coeff[i,4] * z ** 9
            k[0,i] = np.sum(coeff[i,:] * z ** np.arange(1,2*np.size(coeff[i,:])+1,2))
    k[0,0] = vertex[4]
    T_i = np.dot(x, np.transpose(k))
    L = Likelihood_quantile(y, T_i[:,0], 0.1, 0.3)
    #L = - np.nansum(TimeProfile(y, T_i[:,0]))
    return L

def Likelihood_quantile(y, T_i, tau, ts):
    less = T_i[y<T_i] - y[y<T_i]
    more = y[y>=T_i] - T_i[y>=T_i]

    R = (1-tau)*np.sum(less) + tau*np.sum(more)
    #log_Likelihood = exp
    return R

def TimeProfile(y,T_i):
    time_correct = y - T_i
    time_correct[time_correct<=-4] = -4
    p_time = TimeUncertainty(time_correct, 26)
    return p_time

def TimeUncertainty(tc, tau_d):
    TTS = 2.2
    tau_r = 1.6
    a1 = np.exp(((TTS**2 - tc*tau_d)**2-tc**2*tau_d**2)/(2*TTS**2*tau_d**2))
    a2 = np.exp(((TTS**2*(tau_d+tau_r) - tc*tau_d*tau_r)**2 - tc**2*tau_d**2*tau_r**2)/(2*TTS**2*tau_d**2*tau_r**2))
    a3 = np.exp(((TTS**2 - tc*tau_d)**2 - tc**2*tau_d**2)/(2*TTS**2*tau_d**2))*special.erf((tc*tau_d-TTS**2)/(np.sqrt(2)*tau_d*TTS))
    a4 = np.exp(((TTS**2*(tau_d+tau_r) - tc*tau_d*tau_r)**2 - tc**2*tau_d**2*tau_r**2)/(2*TTS**2*tau_d**2*tau_r**2))*special.erf((tc*tau_d*tau_r-TTS**2*(tau_d+tau_r))/(np.sqrt(2)*tau_d*tau_r*TTS))
    p_time = np.log(tau_d + tau_r) - 2*np.log(tau_d) + np.log(a1-a2+a3-a4)
    return p_time

def ReadPMT(geometry):
    f = open(geometry, 'r')
    #f = open(r"./PMT_1t.txt")
    line = f.readline()
    data_list = [] 
    while line:
        num = list(map(float,line.split()))
        data_list.append(num[1:4])
        line = f.readline()
    f.close()
    PMT_pos = np.array(data_list)
    return PMT_pos

def recon(fid, fout, *args):
    PMT_pos, event_count = args
    '''
    reconstruction

    fid: root reference file convert to .h5
    fout: output file
    '''
    # Create the output file and the group
    print(fid) # filename
    class ReconData(tables.IsDescription):
        EventID = tables.Int64Col(pos=0)    # EventNo
        # inner recon
        E_sph = tables.Float16Col(pos=1)        # Energy
        x_sph = tables.Float16Col(pos=2)        # x position
        y_sph = tables.Float16Col(pos=3)        # y position
        z_sph = tables.Float16Col(pos=4)        # z position
        t0 = tables.Float16Col(pos=5)       # time offset
        success = tables.Int64Col(pos=6)    # recon failure   
        Likelihood = tables.Float16Col(pos=7)
        

        # truth info
        x_truth = tables.Float16Col(pos=15)        # x position
        y_truth = tables.Float16Col(pos=16)        # y position
        z_truth = tables.Float16Col(pos=17)        # z position
        E_truth = tables.Float16Col(pos=18)        # z position
                        
        # unfinished
        tau_d = tables.Float16Col(pos=18)    # decay time constant

    # Create the output file and the group
    h5file = tables.open_file(fout, mode="w", title="OneTonDetector",
                            filters = tables.Filters(complevel=9))
    group = "/"
    # Create tables
    ReconTable = h5file.create_table(group, "Recon", ReconData, "Recon")
    recondata = ReconTable.row
    # Loop for event

    f = uproot.open(fid)
    # a = f['SimTriggerInfo']
    e = f['evt']
    print('evtNum:{}'.format(e.array('evtID').shape))
    for eid, chl, ht in zip(e.array('evtID'), e.array("pmtID"), e.array('hitTime')):
        print('pmtID:{},hittime:{}'.format(chl.shape,ht.shape))
        # select Large pmt 
        pe_array = np.zeros(np.size(PMT_pos[:,1])) # 
        fired_PMT = chl[(chl<largeidUp)&(ht<timelimit)]
        time_array= ht[(chl<largeidUp)&(ht<timelimit)]
        c, tempStore = np.unique(fired_PMT, return_counts=True)
        print('counts:{},max:{},hitTime:{}'.format(tempStore.shape, np.max(c),time_array.shape))
        pe_array[0:np.size(tempStore)] = tempStore
        print('processed:{}'.format(eid))
        # initial result
        result_vertex = np.empty((0,5)) # reconstructed vertex
        
        # Constraints
        E_min = -10
        E_max = 10
        tau_min = 0.01
        tau_max = 100
        t0_min = -300
        t0_max = 600

        # inner recon
        # initial value
        x0_in = np.zeros((5,))
        x0_in[0] = 2 # 0.8 + np.log(np.sum(pe_array)/60)
        x0_in[4] = np.mean(time_array) - 26

        x0_in[1] = np.sum(pe_array*PMT_pos[:,0])/np.sum(pe_array)/detRadius
        x0_in[2] = np.sum(pe_array*PMT_pos[:,1])/np.sum(pe_array)/detRadius
        x0_in[3] = np.sum(pe_array*PMT_pos[:,2])/np.sum(pe_array)/detRadius

        a = c2r(x0_in[1:4])
        print('sphere:{};Cartesian:{}'.format(a,x0_in[1:4]))
        #b = r2c(a)
        #print(x0_in[0][1:4],a,b)
        #exit()
        
        # not added yet
        x0 = np.hstack((x0_in[0], a, x0_in[4]))
        result_in = minimize(Likelihood, x0, method='SLSQP',bounds=((E_min, E_max), (0, 1), (None, None), (None, None), (None, None)), args = (coeff_time, coeff_pe, PMT_pos, fired_PMT, time_array, pe_array, cut_time, cut_pe),options={'maxiters': 5000})

        in2 = r2c(result_in.x[1:4])*detRadius
        recondata['x_sph'] = in2[0]
        recondata['y_sph'] = in2[1]
        recondata['z_sph'] = in2[2]
        recondata['E_sph'] = result_in.x[0]
        recondata['success'] = result_in.success
        recondata['Likelihood'] = result_in.fun

        #vertex = result.x[1:4]
        #print(result.x, np.sqrt(np.sum(vertex**2)))
        recondata.append()
        print('result')
        print('success:{};message:{};value of fun:{};niter:{}'.format(result_in.success, result_in.message, result_in.fun, result_in.nit))
        print('%d: [%+.2f, %+.2f, %+.2f] radius: %+.2f, Likelihood: %+.2f' % (event_count, in2[0], in2[1], in2[2], norm(in2), result_in.fun))
        print('-'*60)
        event_count = event_count + 1
    # Flush into the output file
    ReconTable.flush()
    h5file.close()

# Automatically add multiple root files created a program with max tree size limitation.

psr = argparse.ArgumentParser()
psr.add_argument("-o", dest='opt', help="output")
psr.add_argument("-O", dest='order', help="output")
psr.add_argument("-d", dest='orderFile', nargs='+', help="order file")
psr.add_argument("-t", dest='root', help="root file")
psr.add_argument('-T', dest='tup', help='max t ns')
psr.add_argument('-g', dest='geo', help="geometry")
args = psr.parse_args()
# Read PMT position

geometry = args.geo
PMT_pos = ReadPMT(geometry)
event_count = 0
largeidUp = 17613
timelimit = np.int(args.tup)
# Reconstruction
fid = args.root # input file .root
fout = args.opt # output file .h5
coeff_pe, coeff_time,\
    cut_pe, fitcut_pe, cut_time, fitcut_time\
    = load_coeff(*args.orderFile, args.order)

args = PMT_pos, event_count
recon(fid, fout, *args)
