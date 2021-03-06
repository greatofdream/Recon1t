import numpy as np
import scipy, h5py
import scipy.stats as stats
import math
import ROOT
import os,sys
import tables
import uproot, argparse
import scipy.io as scio
from scipy.optimize import minimize
from scipy import special

# physical constant
Light_yield = 4285*0.88 # light yield
Att_LS = 18 # attenuation length of LS
Att_Wtr = 300 # attenuation length of water
tau_r = 1.6 # fast time constant
TTS = 5.5/2.355
QE = 0.20
PMT_radius = 0.254
c = 2.99792e8
n = 1.48
shell = 0.65 # Acrylic

def ReconML():
    fun = lambda x: Likelihood(x)
    return fun

def Likelihood(args):
    Energy,\
    x,\
    y,\
    z,\
    t,\
    tau_d\
    = args
    global shell, PE, time_array, fired_PMT
    distance = np.sqrt((PMT_pos[:,0] - x)**2 + (PMT_pos[:,1] - y)**2 + (PMT_pos[:,2] - z)**2)
    # LS distance
    d1 = np.tile(np.array([x,y,z]),[len(PMT_pos[:,1]),1])
    d2 = PMT_pos
    d3 = d2 - d1
    # cons beyond shell ?
    lmbd = (-2*np.sum(d3*d1,1) \
        + np.sqrt(4*np.sum(d3*d1,1)**2 \
        - 4*np.sum(d3**2,1)*(-np.abs((np.sum(d1**2,1)-shell**2))))) \
        /(2*np.sum(d3**2,1))
    lmbd[lmbd>=1] = 1
    expect = Energy*Light_yield*np.exp(-distance*lmbd/Att_LS - distance*(1-lmbd)/Att_Wtr)*SolidAngle(x,y,z,distance)*QE
    p_pe = - expect + PE*np.log(expect) - np.log(special.factorial(PE))
    # p_pe = - np.log(stats.poisson.pmf(PE, expect))
    Likelihood_pe = - np.nansum(p_pe)


    p_time = TimeProfile(time_array, distance[fired_PMT], tau_d, t)
    Likelihood_time = - np.nansum(p_time)
    
    Likelihood_total = Likelihood_pe + Likelihood_time
    return Likelihood_total

def SolidAngle(x, y, z, distance):
    radius_O1 = PMT_radius # PMT bottom surface
    radius_O2 = 0.315 # PMT sphere surface
    PMT_vector = - PMT_pos/np.transpose(np.tile(np.sqrt(np.sum(PMT_pos**2,1)),[3,1]))
    O1 = np.tile(np.array([x,y,z]),[len(PMT_pos[:,0]),1])
    O2 = PMT_pos
    d1 = np.sqrt(radius_O2**2 - radius_O1**2)
    O3 = (O2 + PMT_vector*d1)
    flight_vector = O2 - O1
    d2 = np.sqrt(np.sum(flight_vector**2,1))
    O4 = O2 - flight_vector/ \
        np.transpose(np.tile(d2*np.sqrt(radius_O2**2*(d2**2 - radius_O2**2)/d2**2),[3,1]))
    # Helen formula
    a = np.sqrt(np.sum((O4-O2)**2,1))
    b = np.sqrt(np.sum((O4-O3)**2,1))
    c = np.sqrt(np.sum((O3-O2)**2,1))
    p = (a+b+c)/2
    d = 2*a*b*c/(4*np.sqrt(p*(p-a)*(p-b)*(p-c)))
    '''
    # this part influence the speed!
    data = np.array([a,b,c])
    sorted_cols = []
    for col_no in range(data.shape[1]):
        sorted_cols.append(data[np.argsort(data[:,col_no])][:,col_no])
    sorted_data = np.column_stack(sorted_cols)

    a = sorted_data[0,:]
    b = sorted_data[1,:]
    c = sorted_data[2,:]
    d[a+b-c<=1*10**(-10)] = 1 # avoid inf
    '''
    d = np.transpose(d)
    chord = 2*np.sqrt(radius_O2**2 - d[d<radius_O2]**2)

    theta1 = np.sum(PMT_vector*flight_vector,1)/np.sqrt(np.sum(PMT_vector**2,1)*np.sum(flight_vector**2,1))
    add_area = 1/3* \
        (radius_O2 - radius_O1*np.abs(theta1[d<radius_O2]) \
        - np.sqrt(radius_O2**2 - radius_O1**2)*np.abs(np.sin(np.arccos(theta1[d<radius_O2]))))*chord
    Omega = (1-d2/np.sqrt(d2**2+radius_O1*np.abs(theta1)))/2
    Omega[d<radius_O2] = Omega[d<radius_O2] + add_area/(4*d2[d<radius_O2]**2)
    return Omega

def TimeProfile(time_array, distance, tau_d, t):
    time_correct = time_array - distance/(c/n)*1e9 - t
    time_correct[time_correct<=-8] = -8
    p_time = TimeUncertainty(time_correct, tau_d)
    return p_time

def TimeUncertainty(tc, tau_d):
    a1 = np.exp(((TTS**2 - tc*tau_d)**2-tc**2*tau_d**2)/(2*TTS**2*tau_d**2))
    a2 = np.exp(((TTS**2*(tau_d+tau_r) - tc*tau_d*tau_r)**2 - tc**2*tau_d**2*tau_r**2)/(2*TTS**2*tau_d**2*tau_r**2))
    a3 = np.exp(((TTS**2 - tc*tau_d)**2 - tc**2*tau_d**2)/(2*TTS**2*tau_d**2))*special.erf((tc*tau_d-TTS**2)/(np.sqrt(2)*tau_d*TTS))
    a4 = np.exp(((TTS**2*(tau_d+tau_r) - tc*tau_d*tau_r)**2 - tc**2*tau_d**2*tau_r**2)/(2*TTS**2*tau_d**2*tau_r**2))*special.erf((tc*tau_d*tau_r-TTS**2*(tau_d+tau_r))/(np.sqrt(2)*tau_d*tau_r*TTS))
    p_time  = np.log(tau_d + tau_r) - 2*np.log(tau_d) + np.log(a1-a2+a3-a4)
    
    return p_time

def con(args):
    Emin,\
    radius\
    = args

    cons = ({'type': 'ineq', 'fun': lambda x: x[0] - Emin},\
    {'type': 'ineq', 'fun': lambda x: radius**2 - (x[1]**2 + x[2]**2+x[3]**2)},\
    {'type': 'ineq', 'fun': lambda x: (x[5] - 26)*(26-x[5])},\
    {'type': 'ineq', 'fun': lambda x: (x[4] + 100)*(300-x[4])})
    return cons

def recon_drc(time_array, fired_PMT, recon_vertex):
    time_corr = time_array - np.sum(PMT_pos[fired_PMT,1:4]-np.tile(recon_vertex[0,1:4],[len(fired_PMT),1]))/(3*10**8)
    index = np.argsort(time_corr)
    fired_PMT_sorted = fired_PMT[index]
    fired_PMT_sorted = fired_PMT_sorted[0:int(np.floor(len(fired_PMT_sorted)/10))]
    drc = np.sum(PMT_pos[fired_PMT_sorted,1:4],0)/len(fired_PMT_sorted)
    return drc

def recon(fid, fout):
    global event_count,shell,PE,time_array,PMT_pos, fired_PMT
    '''
    reconstruction

    fid: root reference file
    fout: output file in this step
    '''

    # Create the output file and the group
    rootfile = ROOT.TFile(fid)
    #TruthChain = rootfile.Get('SimTriggerInfo')
    print(fid)
    '''
    class ChargeData(tables.IsDescription):
        ChannelID = tables.Float64Col(pos=0)
        Time = tables.Float16Col(pos=1)
        PE = tables.Float16Col(pos=2)
        Charge = tables.Float16Col(pos=2)
    '''
    class ReconData(tables.IsDescription):
        EventID = tables.Int64Col(pos=0)
        x = tables.Float16Col(pos=1)
        y = tables.Float16Col(pos=2)
        z = tables.Float16Col(pos=3)
        t0 = tables.Float16Col(pos=4)
        E = tables.Float16Col(pos=5)
        tau_d = tables.Float16Col(pos=6)
        success = tables.Int64Col(pos=7)

    # Create the output file and the group
    h5file = tables.open_file(fout, mode="w", title="OneTonDetector",
                            filters = tables.Filters(complevel=9))
    group = "/"

    # Create tables
    '''
    ChargeTable = h5file.create_table(group, "Charge", ChargeData, "Charge")
    Charge = ChargeTable.row
    '''
    ReconTable = h5file.create_table(group, "Recon", ReconData, "Recon")
    recondata = ReconTable.row
    '''
    # Loop for ROOT files. 
    data = ROOT.TChain("SimpleAnalysis")
    data.Add(fid)
    '''
    # Loop for event
    '''
    for event in data:
        print(event)
        EventID = event.TriggerNo
        print(EventID)
    '''  
    '''  
    psr = argparse.ArgumentParser()
    psr.add_argument("-o", dest='opt', help="output")
    psr.add_argument('ipt', help="input")
    args = psr.parse_args()

    f = uproot.open(args.ipt)
    '''
    f = uproot.open(fid)
    a = f['SimpleAnalysis']
    for tot, chl, PEl, Pkl, nPl in zip(a.array("TotalPE"),
                    a.array("ChannelInfo.ChannelId"),
                    a.array('ChannelInfo.PE'),
                    a.array('ChannelInfo.PeakLoc'),
                    a.array('ChannelInfo.nPeaks')):

    #print("=== TotalPE: {} ===".format(tot))
    #for ch, PE, pk, np in zip(chl, PEl, Pkl, nPl):
    #   print(ch, PE, pk, np)
        CH = np.zeros(np.size(PMT_pos[:,1]))
        PE = np.zeros(np.size(PMT_pos[:,1]))
        fired_PMT = np.zeros(0)
        TIME = np.zeros(0)
        for ch, pe, pk, npk in zip(chl, PEl, Pkl, nPl):
            PE[ch] = pe
            TIME = np.hstack((TIME, pk))
            fired_PMT = np.hstack((fired_PMT, ch*np.ones(np.size(pk))))
        # print(TIME, fired_PMT)
        fired_PMT = fired_PMT.astype(int)
        time_array = TIME
        
        '''
        for ChannelInfo in event.ChannelInfo:
            Charge['ChannelID'] = ChargeInfo.ChannelID
            Charge['Time'] =  ChannelInfo.Peak
            Charge['PE'] =  ChannelInfo.PE
            Charge['Charge'] =  ChannelInfo.Charge
            Charge.append()

            PE = ChannelInfo.nPeaks
            Time =  ChannelInfo.Peak
            ChannelID = ChargeInfo.ChannelID
        '''

        result_recon = np.empty((0,6))
        result_drc = np.empty((0,3))
        result_tdrc = np.empty((0,3))

        # initial value
        x0 = np.zeros((1,6))
        x0[0][0] = PE.sum()/300
        x0[0][1] = np.sum(PE*PMT_pos[:,0])/np.sum(PE)
        x0[0][2] = np.sum(PE*PMT_pos[:,1])/np.sum(PE)
        x0[0][3] = np.sum(PE*PMT_pos[:,2])/np.sum(PE)
        x0[0][4] = 95
        x0[0][5] = 26

        # Constraints
        Emin = 0.01
        args = (Emin, shell)
        cons = con(args)

        # reconstruction
        result = minimize(ReconML(), x0, method='SLSQP', constraints=cons)

        # result
        print(event_count, result.x, result.success)
        event_count = event_count + 1
        recondata['EventID'] = event_count
        recondata['x'] = result.x[1]
        recondata['y'] = result.x[2]
        recondata['z'] = result.x[3]
        recondata['E'] = result.x[0]
        recondata['t0'] = result.x[4]
        recondata['tau_d'] = result.x[5]
        recondata['success'] = result.success
        recondata.append()

        # print(np.sum(result_drc*truth_px)/np.sqrt(np.sum(result_drc**2)*np.sum(truth_px**2)))

    # Flush into the output file
    # ChargeTable.flush()
    ReconTable.flush()
    h5file.close()

# Automatically add multiple root files created a program with max tree size limitation.
if len(sys.argv)!=3:
    print("Wront arguments!")
    print("Usage: python ConvertTruth.py MCFileName outputFileName")
    sys.exit(1)

f = open(r"./PMT1t.txt")
#f = open(r"./PMT_1t_sphere.txt")

line = f.readline()
data_list = []
while line:
    num = list(map(float,line.split()))
    data_list.append(num)
    line = f.readline()
f.close()
global PMT_pos, event_count
event_count = 0
PMT_pos = np.array(data_list)

fid = sys.argv[1]
fout = sys.argv[2]

ROOT.PyConfig.IgnoreCommandLineOptions = True
recon(fid, fout)