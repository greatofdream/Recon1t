import uproot, argparse, h5py, itertools as it, numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

psr = argparse.ArgumentParser()
psr.add_argument("-o", dest='opt', help="output")
psr.add_argument("-z", dest='z', nargs='+', help="z axis")
psr.add_argument('ipt', nargs='+', help="input")
args = psr.parse_args()

zaxis = np.array([np.int(i) for i in args.z])
pdf = PdfPages(args.opt)

for z, filename in zip(zaxis, args.ipt):
    print('{},{}'.format(filename,z))
    f = uproot.open(filename)
    e = f['SIMEVT']

    fig = plt.figure(figsize=(1000,500))
    
    plt.subplot(1,2,1)
    plt.plot(e.array('evtID'), e.array('EvtTime_Sec'))
    plt.xticks(e.array('evtID'))
    plt.title('evtid-evtTime;z:{}'.format(z))
    plt.xlabel('evtID')
    plt.ylabel('EvtTime_sec')
    plt.subplot(1,2,2)
    plt.title('evtid-pmtNum;z:{}'.format(z))
    plt.plot(e.array('evtID'), e.array('fired_PMT_Num'))
    plt.xticks(e.array('evtID'))
    plt.xlabel('evtID')
    plt.ylabel('fired_pmt')
    pdf.savefig()
    plt.close()
pdf.close()