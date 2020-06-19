import uproot, argparse, numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt 
psr = argparse.ArgumentParser()
psr.add_argument("-o", dest='opt', help="output")
psr.add_argument("-z", dest='z', nargs='+', help="z axis")
psr.add_argument('ipt', nargs='+', help="input")
args = psr.parse_args()
zaxis = np.array([np.int(i) for i in args.z])
plot = True
if plot:
    pdf = PdfPages(args.opt)

largepmtid = 17612
for z, filename in zip(zaxis, args.ipt):
    print('{},{}'.format(filename,z))
    f = uproot.open(filename)
    e = f['evt']

    for eid, chl, ht in zip(e.array('evtID'),e.array('pmtID'), e.array('hitTime')):
        if plot:
            fig = plt.figure()
            plt.subplot(1,1,1)
            plt.hist(ht[chl<largepmtid])
            plt.yscale('log')
            plt.title('evtid{}-hitTime:z{}'.format(eid, z))
            plt.xlabel('hitTime')
            plt.ylabel('Entries')
            pdf.savefig()
            plt.close()
        print('evtid{}-hitTime:z{},max{},>5000:{}'.format(eid, z, np.max(ht[chl<largepmtid]),ht[(chl<largepmtid) & (ht>5000)]))
if plot:
    pdf.close()