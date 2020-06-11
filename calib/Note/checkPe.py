
# 检查pe分布
import numpy as np
import matplotlib.pyplot as plt
import tables
import h5py
import os, sys

from matplotlib.backends.backend_pdf import PdfPages
import argparse 
psr = argparse.ArgumentParser()

psr.add_argument("-o", dest='opt', help="output")
psr.add_argument("-f", dest='file', nargs='+', help="pe")
# psr.add_argument("-t", dest='time', nargs='+', help="time")
psr.add_argument("-r", dest='radius', nargs='+', help="radius")
args = psr.parse_args()

files = args.file
ra = np.array([int(i) for i in args.radius])

rewrite = False
pmtNumber = 17613
total_pe = np.zeros((ra.shape[0], pmtNumber))
print(total_pe.shape)

pdf = PdfPages(args.opt)
if rewrite:
    for i, (radius,filename) in enumerate(zip(ra, files)):
        with h5py.File(filename) as ipt:
            EventID = ipt['GroundTruth']['EventID']
            ChannelID = ipt['GroundTruth']['ChannelID']
            event_pe = np.zeros((max(EventID), pmtNumber))
            for k in np.arange(1, max(EventID)):
                hit = ChannelID[EventID == k]
                tabulate = np.bincount(hit)
                event_pe[k-1, 0:np.size(tabulate)] = tabulate
        total_pe[i, :] = np.mean(event_pe, axis=0)
    with h5py.File(args.opt+'.h5','w') as opt:
        opt.create_dataset('totalpe',data=total_pe,
                            compression="gzip", shuffle=True)
else:
    with h5py.File(args.opt+'.h5','r') as ipt:
        total_pe = ipt['totalpe'][:]
sum_pe = np.mean(total_pe, axis=1)

plt.figure()
plt.boxplot(sum_pe)
plt.title('totalpe box')
pdf.savefig()
plt.close()
plt.hist(sum_pe)
plt.title('totalpe distribution')
plt.xlabel('mean pe for each pmt in different z')
plt.ylabel('N')
pdf.savefig()
plt.close()

plt.plot(ra, sum_pe)
plt.title('mean pe for each pmt vs z')
pdf.savefig()
plt.xlabel('mean pe for each pmt')
plt.ylabel('z')
plt.close()
for i in range(0,17613,100):
    plt.plot(ra, total_pe[:,i])
pdf.savefig()

pdf.close()