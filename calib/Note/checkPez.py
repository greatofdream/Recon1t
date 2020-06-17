
# 检查每个点pe分布
import numpy as np
import matplotlib.pyplot as plt
import tables
import h5py
import os, sys

from matplotlib.backends.backend_pdf import PdfPages
import argparse 
psr = argparse.ArgumentParser()

psr.add_argument("-o", dest='opt', help="output")
psr.add_argument("-f", dest='file', help="h5file")
# psr.add_argument("-t", dest='time', nargs='+', help="time")
psr.add_argument("-r", dest='radius', help="radius")
psr.add_argument("-g", dest='geo', help="geometry")
args = psr.parse_args()

filename = args.file
ra = int(args.radius)
rewrite = True
pmtPosition = []
with open(args.geo) as ipt:
    for line in ipt:
        pmtPosition.append(list(map(float, line.split())))
pmtPosition = np.array(pmtPosition)
pmtNumber = pmtPosition.shape[0]
pdf = PdfPages(args.opt)
event_pe=[]
if rewrite:
    with h5py.File(filename) as ipt:
        EventID = ipt['GroundTruth']['EventID']
        ChannelID = ipt['GroundTruth']['ChannelID']
        event_pe = np.zeros((max(EventID)+1, pmtNumber))
        for k in np.arange(0, max(EventID)+1):
            hit = ChannelID[EventID == k]
            print('event {}:hit shape{}'.format(k,hit.shape))
            tabulate = np.bincount(hit)
            event_pe[k-1, 0:np.size(tabulate)] = tabulate

    with h5py.File(args.opt+'.h5','w') as opt:
        opt.create_dataset('eventpe',data=event_pe,
                            compression="gzip", shuffle=True)
else:
    with h5py.File(args.opt+'.h5','r') as ipt:
        event_pe = ipt['eventpe'][:]

print(event_pe.shape)

sum_pe = np.mean(event_pe, axis=0)
print(np.sum(event_pe, axis=1))

plt.figure()
plt.hist(np.sum(event_pe, axis=1))
plt.title('total pe of different event')
pdf.savefig()
plt.close()

plt.figure()
plt.boxplot(np.sum(event_pe, axis=1))
plt.title('total pe of different event')
pdf.savefig()
plt.close()

plt.figure()
plt.boxplot(sum_pe)
plt.title('average pe of all event:box graph')
pdf.savefig()
plt.close()
plt.hist(sum_pe)
plt.title('average pe of all event:distribution')
plt.xlabel('mean pe of event for each pmt')
plt.ylabel('N')
pdf.savefig()
plt.close()

plt.scatter(pmtPosition[:,2], sum_pe,marker=".")
plt.title('mean pe for each pmt vs z of pmt')
plt.xlabel('mean pe for each pmt')
plt.ylabel('z of pmt')
pdf.savefig()
plt.close()

pdf.close()