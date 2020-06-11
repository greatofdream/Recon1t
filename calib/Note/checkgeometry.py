import numpy as np
import matplotlib.pyplot as plt
import tables
import h5py
import os, sys

from matplotlib.backends.backend_pdf import PdfPages
import argparse 
psr = argparse.ArgumentParser()
psr.add_argument("-o", dest='opt', help="output")
psr.add_argument("-g", dest='geo', help="geometry")
args = psr.parse_args()
pmtPosition = []
pmtNumber=17613
with open(args.geo) as ipt:
    for line in ipt:
        pmtPosition.append(list(map(float, line.split())))
pmtPosition = np.array(pmtPosition)
print(pmtPosition[pmtPosition[:,2]<0].shape)
print(np.double(pmtPosition[pmtPosition[:,2]<0].shape[0])/pmtNumber)
exit(0)
pdf = PdfPages(args.opt)
plt.hist(pmtPosition[:,2])
plt.title('z')
pdf.savefig()
plt.close()

plt.hist(pmtPosition[:,0])
plt.title('x')
pdf.savefig()
plt.close()

plt.hist(pmtPosition[:,1])
plt.title('y')
pdf.savefig()
plt.close()
pdf.close()