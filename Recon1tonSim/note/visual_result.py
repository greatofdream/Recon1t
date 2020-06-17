import numpy as np
import matplotlib.pyplot as plt
import tables
import h5py
from scipy.spatial.distance import pdist, squareform
import os, argparse,uproot
from matplotlib.backends.backend_pdf import PdfPages

psr = argparse.ArgumentParser()
psr.add_argument("-o", dest='opt', help="output")
psr.add_argument('ipt', nargs='+', help="input")
psr.add_argument('-p', dest='pos', nargs='+', help="position truth")
psr.add_argument("-z", dest='z', nargs='+', help="z axis")

args = psr.parse_args()
zaxis = np.array([np.int(i) for i in args.z])

pdf = PdfPages(args.opt)
for z, filename, truthfile in zip(zaxis, args.ipt, args.pos):
    print('{}'.format(filename))
    ftruth = uproot.open(truthfile)
    e = ftruth['prmtrkdep']
    with h5py.File(filename, 'r') as ipt:
        fig = plt.figure(figsize=(1000,500))
        plt.subplot(1,2,1)
        print(e.array('edep'))
        print(ipt['Recon']['E_sph_in'][:])
        print(e.array('edepX'))
        print(ipt['Recon']['x_sph_in'][:])
        plt.hist(e.array('edep')-ipt['Recon']['E_sph_in'][:])
        plt.title('distribution of E-error;z:{}'.format(z))
        plt.xlabel('E-error')
        plt.ylabel('Entries')
        plt.subplot(1,2,2)
        plt.title('distribution of R-error;z:{}'.format(z))
        plt.hist(np.sqrt(e.array('edepX')**2+e.array('edepX')**2+e.array('edepX')**2)-
                    np.sqrt(ipt['Recon']['x_sph_in'][:]**2+ipt['Recon']['y_sph_in'][:]**2+ipt['Recon']['z_sph_in'][:]**2))
        plt.xlabel('R-error')
        plt.ylabel('Entries')
        pdf.savefig()
        plt.close()
pdf.close()