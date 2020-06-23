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
EerrorMean = np.zeros(zaxis.shape)
EerrorStd = np.zeros(zaxis.shape)
XerrorMean = np.zeros(zaxis.shape)
YerrorMean = np.zeros(zaxis.shape)
ZerrorMean = np.zeros(zaxis.shape)
RerrorMean = np.zeros(zaxis.shape)
RerrorStd = np.zeros(zaxis.shape)
for i, (z, filename, truthfile) in enumerate(zip(zaxis, args.ipt, args.pos)):
    print('{}'.format(filename))
    ftruth = uproot.open(truthfile)
    e = ftruth['prmtrkdep']
    with h5py.File(filename, 'r') as ipt:
        fig = plt.figure()
        
        print(e.array('edep'))
        print(ipt['Recon']['E_sph'][:])
        print(e.array('edepX'))
        print(ipt['Recon']['x_sph'][:])
        
        L_in = ipt['Recon']['Likelihood'][:]

        recon_pos = np.zeros((np.size(L_in), 3))
        recon_E = np.zeros((np.size(L_in,)))

        recon_pos[:, 0] = ipt['Recon']['x_sph'][:]
        recon_pos[:, 1] = ipt['Recon']['y_sph'][:]
        recon_pos[:, 2] = ipt['Recon']['z_sph'][:]
        recon_E[:] = ipt['Recon']['E_sph'][:]

        truth_x = np.array([i[0] for i in e.array('edepX')])
        truth_y = np.array([i[0] for i in e.array('edepX')])
        truth_z = np.array([i[0] for i in e.array('edepX')])
        truth_E = np.array([i[0] for i in e.array('edep')])
        plt.subplot(1,1,1)
        plt.hist(recon_E)
        plt.axvline(np.average(e.array('edep')), color='green')
        plt.title('distribution of E-error;z:{}'.format(z))
        plt.xlabel('recon-E/MeV')
        plt.ylabel('Entries')
        pdf.savefig()
        plt.close()

        plt.subplot(1,1,1)
        plt.title('distribution of R;z:{}'.format(z))
        print(np.sqrt(recon_pos[:,0]**2+recon_pos[:,1]**2+recon_pos[:,2]**2))
        plt.hist(np.sqrt(recon_pos[:,0]**2+recon_pos[:,1]**2+recon_pos[:,2]**2))
        print(np.sqrt(e.array('edepX')**2+e.array('edepY')**2+e.array('edepZ')**2))
        plt.axvline(np.average(np.sqrt(e.array('edepX')**2+e.array('edepY')**2+e.array('edepZ')**2)), color='green')
        plt.xlabel('R/mm')
        plt.ylabel('Entries')
        pdf.savefig()
        plt.close()
        EerrorMean[i] = np.average(recon_E - truth_E)
        EerrorStd[i] = np.std(recon_E - truth_E)
        
        RerrorMean[i] = np.average(np.sqrt(recon_pos[:,0]**2+recon_pos[:,1]**2+recon_pos[:,2]**2) - np.sqrt(truth_x**2+truth_y**2+truth_z**2))
        RerrorStd[i] = np.std(np.sqrt(recon_pos[:,0]**2+recon_pos[:,1]**2+recon_pos[:,2]**2) - np.sqrt(truth_x**2+truth_y**2+truth_z**2))
plt.subplot(1, 1, 1)

plt.title('Eerror-z')
plt.errorbar(zaxis, EerrorMean, yerr=EerrorStd, fmt='.')
plt.xlabel('z/mm')
plt.ylabel('Eerror')
pdf.savefig()
plt.close()
plt.subplot(1,1,1)
plt.title('Rerror-z')
plt.errorbar(zaxis, RerrorMean, yerr=RerrorStd, fmt='.')
plt.xlabel('z/mm')
plt.ylabel('Rerror')
pdf.savefig()
plt.close()
pdf.close()