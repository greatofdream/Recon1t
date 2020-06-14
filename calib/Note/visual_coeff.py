import numpy as np
import matplotlib.pyplot as plt
import tables
import h5py
import os, sys
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages
import argparse 
psr = argparse.ArgumentParser()
psr.add_argument('ipt', help="input")
psr.add_argument("-o", dest='opt', help="output")
psr.add_argument("-p", dest='pe', nargs='+', help="pe")
psr.add_argument("-t", dest='time', nargs='+', help="time")
psr.add_argument("-r", dest='radius', nargs='+', help="radius")
args = psr.parse_args()

order = args.ipt
peFiles = args.pe
timeFiles = args.time


# fit odd order, even order is 0
def odd_func(x, a, b, c, d, e):
    return a * x**1 + b * x**3 + c * x**5 + d * x**7 + e * x**9
# fit even order, even order is 0
def even_func(x, a, b, c, d, e):
    return a * x**0 + b * x**2 + c * x**4 + d * x**6 + e * x**8
# load data
def LoadDataPE(path, radius, order):
    data = []
    filename = path
    h = tables.open_file(filename,'r')

    coeff = 'coeff' + str(order)
    mean = 'mean' + str(order)
    predict = 'predict' +str(order)
    rate = 'rate' + str(order)
    hinv = 'hinv' + str(order)
    chi = 'chi' + str(order)
    
    a = np.array(h.root[coeff][:])
    b = np.array(h.root[mean][:])
    c = np.array(h.root[predict][:])
    try:
        d = np.array(h.root[rate][:])
    except:
        d = np.array(0)
    e = np.array(h.root[hinv][:])
    f = np.array(h.root[chi][:])
    
    data.append(np.array(np.array((a,b,c,d,e,f))))
    return data


def LoadFileTime(path, radius, order):
    data = []
    filename = path
    h = tables.open_file(filename,'r')

    coeff = 'coeff' + str(order)
    hinv = 'hinv' + str(order)

    a = np.array(h.root[coeff][:])
    e = np.array(h.root[hinv][:])
    #data.append(np.array(np.array((a,e))))
    return (a, e)

    ## gather the data
path = '../coeff_pe_1t_339MeV/'
#ra = np.arange(+0.651, -0.65, -0.01)
#ra = np.arange(16000, -16001, -1000)
#ra = np.append(np.arange(16000,6001,-1000),(np.arange(-5000,-16001,-1000)))
ra = np.array([int(i) for i in args.radius])
sigmaFactor = np.array([ 1/10 if i==-14000 or i==16000 else 1/np.sqrt(5) for i in ra])
detRadius = 17700

coeff_pe = []
mean = []
predict = []
rate = []
hinv = []
chi = []
sigma = []
for radius,filename in zip(ra, peFiles):
    #str_radius = '%+.2f' % radius
    str_radius = '{}'.format(radius)
    
    k = LoadDataPE(filename, str_radius, order)
    k.append(np.array(radius))
    coeff_pe = np.hstack((coeff_pe, np.array(k[0][0])))
    mean = np.hstack((mean, np.array(k[0][1])))
    predict = np.hstack((predict, np.array(k[0][2][:,0])))
    #rate = np.hstack((rate,np.array(k[0][3])))
    #hinv = np.hstack((hinv,np.array(k[0][4])))
    sigma = np.hstack((sigma,np.sqrt(-np.diagonal(k[0][4]))))
    chi = np.hstack((chi,np.array(k[0][5])))
#print('coeff_pe value')
#print(coeff_pe)
coeff_pe = np.reshape(coeff_pe,(-1,np.size(ra)),order='F')
#print(coeff_pe)

mean = np.reshape(mean,(-1,np.size(ra)),order='F')
predict = np.reshape(predict,(-1,np.size(ra)),order='F')
chi = np.reshape(chi,(-1,np.size(ra)),order='F')
sigma = np.reshape(sigma, (-1,np.size(ra)),order='F')
#sigma = coeff_pe*sigmaFactor



pdf = PdfPages(args.opt)
#ra = np.arange(+0.651, -0.65, -0.01)
for i in np.arange(np.size(coeff_pe[:,0])):
    plt.figure(dpi=150)
    
    # segmented
    bd_1 = 0.80
    bd_2l = 0.50 
    bd_2r = 0.80
    bd_3 = 0.7
    
    fit_max = 5
    data = np.nan_to_num(coeff_pe[i,:])
    # x = ra/0.65
    x = ra/detRadius
    #plt.plot(x, data,'.')
    plt.errorbar(x, data, yerr=sigma[i,:], fmt='.')
    index1 = (x<=bd_1) & (x>=-bd_1) & (x!=0)

    if(i%2==1):
        popt1, pcov = curve_fit(odd_func, x[index1], data[index1])
        output1 = odd_func(x[index1], *popt1)
    else:
        popt1, pcov = curve_fit(even_func, x[index1], data[index1])
        output1 = even_func(x[index1], *popt1)

    index2 = (np.abs(x)<=bd_2r) & (np.abs(x)>=bd_2l)
    if(i%2==1):
        popt2, pcov = curve_fit(odd_func, x[index2], data[index2])
        output2 = odd_func(x[index2], *popt2)
        #plt.plot(x[index1], odd_func(x[index1], *popt), 'r-')
    else:
        popt2, pcov = curve_fit(even_func, x[index2], data[index2])
        #plt.plot(x[index1], even_func(x[index1], *popt), 'r-')
        output2 = even_func(x[index2], *popt2)

    index3 = (x >= bd_3) | (x <= - bd_3)
    if(i%2==1):
        popt3, pcov = curve_fit(odd_func, x[index3], data[index3])
        output3 = odd_func(x[index3], *popt3)
    else:
        popt3, pcov = curve_fit(even_func, x[index3], data[index3])
        output3 = even_func(x[index3], *popt3)

    #x_total = np.hstack((x[index1],x[index2],x[index3]))
    #y_total = np.hstack((output1,output2,output3))
    #x_total = np.hstack((x[index1],x[index3]))
    #y_total = np.hstack((output1,output3))
    #index = np.argsort(x_total)

    plt.plot(x[index1],output1)
    plt.plot(x[index3][x[index3]>0],output3[x[index3]>0], color='g')
    plt.plot(x[index3][x[index3]<0],output3[x[index3]<0], color='g')    
    #plt.text(0,0.5,r'fit: z^1=%2.3f, z^3=%2.3f, z^5=%2.3f, z^7=%2.3f, z^9=%2.3f' % tuple(popt))
    plt.xlabel('Relative Radius')
    plt.ylabel('Coefficients')
    plt.title('pe:'+str(i)+'-th Legendre coeff')
    plt.legend(['raw','fit_inner','fit_outer'])
    pdf.savefig()
    #plt.show()
    plt.close()

order = 10
coeff_time = []
path = '../coeff_time_1t_339MeV/'

for radius,filename in zip(ra,timeFiles):
    # str_radius = '%+.2f' % radius
    str_radius = '{}'.format(radius)
    
    a, e = LoadFileTime(filename, str_radius, order)
    #k.append(np.array(radius))
    hinv = np.hstack((hinv,np.sqrt(-np.diagonal(e))))
    coeff_time = np.hstack((coeff_time,np.array(a)))

coeff_time = np.reshape(coeff_time,(-1,np.size(ra)),order='F')
hinv = np.reshape(hinv, (-1,np.size(ra)),order='F')
# ra = np.arange(+0.651, -0.65, -0.01)
for i in np.arange(np.size(coeff_time[:,0])):
    
    # segmented
    bd_1 = 0.80
    bd_2l = 0.50 
    bd_2r = 0.80
    bd_3 = 0.7
    
    fit_max = 5
    data = np.nan_to_num(coeff_time[i,:])
    #x = ra/0.65
    x = ra/detRadius

    index1 = (x<=bd_1) & (x>=-bd_1) & (x!=0)

    if(i%2==1):
        popt1, pcov = curve_fit(odd_func, x[index1], data[index1])
        output1 = odd_func(x[index1], *popt1)
    else:
        popt1, pcov = curve_fit(even_func, x[index1], data[index1])
        output1 = even_func(x[index1], *popt1)

    index2 = (np.abs(x)<=bd_2r) & (np.abs(x)>=bd_2l)
    if(i%2==1):
        popt2, pcov = curve_fit(odd_func, x[index2], data[index2])
        output2 = odd_func(x[index2], *popt2)
        #plt.plot(x[index1], odd_func(x[index1], *popt), 'r-')
    else:
        popt2, pcov = curve_fit(even_func, x[index2], data[index2])
        #plt.plot(x[index1], even_func(x[index1], *popt), 'r-')
        output2 = even_func(x[index2], *popt2)

    index3 = (x >= bd_3) | (x <= - bd_3)
    if(i%2==1):
        popt3, pcov = curve_fit(odd_func, x[index3], data[index3])
        output3 = odd_func(x[index3], *popt3)
    else:
        popt3, pcov = curve_fit(even_func, x[index3], data[index3])
        output3 = even_func(x[index3], *popt3)

    #x_total = np.hstack((x[index1],x[index2],x[index3]))
    #y_total = np.hstack((output1,output2,output3))
    #x_total = np.hstack((x[index1],x[index3]))
    #y_total = np.hstack((output1,output3))
    plt.figure(dpi=150)
    #plt.plot(x, data,'.')
    plt.errorbar(x, data, yerr=hinv[i,:], fmt='.')
    #index = np.argsort(x_total)
    plt.plot(x[index1],output1)
    plt.plot(x[index3][x[index3]>0],output3[x[index3]>0], color='g')
    plt.plot(x[index3][x[index3]<0],output3[x[index3]<0], color='g') 
    #plt.plot(x_total[index],y_total[index])
    #plt.text(0,0.5,r'fit: z^1=%2.3f, z^3=%2.3f, z^5=%2.3f, z^7=%2.3f, z^9=%2.3f' % tuple(popt))
    plt.xlabel('Relative Radius')
    plt.ylabel('Coefficients')
    plt.title('time:'+str(i)+'-th Legendre coeff')
    plt.legend(['raw','fit_inner', 'fit_outer'])
    pdf.savefig()
    #plt.show()
    plt.close()

print('finish plot')
pdf.close()
