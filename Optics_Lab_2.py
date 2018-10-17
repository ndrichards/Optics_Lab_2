from numpy import *
from scipy import optimize
import matplotlib.pyplot as plt
import os

def dataset_func():
    dataset = []
    for f in os.listdir('/Users/lucas/Documents/Optics/Lab_Notes/Optics_Lab_2-master/'):
        data = genfromtxt('/Users/lucas/Documents/Optics/Lab_Notes/Optics_Lab_2-master/%s' % (f,))
        data = data.astype(str)
        data = insert(data,0,f[:-4],axis=0)
        dataset.append(data)
    dataset = pd.DataFrame(dataset)
    dataset = dataset.T
    for index in dataset.columns:
        dataset[dataset[index][0]] = dataset[index]
        del dataset[index]
    dataset = dataset.drop(0)
    dataset = dataset.astype('float')
    return dataset
def guassian(x,c1,c2,c3,c4):
    return c1 + c2*exp(-c3*(x-c4)**2)
def chi_sq(data_1, data_true):
    x=[]
    bin = len(data_1)
    for i in range(len(data_1)):
        z = (data_1[i] - data_true[i])**2/data_true[i]
        x.append(z)
    s = sum(x)/(bin-1)
    if s < 1.:
        print('Great fit: Chi_sq = %.2f' % (s,))
    elif isclose(s,1.,1.):
        print('Okay fit: Chi_sq = %.2f' % (s,))
    else:
        print('Bad fit: Chi_sq = %.2f' % (s,))
    #get the calibration data: T in K, R in Omhs
def fit(T_measured,Resistance,p0):
    #curvefit
    popt, pcovt = optimize.curve_fit(L,Resistance,T_measured,p0)
    err = sqrt(diag(pcovt))
    chi = chi_sq(L(Resistance,*popt),T_measured)
    return popt, err, chi




dataset.head()
plt.plot(dataset['Temperature'], dataset['Resistance'])

p0 = (1.,1.,1.)
args, err, chi = = fit(T_measured,Resistance,p0)

#Look for Tc
R = L(dataset['Temperature'], *args)

def diff(a):
    for index in range(len(a) - 1):
        prev_max = 0
        i = 0
        new_max = a[index-1] - a[index]
        if new_max > prev_max:
            prev_max = new_max
            i = index
    return new_max, i

r_max, index = diff(R)
Tc = dataset['Temperature'][index]
print(Tc)
