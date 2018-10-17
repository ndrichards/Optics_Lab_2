from numpy import *
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd
import os

def dataset_func():
    dataset = []
    for f in os.listdir('/Users/lucas/Documents/Optics/Lab_Notes/Optics_Lab_2-master/'):
        data = pd.read_csv('/Users/lucas/Documents/Optics/Lab_Notes/Optics_Lab_2-master/%s' % (f,))
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

dataset = []
for f in os.listdir('/Users/lucas/Documents/Optics/Lab_Notes/Optics_Lab_2-master/'):
    data = genfromtxt('/Users/lucas/Documents/Optics/Lab_Notes/Optics_Lab_2-master/%s' % (f,))
    print('%s',(f,))
    dataset.append(data[:,0])
    dataset.append(data[:,1])
dataset = pd.DataFrame(dataset)
dataset = dataset.T
#%s ('argon (2).txt',) 0,1
#%s ('flourescent (2).txt',) 2,3
#%s ('incandescent (2).txt',) 4,5
#%s ('mercury (3).txt',) 6,7
#%s ('outdoors.txt',) 8,9
dataset['out_lamda'] = dataset[9]
dataset['out_c'] = dataset[8]
for i in range(9):
    del dataset[i]
dataset.columns
dataset




plt.plot(dataset['Hg_c'][400:2000], dataset['Hg_lamda'][400:2000])


from scipy.signal import argrelextrema
dataset.columns
n=150
m=10 # number of points to be checked before and after
# Find local peaks
#df['min'] = df.iloc[argrelextrema(df.data.values, np.less_equal, order=n)[0]]['data']
dataset['max_Hg'] = dataset.iloc[argrelextrema(dataset['Hg_lamda'].values, greater_equal, order=n)[0]]['Hg_lamda']
dataset['max_Hg'] = dataset['max_Hg'].apply(lambda x: nan if x < 10000 else x)
dataset['max_Ar'] = dataset.iloc[argrelextrema(dataset['Ar_lamda'].values, greater_equal, order=m)[0]]['Ar_lamda']
dataset['max_Ar'] = dataset['max_Ar'].apply(lambda x: nan if x < 10000 else x)
dataset['max_flo'] = dataset.iloc[argrelextrema(dataset['flo_lamda'].values, greater_equal, order=n)[0]]['flo_lamda']
dataset['max_flo'] = dataset['max_flo'].apply(lambda x: nan if x < 10000 else x)
dataset['max_inc'] = dataset.iloc[argrelextrema(dataset['inc_lamda'].values, greater_equal, order=n)[0]]['inc_lamda']
dataset['max_inc'] = dataset['max_inc'].apply(lambda x: nan if x < 10000 else x)
dataset['max_out'] = dataset.iloc[argrelextrema(dataset['out_lamda'].values, greater_equal, order=n)[0]]['out_lamda']
dataset['max_out'] = dataset['max_out'].apply(lambda x: nan if x < 10000 else x)

dataset
plt.scatter(dataset['out_c'][:], dataset['max_out'], c='g')
plt.plot(dataset['out_c'][:], dataset['out_lamda'][:],c='b',label = 'out')
plt.scatter(dataset['inc_c'][:], dataset['max_inc'], c='g')
plt.plot(dataset['inc_c'][:], dataset['inc_lamda'][:],c='b',label = 'inc')


# Plot results

plt.scatter(dataset['Hg_c'][:], dataset['max_Hg'], c='g')
plt.plot(dataset['Hg_c'][:], dataset['Hg_lamda'][:],c='b',label = 'Hg')
plt.scatter(dataset['Ar_c'][:], dataset['max_Ar'], c='r')
plt.plot(dataset['Ar_c'][:], dataset['Ar_lamda'][:],c = 'k',label = 'Ag')
plt.scatter(dataset['flo_c'][:], dataset['max_flo'], c='c')
plt.plot(dataset['flo_c'][:], dataset['flo_lamda'][:],c = 'y',label = 'Flo')
plt.legend()
plt.xlabel("Lambda (nm)")
plt.ylabel("Counts")
plt.savefig("/Users/lucas/Documents/Optics_plot_0.png")
plt.show()

dataset
dataset
index = 2
for i in range(0,9,2):
    plt.plot(dataset[i][400:2000], dataset[i+1][400:2000])


p0 = (1.,1.,1.)
args, err, chi = = fit(T_measured,Resistance,p0)



#Slice the data and get the local maximums for eah signal
ra2261 = dataset['Ra_226 pure signal'][8:16]


#get the max for each peak
max_ra226_1 = np.max(dataset[index][1400:2000])
#max_ra226_2 = np.max(ra2262)



#Find which channel those maximums occur at
where(dataset[1] == max(dataset[index][1400:2000]))
cs2 = int(np.where(dataset['Cs_137 pure signal'] == max_cs137_2)[0])

#ra4,ra3,ra2















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
