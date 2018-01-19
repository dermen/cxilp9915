
runsw = [11, 12, 13, 14, 15, 16, 
    20, 21, 22, 23, 24, 25, 
    28, 29, 30, 31, 32, 33, 34]

data =  { r: get_data2( '', str(r)) for r in runs2 }
import pylab as plt
import time

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np

rmin,rmax = 70,1100
nmin, nmax = 40, 120
#for r in runsw:
r = runsw[4]
#for s in ['dark', 'pumped']
peaks = data[r]['dark'][:,rmin:rmax]
n = peaks.mean(1)
peaks = peaks[ n > nmin ]
n = peaks.mean(1)
peaks = peaks[ n <nmax]
peaks /= peaks.mean(1)[:,None]

n_components = 20
pca = PCA(n_components=n_components)
new_peaks = pca.fit_transform( peaks )  
n_clusters = 3
new_kmeans = KMeans( n_clusters=n_clusters) 
new_kmeans.fit( new_peaks)


peaks0 = peaks[ new_kmeans.labels_==0]
peaks1 = peaks[ new_kmeans.labels_==1]
peaks2 = peaks[ new_kmeans.labels_==2]

#keep cluster 0 for now
D = peaks0

# pumped
peaks = data[r]['pumped'][:,rmin:rmax]
n = peaks.mean(1)
peaks = peaks[ n > nmin ]
n = peaks.mean(1)
peaks = peaks[ n <nmax]
peaks /= peaks.mean(1)[:,None]

n_components = 20
pca = PCA(n_components=n_components)
new_peaks = pca.fit_transform( peaks )  
n_clusters = 3
new_kmeans = KMeans( n_clusters=n_clusters) 
new_kmeans.fit( new_peaks)
peaks0 = peaks[ new_kmeans.labels_==0]
peaks1 = peaks[ new_kmeans.labels_==1]
peaks2 = peaks[ new_kmeans.labels_==2]

#keep cluster 0 for now
P = peaks0

result = []
shots = random.permutation( len( D ) )[:200]
for s in shots:
    o = argsort( ((P - D[s])**2).sum(1))
    result.append( (P[o]-D[s])[20:40].mean(0) )

import pandas as pd

def df_data( data ):
    keys = data.keys()
    new_data = {}
    for k in keys:
        df = pd.DataFrame( 
            { 'shot': data[k]['pumped'], 
              'dark': data[k]['pumped_timestamp'] } )
        new_data[k] = df
    return new_data


def make_elbow_plot( data, max_nclust=12,):
    """data is Nexample x Nfeatures """
    elbow_plot = []
    max_nclust = 12
    for i in xrange( 1, max_nclust+1):
        print("TRygin kmeans with %d clusters (up to %d)"%(i, max_nclust))
        kmeans = KMeans( n_clusters=i)
        kmeans.fit( data)
        elbow_plot.append( kmeans.score(data))
    plt.xlabel('# of clusters')
    plt.ylabel('how good is the clustering')
    plt.plot( np.arange( 1, max_nclust+1), elbow_plot, '.')

    plt.show()

# quick ana

def norm_peaks_inds( peaks, nmin=None,  inds=None):
    if nmin is not None:
        n = peaks.mean(1)
        peaks = peaks[ n > nmin]
        inds = inds[ n > nmin]

    n = peaks.mean(1)
    good = ~is_outlier(n, 2.)
    peaks = peaks[good]
    inds = inds[ good]
    peaks /= peaks.mean(1)[:,None]
    return peaks, inds


def norm_peaks( peaks, nmin=None, times=None, inds=None):
    if times is None:
        if nmin is not None:
            n = peaks.mean(1)
            peaks = peaks[ n > nmin]
        
        n = peaks.mean(1)
        good = ~is_outlier(n, 2.)
        peaks = peaks[good]
        peaks /= peaks.mean(1)[:,None]
        return peaks
    else:
        assert( inds is not None)
        if nmin is not None:
            n = peaks.mean(1)
            peaks = peaks[ n > nmin]
            times = times[ n > nmin]
            inds = inds[ n > nmin]

        n = peaks.mean(1)
        good = ~is_outlier(n, 2.)
        peaks = peaks[good]
        times = times[good]
        inds = inds[ good]
        peaks /= peaks.mean(1)[:,None]
        return peaks, times, inds

def remove_std(peaks, times = None, inds =None): 
    ol = [ where( is_outlier(p))[0] for p in peaks.T]
    bad = unique( concatenate( ol))
    
    if times is None:
        return np.delete( peaks, bad, axis=0)
    else:
        assert( inds is not None)
        return np.delete( peaks, bad, axis=0), np.delete( times, bad, axis=0), np.delete( inds, bad, axis=0)


def remove_std_inds(peaks, inds =None): 
    ol = [ where( is_outlier(p))[0] for p in peaks.T]
    bad = unique( concatenate( ol))
    return np.delete( peaks, bad, axis=0), np.delete( inds, bad, axis=0)


nmin=1.2
rmin=70
NORM_=False
RM_STD_=True
results = {}
all_pump = {}
all_dark = {}
for r in runs:
    D = data[r]['dark'][:,rmin:rmax]
    P = data[r]['pumped'][:,rmin:rmax]
    if NORM_:
        D = norm_peaks(D, nmin=nmin)
        P = norm_peaks(P, nmin=nmin)
    if RM_STD_:
        D = remove_std(D)
        P = remove_std(P)
    pm = P.mean(0)
    dm = D.mean(0)
    pm /= pm.mean()
    dm /= dm.mean()
    all_pump[r] = P
    all_dark[r] = D
    results[r] = pm - dm 


###########
# times
nmin=0 #1.2
rmin=70
NORM_=False
RM_STD_=True

BOUNDS_ = True
minADU=5
maxADU=25

results = {}
all_pump = {}
all_pump_raw = {}
all_dark_raw = {}
all_dark = {}
all_Pt = {}
all_Dt ={}
for r in runs:
    D = data[r]['dark'][:,rmin:rmax]
    P = data[r]['pumped'][:,rmin:rmax]
    DD = D.copy()
    PP = P.copy()


    Dt = data[r]['dark_timestamp'] 
    Pt = data[r]['pumped_timestamp'] 

    D_i = np.arange(D.shape[0])
    P_i = np.arange(P.shape[0])
    if BOUNDS_:
        n = D.mean(1)
        inds = np.where( np.logical_and(n > minADU, n < maxADU) )[0]
        D, Dt, D_i = D[inds], Dt[inds], D_i[inds]

        n = P.mean(1)
        inds = np.where( np.logical_and(n > minADU, n < maxADU) )[0]
        P, Pt, P_i = P[inds], Pt[inds], P_i[inds]

    if NORM_:
        D, Dt, D_i = norm_peaks(D, nmin=nmin, times=Dt, inds=D_i)
        P, Pt, P_i = norm_peaks(P, nmin=nmin, times=Pt, inds=P_i)
    
    if RM_STD_:
        D, Dt, D_i = remove_std(D, times=Dt, inds=D_i)
        P, Pt, P_i = remove_std(P, times=Pt, inds=P_i)
    
    all_dark_raw[r] = DD[ D_i]
    all_pump_raw[r] = PP[ P_i]
    
    pm = P.mean(0)
    dm = D.mean(0)
    pm /= pm.mean()
    dm /= dm.mean()
    all_pump[r] = P
    all_Pt[r] = Pt
    all_dark[r] = D
    all_Dt[r] = Dt
    results[r] = pm - dm 

#end times
#################

def cluster_pros(  peaks_, n_clusters=3, times=None, inds_only=False):
    peaks = peaks_.copy()
    peaks /= peaks.mean(1)[:,None]
    n_components = 20
    pca = PCA(n_components=n_components)
    new_peaks = pca.fit_transform( peaks )  
    new_kmeans = KMeans( n_clusters=n_clusters) 
    new_kmeans.fit( new_peaks)
    
    if not inds_only:
        results = [ peaks[new_kmeans.labels_ ==l] 
            for l in xrange( n_clusters)]
    else:
        results = [ np.where( new_kmeans.labels_ ==l)[0]
            for l in xrange( n_clusters)]

    if times is None:
        return results
    else:
        times_results = [ times[new_kmeans.labels_ ==l] 
            for l in xrange( n_clusters)]
        return results, times_results


#########
# plot kmeans clustering run187
#########
clust_dark = clust136_d
clust_pump = clust136
c = ['b','r', 'g']
c_f = ['lightblue', 'pink', 'palegreen']
clf()
fs = 14
suptitle("K-means clustering on pumped and darks; CXILP9915;run187", fontsize=fs)
subplot(121)
ax = gca()
ax.set_facecolor('grey')
for i_ in xrange( 3):
    m = clust_dark[i_].mean(0)
    s = clust_dark[i_].std(0)
    Nshots = len(clust_dark[i_])
    fill_between( qs1, m+s, m-s, 
        facecolor=c_f[i_] , alpha=0.8) 
    plot( qs1, m, c=c[i_], label='cluster %d; %d shots'%(i_, Nshots))
xlim(.2, 2.3)
xlabel("inverse angstrom", fontsize=fs)
ylabel("Mean radial intensity (normalized)", fontsize=fs)
title("Dark shots", fontsize=fs)
legend()
ax.tick_params(labelsize=fs)


subplot(122)
ax = gca()
ax.set_facecolor('grey')
for i_ in xrange(3):
    m = clust_pump[i_].mean(0)
    s = clust_pump[i_].std(0)
    Nshots = len(clust_pump[i_])
    fill_between( qs1, m+s, m-s, 
        facecolor=c_f[i_] , alpha=0.8) 
    plot( qs1, m, c=c[i_], label='cluster %d; %d shots'%(i_,Nshots ))
xlim(.2, 2.3)
xlabel("inverse angstrom", fontsize=fs)
ylabel("Mean radial intensity (normalized)", fontsize=fs)
title("Pumped shots", fontsize=fs)
ax.tick_params(labelsize=fs)
legend()

######
# plot diffs
######
figure(2)
ax = gca()
for i in xrange( 3 ):
    m = clust_pump[i].mean(0)
    m_d = clust_dark[i].mean(0)
    diff = m/m.mean() - m_d/m_d.mean()
    n = abs( smooth( diff, window_size=100) ).max()
    diff /= n
    plot( qs1, diff, label="cluster %d"%i, 
        color=c[i])
ax.set_facecolor('lightgrey')
xlabel("Q (inverse angstrom)", fontsize=fs)
ylabel("Difference intensity", fontsize=fs)



# within lines

def line1(x):
    return ones_like(x)* 815.

def line2(x):
    X1, Y1 = 1145., 815.
    X2, Y2 = 1103., 733.
    return ( (Y1-Y2) / (X1-X2) )*(x-X1) + Y1

def line3(x):
    x1,y1 = 1109., 815.
    x2,y2 = 1103., 733.
    return ( (y1-y2) / (x1-x2) )*(x-x1) + y1
    #x1,y1 = 1109, 815
    #x2,y2 = 1103, 733
    #return ( (y2-y1) / (x2-x1) )*(x-x1) + y1

    

xvals = linspace( 1100, 1170,100)
plot( xvals, line1(xvals), 'w')
plot( xvals, line2(xvals), 'w')
plot( xvals, line3(xvals), 'w')

def line4(x):
    return ones_like(x)* 846.

def line5(x):
    X1, Y1 = 1135., 975.
    X2, Y2 = 1154., 846.
    return ( (Y1-Y2) / (X1-X2) )*(x-X1) + Y1

plot( ones(100)*1135, linspace( 846, 975,100), 'w')
plot( xvals, line4(xvals), 'w')
plot( xvals, line5(xvals), 'w')


nrows=3
ncols=5
inds = random.permutation(len(imgs))
fig, axs = subplots( nrows=nrows,ncols=ncols)
k = 0
for i in xrange(nrows):
    for j in xrange( ncols):
        ax = axs[i][j]
        ax.imshow( imgs[inds[k]],cmap='gnuplot', vmin=0, vmax=1000, aspect='auto')
        ax.set_xlim(750,950)
        ax.set_ylim(700,1100)
        k += 1




# script for LCLS
# script for LCLS
# script for LCLS
# script for LCLS
# script for LCLS
# script for LCLS
# script for LCLS
# script for LCLS

num_events={1: 1267,
 2: 1469,
 3: 794,
 4: 986,
 5: 587,
 6: 2397,
 7: 9961,
 8: 9438,
 9: 20286,
 10: 924,
 11: 23200,
 12: 22098,
 13: 22410,
 14: 21331,
 15: 21025,
 16: 22509,
 17: 21806,
 18: 12448,
 19: 25251,
 20: 21509,
 21: 44717,
 22: 26015,
 23: 21197,
 24: 21505,
 25: 21356,
 26: 1431,
 27: 24079,
 28: 39956,
 29: 34474,
 30: 24098,
 31: 13056,
 32: 43795,
 33: 40809,
 34: 40445,
 35: 1157,
 36: 943,
 37: 17011,
 38: 10279,
 39: 17187,
 40: 2509,
 41: 2218,
 42: 58757,
 43: 32319,
 44: 35496,
 45: 64543,
 46: 86249,
 47: 164707,
 48: 39452,
 49: 44309,
 50: 54643,
 51: 61181,
 52: 25207,
 53: 29317,
 54: 50188,
 55: 29545,
 56: 2477,
 57: 9264,
 58: 25770,
 59: 9557,
 60: 6459,
 61: 9416,
 62: 10031,
 63: 6041,
 64: 17729,
 65: 11024,
 66: 8198,
 67: 34932,
 68: 19847,
 69: 832,
 70: 3360,
 71: 21709,
 72: 21821,
 73: 6594,
 74: 35716,
 75: 53575,
 76: 53551,
 77: 36167,
 78: 27152,
 79: 21740,
 80: 21471,
 81: 26879,
 82: 35878,
 83: 50119,
 84: 14704,
 85: 14592,
 86: 14404,
 87: 14560,
 88: 14560,
 89: 16712,
 90: 14417,
 91: 15076,
 92: 14487,
 93: 21773,
 94: 28844,
 95: 21626,
 96: 24169,
 97: 26938,
 98: 35804,
 99: 53673,
 100: 35895,
 101: 23958,
 102: 6508,
 103: 36132,
 104: 35993,
 105: 35928,
 106: 36562,
 107: 35944,
 108: 28293,
 109: 71395,
 110: 71787,
 111: 71486,
 112: 71793,
 113: 71674,
 114: 71211,
 115: 72544,
 116: 80003,
 117: 73218,
 120: 36192,
 121: 71173,
 122: 73450,
 123: 15694,
 124: 72758,
 125: 72016,
 126: 79467,
 127: 71548,
 128: 74182,
 129: 73597,
 130: 13639,
 133: 71581,
 134: 73277,
 135: 9877,
 136: 44084,
 137: 42243,
 138: 37739,
 139: 37509,
 140: 46856,
 141: 38177,
 142: 19244,
 143: 42821,
 144: 36214,
 145: 37740,
 146: 36311,
 147: 35691,
 148: 35686,
 153: 36881,
 154: 9724,
 155: 17264,
 156: 2236,
 157: 2320,
 158: 2411,
 159: 1675,
 160: 2156,
 162: 3757,
 163: 91199,
 164: 44376,
 165: 44015,
 166: 46530,
 167: 36068,
 168: 36401,
 169: 35854,
 170: 157581,
 171: 36063,
 172: 35834,
 173: 35728,
 174: 35718,
 175: 39923,
 176: 36114,
 177: 36131,
 178: 36297,
 179: 30399,
 180: 37695,
 181: 36064,
 182: 44267,
 183: 35886,
 184: 35869,
 185: 14499,
 186: 39905,
 187: 155750,
 189: 35753,
 190: 35672,
 191: 35741,
 192: 35833,
 193: 37306,
 194: 35731,
 195: 35721,
 196: 35694,
 197: 37983,
 198: 61888,
 199: 36532,
 200: 37409,
 201: 35769,
 202: 35704,
 203: 36016,
 204: 35748,
 205: 35718,
 206: 35743,
 207: 35702,
 208: 35759,
 209: 39309,
 210: 35641,
 211: 35765,
 212: 39356,
 213: 4360,
 214: 4862,
 215: 2202,
 216: 35719,
 217: 35697,
 218: 35733,
 219: 21303,
 220: 35760,
 221: 36018,
 222: 35707,
 223: 35715,
 224: 37289,
 225: 2488,
 226: 35638,
 227: 35742,
 228: 12584,
 236: 29987,
 237: 10248,
 238: 702,
 239: 10829,
 240: 19454,
 241: 28464,
 242: 14559,
 243: 14150,
 244: 16670,
 245: 14948,
 253: 1633,
 254: 16772,
 255: 7172,
 256: 5865,
 257: 3638,
 258: 4377,
 259: 3379,
 260: 9580,
 261: 3040,
 262: 4518,
 263: 15845,
 264: 917,
 265: 3325,
 266: 1303,
 267: 44918}

def norm_peaks( peaks, nmin=None, times=None, inds=None):
    if times is None:
        if nmin is not None:
            n = peaks.mean(1)
            peaks = peaks[ n > nmin]
        
        n = peaks.mean(1)
        good = ~is_outlier(n, 2.)
        peaks = peaks[good]
        peaks /= peaks.mean(1)[:,None]
        return peaks
    else:
        assert( inds is not None)
        if nmin is not None:
            n = peaks.mean(1)
            peaks = peaks[ n > nmin]
            times = times[ n > nmin]
            inds = inds[ n > nmin]

        n = peaks.mean(1)
        good = ~is_outlier(n, 2.)
        peaks = peaks[good]
        times = times[good]
        inds = inds[ good]
        peaks /= peaks.mean(1)[:,None]
        return peaks, times, inds

def remove_std(peaks, times = None, inds =None): 
    ol = [ where( is_outlier(p))[0] for p in peaks.T]
    bad = unique( concatenate( ol))
    
    if times is None:
        return np.delete( peaks, bad, axis=0)
    else:
        assert( inds is not None)
        return np.delete( peaks, bad, axis=0), np.delete( times, bad, axis=0), 
            np.delete( inds, bad, axis=0)

from differencing import get_data2

runs = range( 254, 264)  
data = {r:get_data2("", str(r)) for r in runs}
r = runs[0]
###########
# times
nmin=0 #1.2
rmin=70
NORM_=True#False
RM_STD_=True
results = {}
all_pump = {}
all_pump_raw = {}
all_dark_raw = {}
all_dark = {}
all_Pt = {}
all_Dt ={}

D = data[r]['dark'][:,rmin:rmax]
P = data[r]['pumped'][:,rmin:rmax]
DD = D.copy()
PP = P.copy()

Dt = data[r]['dark_timestamp'] 
Pt = data[r]['pumped_timestamp'] 

D_i = np.arange(D.shape[0])
P_i = np.arange(P.shape[0])
if NORM_:
    D, Dt, D_i = norm_peaks(D, nmin=nmin, times=Dt, inds=D_i)
    P, Pt, P_i = norm_peaks(P, nmin=nmin, times=Pt, inds=P_i)

if RM_STD_:
    D, Dt, D_i = remove_std(D, times=Dt, inds=D_i)
    P, Pt, P_i = remove_std(P, times=Pt, inds=P_i)

all_dark_raw[r] = DD[ D_i]
all_pump_raw[r] = PP[ P_i]

pm = P.mean(0)
dm = D.mean(0)
pm /= pm.mean()
dm /= dm.mean()
all_pump[r] = P
all_Pt[r] = Pt
all_dark[r] = D
all_Dt[r] = Dt
results[r] = pm - dm 

#end times
#################
import psana
import h5py
Fsh = (1734, 1731)
Bsh = (1738, 1742)
orderD = argsort( all_dark_raw[r].mean(-1))[::-1]
timesD = all_Dt[r][ orderD]

orderP = argsort( all_pump_raw[r].mean-1))[::-1]
timesP = all_Pt[r][ orderP]

timesP = timesP[ : min( len(timesP) , max_N  )  ]
timesD = timesD[ : min( len(timesD) , max_N  )  ]

ds = psana.DataSource("exp=cxilp9915:run=%d"%r)
events = ds.events()
detB = psana.Detector( "DsdCsPad", ds.env() ) 
detF = psana.Detector( "DscCsPad", ds.env() ) 
N = num_events[r]

out = h5py.File("run%d_front_backs.h5py", "w")
darkF = out.create_dataset("dark_front", compression='lzf', shape=(0,Fsh[0], Fsh[1]), 
    dtype=np.uint16)
darkB = out.create_dataset("dark_back",  compression='lzf',  shape=(0,Bsh[0], Bsh[1]),
    dtype=np.uint16)
pumpF = out.create_dataset("pump_front",compression='lzf', shape=(0,Fsh[0], Fsh[1]),
    dtype=np.uint16)
pumpB = out.create_dataset("pump_back", compression='lzf', shape=(0,Bsh[0], Bsh[1]),
    dtype=np.uint16)

pump_count = 0
dark_count = 0
for i in xrange( N):
    ev = events.next()
    if ev is None:
        continue
    eid = ev.get( psana.EventId)
    t,nt = eid.time()
    ts = t + 1e-9*nt
    if ts in timesD:
        imgF = detF.image(ev)
        imgB = detB.image(ev)
        if imgF is None:
            continue
        if imgB is None:
            continue
        imgF[ imgF < 0] = 0
        imgB[ imgB < 0] = 0
        imgF[ imgF > 60000] = 0
        imgB[ imgB > 60000] = 0
        dark_count += 1
        darkF.resize( (dark_count, Fsh[0], Fsh[1]),)
        darkB.resize( (dark_count, Bsh[0], Bsh[1]),)
        darkF[dark_count] = imgF.astype(np.uint16)
        darkB[dark_count] = imgB.astype(np.uint16)
    elif ts in timesP:
        imgF = detF.image( ev)
        imgB = detB.image(ev)
        if imgF is None:
            continue
        if imgB is None:
            continue
        imgF[ imgF < 0] = 0
        imgB[ imgB < 0] = 0
        imgF[ imgF > 60000] = 0
        imgB[ imgB > 60000] = 0
        pump_count += 1
        pumpF.resize( (pump_count, Fsh[0], Fsh[1]),)
        pumpB.resize( (pump_count, Bsh[0], Bsh[1]),)
        pumpF[pump_count] = imgF.astype(np.uint16)
        pumpB[pump_count] = imgB.astype(np.uint16)

out.close()





#############
#### rad pro script

from loki.RingData import RadialProfile
import numpy as np

mask = {"back":np.load("mask1_mask2.npy"), 
    "front":np.load("front_mask.npy") }

cent = {"back":np.load("cent_back.npy"),
    "front":np.load("cent_front.npy")}

img_sh = {"front":(1734, 1731), 
    "back": (1738, 1742) }

radpro_makers = 
    {"back": 
        RadialProfile( cent["back"], 
                        img_sh["back"], 
                        mask["back"], 
                        minlength=1300),
    "front": 
        RadialProfile(cent["front"],  
                    img_sh["front"], 
                    mask["front"], 
                    minlength=1300)}


import h5py
out = h5py.File("master_radial_dump.h5py", "w")

r = 242
fname = "run%d_front_backs.h5py"%r
h5 = h5py.File(fname, 'r')

dark_data = {"back":h5['dark_back'],
            "front": h5['dark_front'] }

Ndark = dark_data["back"].shape[0]

pump_data = {"back":h5['pump_back'],
            "front": h5['pump_front'] }
Npump = dark_data["back"].shape[0]

dark_radpros = {"back":[], "front":[]}
for i_dark in xrange( Ndark):
    radpro = radpro_maker["back"].calculate( dark_data["back"][i_dark]  )
    dark_radpros["back"].append(radpro)
    radpro = radpro_maker["front"].calculate( dark_data["front"][i_dark]  )
    dark_radpros["front"].append(radpro)

pump_radpros = {"back":[], "front":[]}
for i_pump in xrange( Npump):
    radpro = radpro_maker["back"].calculate( pump_data["back"][i_pump]  )
    pump_radpros["back"].append(radpro)
    radpro = radpro_maker["front"].calculate( pump_data["front"][i_pump]  )
    pump_radpros["front"].append(radpro)


for status in ["dark", "pump"]:
    for det_pos in ["back","front"]:
        out.create_dataset( "run%d/%s/%s"%(r, status, det_pos), 
            data=pump_radpros[det_pos].astype(np.float32),
            compression="lzf",  
            shuffle=True )

out.close()







###########
# make the plot of radials in time
from itertools import cycle
colors = cycle(['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'])

runs = range( 191, 264)

data = {'dark':{'mn':[],'times':[]}, 
    'pumped':{'mn':[], 'times':[]}}
darks = {}
pumps = {}
for r in runs:
    try:
        out = get_data2("",str(r))
        for stat in ['dark', 'pumped']:
            pros = out[stat]
            t = out['%s_timestamp'%stat]
            order = np.argsort(t)
            mn = pros[order,150:1150].mean(1)
            data[stat]['times'].append(t[order])
            data[stat]['mn'].append(mn)
    except:
        pass



# reduce by mean ADU
# reduce by mean ADU
# reduce by mean ADU
# reduce by mean ADU
# reduce by mean ADU
# reduce by mean ADU
# reduce by mean ADU
# reduce by mean ADU
# reduce by mean ADU
# reduce by mean ADU
# reduce by mean ADU
# reduce by mean ADU
# reduce by mean ADU
# reduce by mean ADU
# reduce by mean ADU
# reduce by mean ADU
# reduce by mean ADU
# reduce by mean ADU
# reduce by mean ADU
# reduce by mean ADU




import h5py, glob

fnames = glob.glob("radials_alpha/*h5py") + glob.glob("radials_beta/*h5py") 


runs = [ f.split("/")[1].split("_")[0].split("run")[1] 
    for f in fnames ] 

assert( len(runs) == len(fnames))

data = { r: h5py.File(fnames[i], 'r')['run%s'%r ]
    for i,r in enumerate(runs) }

pump_front = array( [ np.round( ma.masked_equal(data[r]['pump']['front'].\
    value[:,150:1150],0).mean(1).data, 1) for r in runs ] )

dark_front = array( [ np.round( ma.masked_equal(data[r]['dark']['front'].\
    value[:,150:1150],0).mean(1).data, 1) for r in runs ] )

runs_old = array( runs)
runs = runs[:17]
runs = sorted(runs)
pump_front = array( [ np.round( ma.masked_equal(data[r]['pump']['front'].\
    value[:,150:1150],0).mean(1).data, 1) for r in runs ] )
dark_front = array( [ np.round( ma.masked_equal(data[r]['dark']['front'].\
    value[:,150:1150],0).mean(1).data, 1) for r in runs ] )
vals = reduce(lambda x,y: intersect1d(x,y), 
    list(pump_front) + list(dark_front))
pump_pros = {v: [data[runs[i]]['pump/front'].value[where(d == v)[0]] 
    for i,d in enumerate(pump_front)] for v in vals }
dark_pros = {v: [data[runs[i]]['dark/front'].value[where(d == v)[0]] 
    for i,d in enumerate(dark_front)] for v in vals }


pump_pros_back = {v: [data[runs[i]]['pump/back'].value[where(d == v)[0]] 
    for i,d in enumerate(pump_front)] for v in vals }
dark_pros_back = {v: [data[runs[i]]['dark/back'].value[where(d == v)[0]] 
    for i,d in enumerate(dark_front)] for v in vals }


#out = h5py.File("alpha_beta_merged_frontback.h5py", 'w')
#for v in vals:
#    key = "ADU_%.1f"%v
#    for i_r,r in enumerate(runs):
#        out.create_dataset(key+"/pump/front/run%s"%r, data=pump_pros[v][i_r])
#        out.create_dataset(key+"/dark/front/run%s"%r, data=dark_pros[v][i_r])
#        out.create_dataset(key+"/pump/back/run%s"%r, data=pump_pros_back[v][i_r])
#        out.create_dataset(key+"/dark/back/run%s"%r, data=dark_pros_back[v][i_r])
#out.close()

delays = { runs[i]:d for i,d in enumerate( np.array([0. , 0.5, 0.5, 1. , 
    1.5, 2. , 2.5, 3. , 
    3.5, 4. , 4.5, 5. , 5.5,
    6. , 6.5, 7. , 9.5]) ) }

i1 = 60
i2 = 1100
i1_back = 270
i2_back = 305
rs = arange( i1,i2)
rs_back = arange( i1_back, i2_back)
detdist = .279
detdist_back = 2.7
wavelen = 1.305
thetas =  np.arctan( 0.00011 * rs/ detdist   ) * .5
thetas_back =  np.arctan( 0.00011 * rs_back/ detdist_back   ) * .5
qs = 4 * np.pi * np.sin(thetas)/ wavelen
qs_back = 4 * np.pi * np.sin(thetas_back)/ wavelen
sa = np.cos(thetas)**3
sa_back = np.cos(thetas_back)**3
phot_adu = 33.
phot_adu_back = 27.

all_mns = {"front":{}, "back":{}}
all_std = {"front":{}, "back":{}}

for i_r, r in enumerate( runs):
    diffs = []
    diffs_back = []
    for v in vals:
        p = pump_pros[v][i_r]
        d = dark_pros[v][i_r]
        n = min( p.shape[0], d.shape[0] )
        pd = p[:n].mean(0) - d[:n].mean(0)
        diffs.append( pd)
        #############  back detector #######
        p = pump_pros_back[v][i_r]
        d = dark_pros_back[v][i_r]
        n = min( p.shape[0], d.shape[0] )
        pd = p[:n].mean(0) - d[:n].mean(0)
        diffs_back.append( pd)
        
        print runs[i_r],n
    all_mns["front"][r] = mean( diffs,0)[i1:i2]/sa/phot_adu
    all_std["front"][r] = std( diffs,0)[i1:i2]/sa/phot_adu
    
    all_mns["back"][r] = mean( diffs_back,0)[i1_back:i2_back]/sa_back/phot_adu_back
    all_std["back"][r] = std( diffs_back,0)[i1_back:i2_back]/sa_back/phot_adu_back



img_back = array([all_mns["back"][r] for r in good_runs])
img = array([all_mns["front"][r] for r in good_runs])


# back img binned
dq = 0.001
qbins_back = linspace( qs_back[0], qs_back[-1], int((qs_back[-1]-qs_back[0]) / dq))
img_bin_back = array( [histogram( qs_back, bins=qbins_back, weights=row, )[0]  
    for row in img_back] )
qs_bin_back = .5*(qbins_back[1:] + qbins_back[:-1] )
imshow( img_back , aspect='auto', cmap='gnuplot') 
ax = gca()
yticks = arange( len( good_runs) )
ylabs = map( lambda x: "%.1f"%x, [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 
    5.0, 5.5, 6.0, 6.5, 7.0])
ax.set_yticklabels(ylabs)
ax.set_yticks(yticks)
xticks = [2]
xlabs = map( lambda x: "%.3f"%qs_bin_back[x], xticks)
ax.set_xticklabels(xlabs)
ax.set_xticks(xticks)
cbar = colorbar()
cbar.ax.set_ylabel("photons", rotation=-90, labelpad=20)
ax.set_xlabel(r"Q/$\AA^{-1}$", fontsize=13, labelpad=10)
ax.set_ylabel("time delay (ps)", fontsize=13, labelpad=10)
ax.tick_params(labelsize=12)

run_s = ", ".join( good_runs)
suptitle("Back detector difference profiles\nruns: %s"%run_s, 
    fontsize=13)




# back image
imshow( img_back , aspect='auto', cmap='gnuplot') 
ax = gca()
yticks = arange( len( good_runs) )
ylabs = map( lambda x: "%.1f"%x, [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 
    5.0, 5.5, 6.0, 6.5, 7.0])
ax.set_yticklabels(ylabs)
ax.set_yticks(yticks)
xticks = [0,5,10,15,20,25,30]
xlabs = map( lambda x: "%.3f"%qs_back[x], xticks)
ax.set_xticklabels(xlabs)
ax.set_xticks(xticks)
cbar = colorbar()
cbar.ax.set_ylabel("photons", rotation=-90, labelpad=20)
ax.set_xlabel(r"Q/$\AA^{-1}$", fontsize=13, labelpad=10)
ax.set_ylabel("time delay (ps)", fontsize=13, labelpad=10)
ax.tick_params(labelsize=12)

run_s = ", ".join( good_runs)
suptitle("Back detector difference profiles\nruns: %s"%run_s, 
    fontsize=13)


# front img binned
qbins = linspace( qs[0], qs[-1], int((qs[-1]-qs[0]) / dq))
img_bin = array( [histogram( qs, bins=qbins, weights=row, )[0]  
    for row in img] )
qs_bin = .5*(qbins[1:] + qbins[:-1] )
imshow( img_bin , aspect='auto', cmap='gnuplot') 
ax = gca()
yticks = arange( len( good_runs) )
ylabs = map( lambda x: "%.1f"%x, [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 
    5.0, 5.5, 6.0, 6.5, 7.0])
ax.set_yticklabels(ylabs)
ax.set_yticks(yticks)
xticks = [0,20,40,60,80]
xlabs = map( lambda x: "%.3f"%qs_bin[x], xticks)
ax.set_xticklabels(xlabs)
ax.set_xticks(xticks)
cbar = colorbar()
cbar.ax.set_ylabel("photons", rotation=-90, labelpad=20)
ax.set_xlabel(r"Q/$\AA^{-1}$", fontsize=13, labelpad=10)
ax.set_ylabel("time delay (ps)", fontsize=13, labelpad=10)
ax.tick_params(labelsize=12)
run_s = ", ".join( good_runs)
suptitle("Front detector difference profiles\nruns: %s"%run_s, 
    fontsize=13)

# front img
imshow( img , aspect='auto', cmap='gnuplot') 
ax = gca()
yticks = arange( len( good_runs) )
ylabs = map( lambda x: "%.1f"%x, [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 
    5.0, 5.5, 6.0, 6.5, 7.0])
ax.set_yticklabels(ylabs)
ax.set_yticks(yticks)
xticks = [0,200,400,600,800]
xlabs = map( lambda x: "%.3f"%qs[x], xticks)
ax.set_xticklabels(xlabs)
ax.set_xticks(xticks)
cbar = colorbar()
cbar.ax.set_ylabel("photons", rotation=-90, labelpad=20)
ax.set_xlabel(r"Q/$\AA^{-1}$", fontsize=13, labelpad=10)
ax.set_ylabel("time delay (ps)", fontsize=13, labelpad=10)
ax.tick_params(labelsize=12)
run_s = ", ".join( good_runs)
suptitle("Front detector difference profiles\nruns: %s"%run_s, 
    fontsize=13)









dq = 0.0025
qbins_back = linspace( qs_back[0], qs_back[-1], int((qs_back[-1]-qs_back[0]) / dq))
img_bin_back = array( [histogram( qs_back, bins=qbins_back, weights=row, )[0]  
    for row in img_back] )
norm_bin_back =array( [histogram( qs_back, bins=qbins_back, )[0]   
    for row in img_back] )
qs_bin_back = .5*(qbins_back[1:] + qbins_back[:-1] )

qbins = linspace( qs[0], qs[-1], int((qs[-1]-qs[0]) / dq))
img_bin = array( [histogram( qs, bins=qbins, weights=row, )[0]  
    for row in img] )
norm_bin =array( [histogram( qs, bins=qbins, )[0]   
    for row in img] )
qs_bin = .5*(qbins[1:] + qbins[:-1] )







###########################
#######
import psana
import argparse
import h5py
import numpy as np

from loki.RingData import RadialProfile


parser = argparse.ArgumentParser(description='Data grab')
parser.add_argument('-r', '--run', type=int, required=True, help='run number to process')
parser.add_argument('-m', '--maxcount', type=int, default=0, help='max shots to process')
parser.add_argument('-s', '--start', type=int, default=0, help='first event to process')
parser.add_argument('-f', '--fname', type=str, default=None, help='output basename')
args = parser.parse_args()

out = h5py.File( out_fname, "w")

# calib files
mask_fname_back = "mask1_mask2.npy"
mask_fname = "front_mask.npy"
cent_fname= "cent_front.npy"
cent_fname_back= "cent_back.npy"


# radial profile makers
mask = {"back":np.load(mask_fname_back), 
    "front":np.load(mask_fname) }

cent = {"back":np.load(cent_fname_back),
    "front":np.load(cent_fname)}

img_sh = {"front":(1734, 1731), 
    "back": (1738, 1742) }

radpro_makers = 
    {"back": 
        RadialProfile( cent["back"], 
                        img_sh["back"], 
                        mask["back"], 
                        minlength=1300),
    "front": 
        RadialProfile(cent["front"],  
                    img_sh["front"], 
                    mask["front"], 
                    minlength=1300)}



# experiment run number
exp = "cxilp9915"

# evr codes
pump_on = 183
pump_off = 179
probe_off = 162

# 
ds = psana.DataSource("exp=%s:run=%s"%(exp, args.run))
env = ds.env()

# get the detector names for experiment/run
detnames = [ d for sl in psana.DetNames() for d in sl]

# evr codes for laser logic
code_dets = [ psana.Detector(d, env) for d in detnames if d.startswith("evr") ]

# get the cspads
cspad = {"front":None, "back":None}

assert( "DscCsPad" in detnames )
cspad["front"] = psana.Detector("DscCsPad", env)

assert( "DscCsPad" in detnames )
cspad["back"] = psana.Detector("DsdCsPad", env)

#trace
trace_det = psana.Detector('Timetool', env)
ebeam_det = psana.Detector('EBeam', env)
phase_cav_det = psana.Detector('PhaseCavity', env)

events = ds.events()
shot_counter = 0
pump = []
times_sec = []
times_nanosec = []
front_radials = []
back_radials = []
opal_lineouts = []
phot_energy = []
charge1 = []
charge2 = []
seen_evts = 0

for i_ev, ev in events()
    if i_ev < args.start:
        print ("Skipping event %d/%d"%(i_ev+1, args.start))
        continue
    
    seen_evts += 1
    if seen_evts == args.maxcount:
        print("Reached max counts!")
        break
    
    if ev is None:
        continue

    trace = trace_det.image( ev )
    ebeam = ebeam_det.get(ev)
    phase_cav = phase_cav_det.get(ev)

#   parse all event codes
    codes = []
    for det in code_dets:
        c = det.eventCodes(ev)
        if c is not None:
            codes += c
    if probe_off in codes:
        continue
    if pump_on in codes:
        pump.append( 1  )
    elif pump_off in codes:
        pump.append( 0  )
    else:
        continue
    
    img_front = cspad["front"].image( ev)
    if img is None:
        continue
    
    img_back = cspad["back"].image( ev)
    if img_back is None:
        continue
    
    radpro = rapro_makers["front"].calculate(img)
    radpro_back = rapro_makers["back"].calculate(img_back)

    eid = ev.get( psana.EventId)
    t,nt = eid.time()

    front_radials.append( radpro)
    back_radials.append( radpro_back)
    times_sec.append(t)
    times_nanosec.append(nt)
    opal_lineouts.append( trace[15:65].mean(0))
    phot_energy.append( ebeam.ebeamPhotonEnergy() )
    charge1.append( phase_cav.charge1() )
    charge2.append( phase_cav.charge2() )

    shot_counter += 1
    print("processed %d / %d images"%( shot_counter / (i_ev +1 - args.start) ))

out.create_dataset("radials/front", data=front_radials,dtype=np.float32, 
    compression='lzf', shuffle=True)
out.create_dataset("radials/back", data=back_radials,dtype=np.float32, 
    compression='lzf', shuffle=True)
out.create_dataset("opal_lineout", data=opal_lineout,dtype=np.float32, 
    compression='lzf', shuffle=True)
out.create_dataset("seconds", data=times_sec,dtype=np.int64, 
    compression='lzf', shuffle=True)
out.create_dataset("nanoseconds", data=times_nanosec,dtype=np.int64, 
    compression='lzf', shuffle=True)
out.create_dataset("charge1", data=charge1,dtype=np.float32, 
    compression='lzf', shuffle=True)
out.create_dataset("charge2", data=charge2,dtype=np.float32, 
    compression='lzf', shuffle=True)
out.create_dataset("photon_energy", data=phot_energy, dtype=np.float64, 
    compression='lzf', shuffle=True)





# XTAL HITS
# XTAL HITS
# XTAL HITS
# XTAL HITS
# XTAL HITS
hits_fname = "hits_196_3.h5py"
Ish = ( 1736, 1734)

import psana
import h5py
ds = psana.DataSource("exp=cxin6016:run=%d"%196)
det = psana.Detector( "DscCsPad", ds.env() )
events = ds.events()

cut = 2.8e6
with h5py.File(hits_fname, 'w') as f:
    dset = f.create_dataset('data', 
        shape=(0,Ish[0], Ish[1]) , 
        dtype=np.uint16, 
        compression='lzf', 
        maxshape=(None,Ish[0],Ish[1]))

    count = 0
    for i in xrange( 1000):
        print(i)
        ev = events.next()
        if ev is None:
            continue
        img = det.image(ev)
        if img is None:
            continue
        img[ img < 0 ] = 0
        img[ img > 60000] = 0
        img_mn = img[ img > 0].sum()
        if img_mn > cut:
            dset.resize( ( count+1, Ish[0], Ish[1] ) )
            dset[count] = img.astype(np.uint16)
            count += 1

###################
###################
###################
###################
###################
###################
###################


runs = [241, 245, 255, 256, 257, 258, 259 ]

P = {}
D = {}
Diff = {}
runs = [85, 181, 183, 184, 212, 241, 245, 258]
for r in runs:
    h5 = h5py.File("run%d_ALL.h5py"%r,"r")
    P[r] = h5["pumped/imgs_back"].value
    D[r] = h5["dark/imgs_back"].value
    
    P[r] = np.ma.masked_equal( P[r], 0)
    D[r] = np.ma.masked_equal( D[r], 0)

    Diff[r] = P[r].mean(0)/ P[r].mean()  - D[r].mean(0) / D[r].mean()
    s = (r, D[r].shape[0], P[r].shape[0])
    print("run %d: %d dark, %d pumped"%s)

import h5py
#runs = range( 260, 264)
#runs = range( 258, 260)
runs = range( 255, 258)
for r in runs:
    h5 = h5py.File("run%d_ALL.h5py"%r,"r")
    P[r] = h5["pumped/imgs_back"].value
    D[r] = h5["dark/imgs_back"].value
    
    P[r] = np.ma.masked_equal( P[r], 0)
    D[r] = np.ma.masked_equal( D[r], 0)

    Diff[r] = P[r].mean(0)/ P[r].mean()  - D[r].mean(0) / D[r].mean()
    s = (r, D[r].shape[0], P[r].shape[0])
    print("run %d: %d dark, %d pumped"%s)



runs = range( 255, 258)


vmin=10
vmax=100
aspect='auto'
cmap='gnuplot'

subplot(221)
r = 241
img = mean( [  d / d.mean() for d in D[r]], 0)
imshow( img, cmap=cmap, aspect=aspect, vmin=vmin, vmax=vmax)

subplot(222)
r = 241
img = mean( [  d / d.mean() for d in P[r]], 0)
imshow( img, cmap=cmap, aspect=aspect, vmin=vmin, vmax=vmax)

subplot(223)
r = 245
img = mean( [  d / d.mean() for d in P[r]], 0)
imshow( img, cmap=cmap, aspect=aspect, vmin=vmin, vmax=vmax)

subplot(224)
r = 258
img = mean( [  d / d.mean() for d in P[r]], 0)
imshow( img, cmap=cmap, aspect=aspect, vmin=vmin, vmax=vmax)

subplot(221)
gca().axis("off")
title("dark")

subplot(222)
gca().axis("off")
title("10 ps")

subplot(223)
gca().axis("off")
title("500 ps")

subplot(224)
gca().axis("off")
title("8 ns")



figure(2)
subplot(221)
r = 181
img = mean( [  d / d.mean() for d in D[r]], 0)
imshow( img, cmap=cmap, aspect=aspect, vmin=vmin, vmax=vmax)

subplot(222)
r = 181
img = mean( [  d / d.mean() for d in P[r]], 0)
imshow( img, cmap=cmap, aspect=aspect, vmin=vmin, vmax=vmax)

subplot(223)
r = 183
img = mean( [  d / d.mean() for d in P[r]], 0)
imshow( img, cmap=cmap, aspect=aspect, vmin=vmin, vmax=vmax)

subplot(224)
r = 184
img = mean( [  d / d.mean() for d in P[r]], 0)
imshow( img, cmap=cmap, aspect=aspect, vmin=vmin, vmax=vmax)


subplot(221)
gca().axis("off")
title("dark")

subplot(222)
gca().axis("off")
title("1 ns")

subplot(223)
gca().axis("off")
title("16 ns")

subplot(224)
gca().axis("off")
title("32 ns")






#
nrows=2
ncols=6
titles = [ "8ns","16ns", "32ns", "64ns", "128ns", "256ns"  ]
titles += [ "dark"]*6 #"8ns","16ns", "32ns", "64ns", "128ns", "256ns"  ]
fig,axs = subplots(nrows=nrows, ncols=ncols)
k = 0
#imgs = {}
#Dimgs = {}
vmin=0
vmax=20
k=0
runs = range( 258, 264) + range( 258, 264)
for i in xrange( nrows):
    for j in xrange( ncols):
        r = runs[k]   
        ax = axs[i][j]
        ax.clear()
        if i == 0:
            img = imgs[r]
        else:
            img = Dimgs[r]
        #img = mean( [ I / I[ I>0].mean() for I in D[r] ] , 0)
        #Dimgs[r]= img
        ax.imshow( img, cmap='gnuplot', 
            aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_title(titles[k])
        ax.axis('off')
        k += 1




#################
# rad pro oil and water shit
import pylab as plt
import numpy as np
import h5py
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from loki.utils.postproc_helper import is_outlier

def remove_std_inds(peaks, inds =None): 
    ol = [ where( is_outlier(p))[0] for p in peaks.T]
    bad = unique( concatenate( ol))
    return np.delete( peaks, bad, axis=0), np.delete( inds, bad, axis=0)

def cluster_pros(  peaks_, n_clusters=3, times=None, inds_only=False):
    peaks = peaks_.copy()
    peaks /= peaks.mean(1)[:,None]
    n_components = 20
    pca = PCA(n_components=n_components)
    new_peaks = pca.fit_transform( peaks )  
    new_kmeans = KMeans( n_clusters=n_clusters) 
    new_kmeans.fit( new_peaks)
    
    if not inds_only:
        results = [ peaks[new_kmeans.labels_ ==l] 
            for l in xrange( n_clusters)]
    else:
        results = [ np.where( new_kmeans.labels_ ==l)[0]
            for l in xrange( n_clusters)]

    if times is None:
        return results
    else:
        times_results = [ times[new_kmeans.labels_ ==l] 
            for l in xrange( n_clusters)]
        return results, times_results

#n_components = 20
#pca = PCA(n_components=n_components)
#new_peaks = pca.fit_transform( peaks )  
def make_elbow_plot( data, max_nclust=12,):
    """data is Nexample x Nfeatures """
    elbow_plot = []
    max_nclust = 12
    for i in xrange( 1, max_nclust+1):
        print("TRygin kmeans with %d clusters (up to %d)"%(i, max_nclust))
        kmeans = KMeans( n_clusters=i)
        kmeans.fit( data)
        elbow_plot.append( kmeans.score(data))
    plt.xlabel('# of clusters')
    plt.ylabel('how good is the clustering')
    plt.plot( np.arange( 1, max_nclust+1), elbow_plot, '.')

    plt.show()

# quick ana
def norm_peaks_inds( peaks, nmin=None,  inds=None):
    if nmin is not None:
        n = peaks.mean(1)
        peaks = peaks[ n > nmin]
        inds = inds[ n > nmin]

    n = peaks.mean(1)
    good = ~is_outlier(n, 2.)
    peaks = peaks[good]
    inds = inds[ good]
    peaks /= peaks.mean(1)[:,None]
    return peaks, inds

## main:
data = {}

f = h5py.File("profiles_gzip9.h5py", 'r')   # inpit
r1=27 # rad pro min,max
r2=600
pca_pros = []
all_tid = []
all_pid = []
all_run = []
for run in range( 1000):
    try:
        tid = f['run%d'%run]['train_id'].value 
        pid = f['run%d'%run]['pulse_id'].value 
# chronological ordering of indices...
        order = np.lexsort( ( pid, tid ) ) # first sorts tid, then pid

# ordered data:
        tid = tid[order]
        pid = pid[order]
        pros = f['run%d'%run]['profiles'].value[:,r1:r2].astype(np.float32)
        pros_raw = pros.copy()

# remove outlier radials profiles, preserve indices each time
        pros, inds = norm_peaks_inds( pros, nmin=0,
                            inds=np.arange( pros.shape[0] ))
        pros, inds = remove_std_inds( pros, inds)  

        #n_components = 20
        #pca = PCA(n_components=n_components)
        #pca_pros.append( pca.fit_transform( pros )   )

        n = len(tid)
        bad = [i for i in np.arange( n) if i not in inds ]

# elbow plot showed ~3/4 clusters is ideal..
        clus = cluster_pros( pros, 4, inds_only=True )
        #clus.append( np.array(bad) )
    except:
        print("bad %d"%run)
        continue
    #print("doog %d"%run)
    data[run] = {}
    for i_c, c in enumerate(clus):
        raw_inds = inds[c]
        data[run][i_c] =  {"pros":pros_raw[raw_inds], 
            "pulse":pid[raw_inds],"train": tid[raw_inds] }


#melt
all_pros = []
all_pid = []
all_tid = []
all_clust = []
all_runs = []
for k in keys:
    for i in xrange(3):
        n = data[k][i]['pros'].shape[0]
        all_runs.append( [k]*n )
        all_pros.append( data[k][i]['pros'])
        all_clust.append( [i]*n)
        all_tid.append( data[k][i]['train'])
        all_pid.append( data[k][i]['pulse'])
all_runs= np.hstack( all_runs)
all_pros = np.vstack( all_pros)
all_tid = np.hstack( all_tid)
all_pid = np.hstack( all_pid )
all_clust = np.hstack( all_clust)
order = np.lexsort( ( all_pid, all_tid ) )


all_clust = all_clust[order]
all_runs = all_runs[order]
all_pid = all_pid[order]
all_tid = all_tid[order]
all_pros = all_pros[order]



