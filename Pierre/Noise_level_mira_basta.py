import xarray as xr
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from datetime import datetime, timedelta
import matplotlib.colors as colors
import matplotlib.dates as mdates

sites=["D17"]

month='02'
moments=False
day=np.arange(15,18,1)




## Mira

for site in sites:
    time=np.arange(0,24,1)

    globals()['Ze_'+site]=[]


    for d in day:
        for t in time:
            print(site+" Mira: day = ",d," time = ",t)
            if len(str(d))==1:
                d="0"+str(d)

            if t>9:
                if moments==False:
                    file_path = "/home/grzegorc/AWACA_DATA/Mira/"+site+"/"+month+"/"+str(d)+"/2025"+month+str(d)+"_"+str(t)+"00.znc"

                else:
                    file_path = "/home/grzegorc/AWACA_DATA/Mira/"+site+"/"+month+"/"+str(d)+"/2025"+month+str(d)+"_"+str(t)+"00_moments.znc"

            else:
                if moments==False:
                    file_path = "/home/grzegorc/AWACA_DATA/Mira/"+site+"/"+month+"/"+str(d)+"/2025"+month+str(d)+"_0"+str(t)+"00.znc"

                else:
                    file_path = "/home/grzegorc/AWACA_DATA/Mira/"+site+"/"+month+"/"+str(d)+"/2025"+month+str(d)+"_0"+str(t)+"00_moments.znc"


            try:
                nc_data = Dataset(file_path, "r")
                globals()['Ze_'+site].append(10*np.log10(nc_data['Zg'][:]))
                dz=nc_data['drg'][:]
                globals()['Alt_mira_'+site]=np.arange(0,np.shape(globals()['Ze_'+site][0])[1]*dz,dz)
            except:
                print('no data: ',site+" Mira: day = ",d," time = ",t, file_path)

    globals()['Ze_'+site+'_mira']=np.concatenate(globals()['Ze_'+site])

    globals()['Noise_'+site] = np.nanmin(globals()['Ze_'+site+'_mira'], axis=0)
    # globals()['Noise_'+site] = 10**(np.nanmin(globals()['Ze_'+site+'_mira'], axis=0)/10.)



def Noise_funct(r, slope, intercept):

    Noise = np.log10(r/1000)*slope + intercept

    return Noise


def fit_curve(r, Noise):
    mask = r > 0
    popt, pcov = curve_fit(Noise_funct,r[mask],Noise[mask],p0=[1e-3, 1e-3],method='lm',maxfev=10000)
    return popt


# Fit each station
slope, intercept = fit_curve(globals()['Alt_mira_'+site][10:],globals()['Noise_'+site][10:])

plt.figure('Mira noise level')

plt.scatter(globals()['Noise_'+site],globals()['Alt_mira_'+site]/1000,label='Noise Mira')

plt.plot(Noise_funct(globals()['Alt_mira_'+site], slope, intercept),globals()['Alt_mira_'+site]/1000,label='Fitted curve',color='orange')

plt.xlabel('Noise level [dBZ]')
plt.ylabel('Altitude [km]')

plt.legend()
plt.show()


## Basta
