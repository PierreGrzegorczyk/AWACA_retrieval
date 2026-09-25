import xarray as xr
import numpy as np
from netCDF4 import Dataset
import matplotlib.pylab as plt
import numpy as np
from datetime import datetime, timedelta
import matplotlib.colors as colors
import matplotlib.dates as mdates

##
sites=["D17"]

month='02'
moments=True
day=np.arange(15,19,1)

for site in sites:
    time=np.arange(0,24,1)

    globals()['Ze_'+site]=[]
    globals()['Dopp_'+site]=[]
    globals()['RMSg_'+site]=[]
    # globals()['SKWg_'+site]=[]
    # globals()['Time_'+site]=[]

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
                globals()['Dopp_'+site].append(nc_data['VELg'][:])
                # globals()['Time_'+site].append(nc_data['time'][:])
                globals()['RMSg_'+site].append(nc_data['RMSg'][:])
                # globals()['SKWg_'+site].append(nc_data['SKWg'][:])
                dz=nc_data['drg'][:]
                globals()['Alt_mira_'+site]=np.arange(0,np.shape(globals()['Ze_'+site][0])[1]*dz,dz)

            except:
                print('no data: ',site+" Mira: day = ",d," time = ",t)



    Time_mira=np.concatenate(globals()['Time_'+site])

    globals()['Ze_'+site+'_mira']=np.concatenate(globals()['Ze_'+site])
    globals()['Dopp_'+site+'_mira']=np.concatenate(globals()['Dopp_'+site])
    globals()['RMSg_'+site+'_mira']=np.concatenate(globals()['RMSg_'+site])
    globals()['SKWg_'+site+'_mira']=np.concatenate(globals()['SKWg_'+site])

    globals()['Time_mira_utc'+site] = [datetime.utcfromtimestamp(ts) for ts in Time_mira]


    ds = xr.Dataset(data_vars=dict(Ze=(["time", "height"], globals()['Ze_'+site+'_mira']),
            Dopp=(["time", "height"], globals()['Dopp_'+site+'_mira']),
            RMSg=(["time", "height"], globals()['RMSg_'+site+'_mira']),
            SKWg=(["time", "height"], globals()['SKWg_'+site+'_mira']),),
        coords=dict(time=Time_mira,
            height=globals()['Alt_mira_'+site],))

    ds["time_utc"] = xr.DataArray(globals()['Time_mira_utc'+site], dims="time")

    ds.to_netcdf("/home/grzegorc/AWACA_DATA/Mira/"+site+"/Mira_"+str(month)+"_"+str(day[0])+"_"+str(day[-1])+"_"+site+".nc")
    ds.close()
