import matplotlib.pylab as plt
import numpy as np
import h5py



data = h5py.File('/home/grzegorc/AWACA/Retrieval/DeepSpectralRetrieval-main/spectra_database_creation/output_data/batch0_0_all_wind_ts.h5','r')
print(data.keys())



spectrum_Ka_256=data['spectrum_Ka_256']
spectrum_W_256=data['spectrum_W_256']
temperature=data['temperature']
ZE=data['ZE']
MDV=data['MDV']


plt.figure('test')
for i in range(0,112038,1000):
    plt.plot(spectrum_W_256[:][i])
plt.show()




plt.scatter(temperature[:],MDV[:,0])





spc = nc_data["SPCco"][:]
spc_dB = 10 * np.log10(spc)

print(spc.min(), spc.max())



spc = nc_data["SPCcx"][:]

i_time = 100
i_range = 100

s = spc[i_time, i_range, :]

print(s.min(), s.max())
print(10*np.log10(s).min(), 10*np.log10(s).max())





for name in nc_data.variables:
    print(name, nc_data.variables[name].dimensions)



spc_var = nc_data.variables["SPCcx"]

print(spc_var.ncattrs())

for a in spc_var.ncattrs():
    print(a, "=", spc_var.getncattr(a))



dv=nc_data.variables['NyquistVelocity'][:]*2/nc_data.variables['nfft'][:]
SPCco=nc_data.variables['SPCco'][:]
SPCco_dB = 10 * np.log10(SPCco)

SPCco_dv=SPCco * dv

SPCco_dv_dB = 10 * np.log10(SPCco_dv)



