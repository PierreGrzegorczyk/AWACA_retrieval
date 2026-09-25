import numpy as np
import h5py
import json
import argparse


parser = argparse.ArgumentParser(description='prepare files for training')
parser.add_argument('config_file', type=str)
args = parser.parse_args()

config = json.load(open(args.config_file))

#config = json.load(open('/home/grzegorc/AWACA/Retrieval/DeepSpectralRetrieval-main/direct_retrieval/config.json'))
data_path = "/home/grzegorc/AWACA/Retrieval/DeepSpectralRetrieval-main/spectra_database_creation/output_data/"

h5_file = data_path + 'batch0_0_all_wind_ts.h5'  # input file
output_file = data_path + 'targets.npy'
output_file_mean = data_path + 'targets_mean.npy'
output_file_std = data_path + 'targets_std.npy'

## Generate npy files of target values
target_keys = [
    'a_mass_size', 'alpha_area_size', 'aspect_ratio', 'b_mass_size',
    'beta_area_size', 'dmean', 'iwc', 'noise_level_W', 'noise_level_Ka',
    'windW', 'windKa',
]

with h5py.File(h5_file, 'r') as data:
    # scalar target variables
    cols = [np.asarray(data[key]) for key in target_keys]

    # SIGMA_TURB is 2D (N, 2) — append its two columns in [:,1] then [:,0] order
    sigma_turb = np.asarray(data['SIGMA_TURB'])
    cols.append(sigma_turb[:, 1])
    cols.append(sigma_turb[:, 0])

    targets = np.column_stack(cols)

    targets_mean = np.array([np.mean(c) for c in cols])
    targets_std = np.array([np.std(c) for c in cols])

np.save(output_file, targets)
print('Saved:', output_file, targets.shape)

np.save(output_file_mean, targets_mean)
print('Saved:', output_file_mean, targets_mean.shape)

np.save(output_file_std, targets_std)
print('Saved:', output_file_std, targets_std.shape)


## Generate json and npy files for normalization
# config_file = sys.argv[1]
#
# with open(config_file, 'r') as f:
#     config = json.load(f)
#
# print(config)

i_end_ds=config["i_end_ds"]
i_start_ds=config["i_start_ds"]

spectrum_W_256=h5py.File(h5_file, 'r')['spectrum_W_256'][config["i_start_ds"]:config["i_end_ds"],:]
spectrum_Ka_256=h5py.File(h5_file, 'r')['spectrum_Ka_256'][config["i_start_ds"]:config["i_end_ds"],:]


#std file
output = {"spectrum_W_256": np.std(spectrum_W_256).tolist(),"spectrum_Ka_256": np.std(spectrum_Ka_256).tolist()}

with open(data_path+"spectrum_std.json", "w") as f:
    json.dump(output, f)

print('Saved:', data_path+"spectrum_std.json")


#mean file
output = {"spectrum_W_256": np.mean(spectrum_W_256).tolist(),"spectrum_Ka_256": np.mean(spectrum_Ka_256).tolist()}

with open(data_path+"spectrum_mean.json", "w") as f:
    json.dump(output, f)



print('Saved:', data_path+"spectrum_mean.json")















#std = np.load(data_path + 'targets_mean.npy')
#print(std)
#print(np.any(std == 0), np.any(np.isnan(std)))


#h5_file = data_path + 'batch0_0_all_wind_ts.h5'  # input file

#targets = np.load(data_path + 'targets.npy')
#targets = data_path + 'batch0_0_merged.h5'
#print(np.any(np.isnan(targets)), np.any(np.isinf(targets)))
#with h5py.File(h5_file, 'r') as f:
#    print(np.any(np.isnan(f['spectrum_Ka_256'][:])))



#    h5_file="/home/grzegorc/AWACA/Retrieval/DeepSpectralRetrieval-main/spectra_database_creation/output_data/batch0_0_all_wind_ts.h5"
