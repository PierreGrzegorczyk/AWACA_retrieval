 1132  conda activate copsp-env
 1133  conda activate LORIS-env
 1134  cd 
 1135  cd AWACA
 1136  ls
 1137  cd Retrieval/DeepSpectralRetrieval-main/spectra_database_creation/
 1138  conda activate cosp-env
 1139  cd 
 1140  cd AWACA/PAMTRA/
 1141  mmt
 1142  llt
 1143  cd ../LORIS
 1144  llt
 1145  cd PAMTRA/
 1146  llt
 1147  ./Start_Pam.sh
 1148  source Start_Pam.sh
 1149  vi Start_Pam.sh
 1150  cd ~/AWACA/Retrieval/DeepSpectralRetrieval-main/spectra_database_creation
 1151  llt
 1152  python3 00_simulate_spectra.py
 1153  python3 00_simulate_spectra.py 0 1000
 1154  conda install -c conda-forge pyarrow

