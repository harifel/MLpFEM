import pandas as pd
import glob
import os
import numpy as np

path = r'D:\MLpFEM_Database\\'

path = r'D:\_MLpFEM_Database\__endfiles\_Database\\'
path = r'C:\Users\haris\Documents\GitHub\MLpFEM\data\_database_Hostun\\'
path = r'C:\Users\haris\Documents\GitHub\MLpFEM\data\_database_psi_200kPa\\'


# List all NPY files in the specified path
npy_files = glob.glob(path + '*.npy')

# Sort the list of files by modification time
sorted_files = sorted(npy_files, key=os.path.getmtime)


columns = ['E50ref', 'Eoedref','Eurref','phi','cref',
           'psi','m','nu','Rf','K0NC','CellPressure',
           'q','eps_y','eps_v','p','sig_y','eps_y_oed']

# Create a new empty DataFrame with the same column names as df
tot_data_npy = pd.DataFrame(columns=columns)

# Print each sorted file in the path
for file in sorted_files:
    print(file)

    df = pd.DataFrame(np.load(file, allow_pickle=True), columns=columns)
    df = df[df['cref'] <= 0]
    tot_data_npy = pd.concat([tot_data_npy, df], ignore_index=True)

# Save NPY data to an NPY file
npy_export_file_path = path + r'\HardeningSoil_Database_psi_0.npy'
np.save(npy_export_file_path, tot_data_npy.to_numpy())
