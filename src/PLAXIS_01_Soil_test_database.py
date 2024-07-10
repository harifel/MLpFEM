import numpy as np
import pandas as pd
from itertools import product
import datetime
from plxscripting.easy import *
import os
import time
import subprocess
import matplotlib.pyplot as plt


# =============================================================================
# Functions to extract information from soil test tool
# =============================================================================
def get_values2(resultobject):
    str_list = s_t.call_and_handle_command('echo {}'.format(resultobject.Name.value))
    result_list = eval(str_list)
    return result_list


# =============================================================================
# Functions to run soil test tool
# =============================================================================

def triaxial_and_oed(E50ref, phi, powerm, nu, Rf, K0NC, dilatancy, CellPressure, Eoedref, Eurref, cref, df):
    """ runs a triaxial test and returns some basic results """
    # Defining HS model parameters in SoilTest tool
    g_t.Material.E50Ref = E50ref
    g_t.Material.phi = phi
    g_t.Material.K0NC = K0NC
    g_t.Material.PowerM = powerm
    g_t.Material.nuUR = nu
    g_t.Material.RF = Rf
    g_t.Material.EOedRef = Eoedref
    g_t.Material.EURRef = Eurref
    g_t.Material.psi = dilatancy
    g_t.Material.cRef = cref

    # Running a triaxial test
    try:
        g_t.Triaxial.CellPressure = CellPressure
        g_t.Triaxial.MaximumStrain = 25
        g_t.Triaxial.Steps = 249

        g_t.calculate(g_t.Triaxial)


        q = get_values2(g_t.Triaxial.Results.DeviatoricStress)
        eps_yy = get_values2(g_t.Triaxial.Results.Epsyy)
        eps_v = get_values2(g_t.Triaxial.Results.TotalVolumetricStrain)
        p = get_values2(g_t.Triaxial.Results.MeanEffStress)


    except:
        q = float("nan")
        eps_yy = float("nan")
        eps_v = float("nan")
        p = float("nan")

    # Running an Oedometer test
    try:
        g_t.Oedometer.PhaseTable[0].Siginc = -400
        g_t.Oedometer.PhaseTable[0].Steps = 374

        g_t.Oedometer.addphase()
        g_t.Oedometer.PhaseTable[1].Siginc = 350
        g_t.Oedometer.PhaseTable[1].Steps = 374
        g_t.calculate(g_t.Oedometer)

        if len(g_t.Oedometer.PhaseTable) > 1:
            g_t.Oedometer.deletephase(1)

        sigma_y = get_values2(g_t.Oedometer.Results.SigyyE)
        eps_yy_oed = get_values2(g_t.Oedometer.Results.Epsyy)


    except:
        sigma_y = float("nan")
        eps_yy_oed = float("nan")

        if len(g_t.Oedometer.PhaseTable) > 1:
            g_t.Oedometer.deletephase(1)


    new_data = {
    "E50ref": E50ref,
    "Eoedref": Eoedref,
    "Eurref": Eurref,
    "phi": phi,
    "cref": cref,
    "psi": dilatancy,
    "m": powerm,
    "nu": nu,
    "Rf": Rf,
    "K0NC": K0NC,
    "CellPressure": CellPressure,
    "q": q,
    "eps_y": eps_yy,
    "eps_v": eps_v,
    "p'": p,
    "sig_y": sigma_y,
    "eps_y_oed": eps_yy_oed
    }

    # Append the new row to the DataFrame
    new_row_df = pd.DataFrame([new_data], columns=df.columns)
    df = pd.concat([df, new_row_df], ignore_index=True)

    return df


def restart_soil_test_tool():
    # Close existing soil test tool connection
    try:
        g_t.close()
        s_i.close()
    except Exception as e:
        print(f"Error closing soil test tool: {e}")

    s_i.new()

    HS_mat = g_i.soilmat("Identification", "HS",

                         ###Mechanical input
                         "SoilModel", 3,

                         "nuUR", 0.2,
                         "E50ref", 30e3,
                         "EOedRef", 30e3,
                         "EURRef", 90e3,
                         "PowerM", 0.5,

                         "UseDefaults", True,
                         "cref", 0,
                         "phi", 35,
                         "psi", 0)

    HS_mat.UseDefaults = False
    HS_mat.K0NC = 0.5

    inputport = 10001

    g_i.soiltest(HS_mat)

    return None


# =============================================================================
# Code for running soil test tool and PLAXIS
# =============================================================================

inputport = 10000
plaxispw = r'9U6^fz!1EZMA?^M8'  # needs user interaction
plaxis_path = r'C:\Program Files\Seequent\PLAXIS 2D 2023.2\\'
plaxis_input = 'Plaxis2DXInput.exe'

# first launch PLAXIS
args = [os.path.join(plaxis_path, plaxis_input),
            "--AppServerPort={}".format(inputport),
            "--AppServerPassWord={}".format(plaxispw)]
inputprocess = subprocess.Popen(args)

s_i, g_i = new_server('localhost', inputport, password=plaxispw)

inputport = 10001
s_t, g_t = new_server('localhost', inputport, password='')

s_i.new()

HS_mat = g_i.soilmat("Identification", "HS",

                     ###Mechanical input
                     "SoilModel", 3,
                     "nuUR", 0.2,
                     "E50ref", 30e3,
                     "EOedRef", 30e3,
                     "EURRef", 90e3,
                     "PowerM", 0.5,
                     "UseDefaults", True,
                     "cref", 0,
                     "phi", 35,
                     "psi", 0)

HS_mat.UseDefaults = False
HS_mat.K0NC = 0.5

g_i.soiltest(HS_mat)


# =============================================================================
# Sampling plan
# =============================================================================

df = pd.DataFrame(columns = ["E50ref","Eoedref","Eurref","phi","cref","psi","m","nu",
                             "Rf", "K0NC", "CellPressure","q","eps_y","eps_v",
                             "p'","sig_y","eps_y_oed"])

N_SAMPLES = 1500

# Generate the initial dataset
df1 = pd.DataFrame({
    "E50ref": np.random.uniform(13000, 75000, size=N_SAMPLES),
    "phi": np.random.uniform(28, 40, size=N_SAMPLES),
    "m": np.random.uniform(0.3, 0.7, size=N_SAMPLES),
    "nu": np.random.uniform(0.1, 0.3, size=N_SAMPLES),
    "Rf": np.random.uniform(0.70, 0.97, size=N_SAMPLES)
})

# # Rf
# df1["Rf"] = 0.90
# K0NC
df1["K0NC"] = 1 - np.sin(np.radians(df1["phi"]))


# Define the list of different values for each parameter
cell_pressure_values = [200]

eoed_values = np.linspace(0.8, 1.2, 3)
eur_values = np.linspace(2, 4, 3)
c_values = 0


# Define the list of functions for psi calculation
psi_functions = [lambda phi: max(phi - 30, 0),  # Ensures that psi is not negative
                 lambda phi: phi / 3, 
                 lambda phi: 0]


# Create a list to hold the temporary DataFrames
temp_dfs = []

# Iterate over each combination of the defined parameters
for p, eoed, eur, psi_func in product(cell_pressure_values, eoed_values, eur_values, psi_functions):
    temp_df = df1.copy()
    temp_df["psi"] = temp_df["phi"].apply(psi_func)  # Apply the psi functions to calculate psi from phi
    temp_df["CellPressure"] = p
    temp_df["Eoedref"] = np.floor(eoed * temp_df["E50ref"] * 10) / 10  # round after at first decimal after comma
    temp_df["Eurref"] = np.ceil(eur * temp_df["E50ref"] * 10) / 10  # round after at first decimal after comma
    temp_df["cref"] = c_values
    temp_dfs.append(temp_df)

# Concatenate all the temporary DataFrames to create the final DataFrame
df_extended = pd.concat(temp_dfs, ignore_index=True)
array = df_extended.values


# =============================================================================
# Conducting runs based on sampling plan 243,000 runs were executed on seperate computers
# =============================================================================
xmin = 0
xmax = int(len(array))

i = 0
for x in array[xmin:xmax]:

    df = triaxial_and_oed(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], df)

    if (i + 1) % 1000 == 0:  # Check if this is the 1000th iteration
        current_datetime = datetime.datetime.now()
        print(f"Status: {i+1+xmin} of {int(len(array))} | {np.round((i+1+xmin)/(len(array))*100, 4)} % | {current_datetime}")

        np.save(f"HardeningSoil_{i+1+xmin}_{xmin}_{xmax}_200kPa.npy", df)
        df.to_csv(f'HardeningSoil_{i+1+xmin}_{xmin}_{xmax}_200kPa.csv', index=False)

        # Restart only the soil test tool
        restart_soil_test_tool()


    i = i + 1

np.save("HardeningSoil_Database_end.npy", df)
df.to_csv('HardeningSoil_Database_end.csv', index=False)
