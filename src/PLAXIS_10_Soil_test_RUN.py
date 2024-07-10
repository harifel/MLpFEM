import numpy as np
import pandas as pd
from itertools import product
#from Soil_Test import *
#from Soil_Test import plaxis_shut_down
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

def x_array(x_matrix):
    X_return = []
    for s in x_matrix:
        # Remove '[' and ']' characters, split by '\n' to get individual lines
        s = str(s)
        lines = s.strip('[]').split(',')
        values = []
        for line in lines:
            # Split each line by spaces, convert values to floats, and extend the list
            values.extend([float(val) for val in line.split()])
        X_return.append(np.array(values))
    X_return = np.array(X_return)
    return X_return

# =============================================================================
# Functions to run soil test tool
# =============================================================================

def triaxial_and_oed(E50ref, Eoedref, Eurref, phi, cref, dilatancy, powerm, nu, Rf, K0NC, CellPressure, path):

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


        with open(path + 'Triaxial test_q_eps1.txt', 'w', encoding='utf-8') as file:
            for eps, q_value in zip(eps_yy, q):
                file.write(f"{float(eps):.6f}\t{float(q_value):.6f}\n")

        with open(path + 'Triaxial test_epsvol.txt', 'w', encoding='utf-8') as file:
            for eps, q_value in zip(eps_yy, eps_v):
                file.write(f"{float(eps):.6f}\t{float(q_value):.6f}\n")

        with open(path + 'Triaxial test_path.txt', 'w', encoding='utf-8') as file:
            for eps, q_value in zip(p, q):
                file.write(f"{float(eps):.6f}\t{float(q_value):.6f}\n")

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

        with open(path + 'Oedometer.txt', 'w', encoding='utf-8') as file:
            for eps, q_value in zip(eps_yy_oed, sigma_y):
                file.write(f"{float(eps):.6f}\t{float(q_value):.6f}\n")
    except:
        sigma_y = float("nan")
        eps_yy_oed = float("nan")

        if len(g_t.Oedometer.PhaseTable) > 1:
            g_t.Oedometer.deletephase(1)

    return None

# =============================================================================
# Code for running soil test tool and PLAXIS
# =============================================================================

inputport = 10010
plaxispw = r'9U6^fz!1EZMA?^M8'  # needs user interaction
plaxis_path = r'C:\Program Files\Seequent\PLAXIS 2D 2023.2\\'
plaxis_input = 'Plaxis2DXInput.exe'

# first launch PLAXIS
args = [os.path.join(plaxis_path, plaxis_input),
            "--AppServerPort={}".format(inputport),
            "--AppServerPassWord={}".format(plaxispw)]
inputprocess = subprocess.Popen(args)

s_i, g_i = new_server('localhost', inputport, password=plaxispw)

inputport = 10000
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

#x = pd.read_csv(r'..\data\pred_analysis_realTest_karlsruhe.txt', delimiter='\t', header=None)
# x = pd.read_csv(r'C:\Users\haris\Documents\GitHub\MLpFEM\data\pred_analysis_realTest_karlsruhe.txt', delimiter='\t', header=None)
# x = pd.read_csv(r'C:\Users\haris\Documents\GitHub\MLpFEM\data\pred_analysis_synthetictest_101_SyntheticExampleB.txt', delimiter='\t', header=None)
# x = pd.read_csv(r'C:\Users\haris\Documents\GitHub\MLpFEM\data\pred_analysis_realTest_02_analysis_RealTest_Karlsruhe_100kPa_new.txt', delimiter='\t', header=None)
# x = pd.read_csv(r'C:\Users\haris\Documents\GitHub\MLpFEM\data\pred_analysis_realTest_01_analysis_RealTest_Karlsruhe_200kPa_new_database.txt', delimiter='\t', header=None)
# x = pd.read_csv(r'C:\Users\haris\Documents\GitHub\MLpFEM\data\pred_analysis_synthetictest_102_SyntheticExample_Final.txt', delimiter='\t', header=None)
# x = pd.read_csv(r'C:\Users\haris\Documents\GitHub\MLpFEM\data\pred_analysis_realTest_01_analysis_RealTest_Karlsruhe_200kPa_Islam.txt', delimiter='\t', header=None)
# x = pd.read_csv(r'C:\Users\haris\Documents\GitHub\MLpFEM\data\pred_analysis_realTest_02_analysis_RealTest_Karlsruhe_100kPa_Islam.txt', delimiter='\t', header=None)

# path = r'..\\data\06_analysis_RealTest_Karlsruhe\10_MachineLearning\\'
# path = r'..\\data\100_SyntheticExampleA\11_SoilTestOutput\\'
# path = r'..\\data\01_analysis_RealTest_Karlsruhe_200kPa\10_MachineLearning\\'
# path = r'..\\data\\102_SyntheticExample_Final\10_MachineLearning\\'
#path = r'..\\data\02_analysis_RealTest_Karlsruhe_100kPa\11_Islam\\'

# triaxial_and_oed(x.iloc[0,1], x.iloc[1,1], x.iloc[2,1], x.iloc[3,1],
#                         x.iloc[4,1], x.iloc[5,1], x.iloc[6,1], x.iloc[7,1],
#                         x.iloc[8,1], x.iloc[9,1], x.iloc[10,1], path=path)

for i in range(5):
    list_exe = rf'..\data\10_summary_pred{i+1}.txt'
    x = pd.read_csv(list_exe, delimiter='\t', header=None)
    path = rf'..\\data\10_EBO\{i+1}_Pred{i+1}\\'

    triaxial_and_oed(x.iloc[0,1], x.iloc[1,1], x.iloc[2,1], x.iloc[3,1],
                          x.iloc[4,1], x.iloc[5,1], x.iloc[6,1], x.iloc[7,1],
                          x.iloc[8,1], x.iloc[9,1], x.iloc[10,1], path=path)
