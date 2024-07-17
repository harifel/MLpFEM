import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from ML_Modelevaluation import SoilModel, HSdata_process


######################## Define the text size of each plot globally ###########
SMALL_SIZE = 11
BIGGER_SIZE = 11

plt.rc('font', family='Times New Roman')
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
######################## Define the text size of each plot globally ###########


colors = ['k', 'r', 'g']
markers = ['o', 's', 'd']
markersizes = [5, 2, 2]
markeverys= [1,5,8]

file_names_real = ['Triaxial test_q_eps1.txt',
                   'Triaxial test_epsvol.txt',
                   'Triaxial test_path.txt',
                   'Oedometer.txt']

cm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(2, 2, figsize=(16*cm, 8*cm), dpi=500)

def interpolate_and_plot(ax, file_true,
                         xlabel, ylabel, ylim, xlim,
                         invert_x=False, invert_y=False, degree=15,
                         x_multi = 1, y_multi = 1, interp_change = True,  x_syn = None, num = 250):

    df_data_true = pd.read_csv(file_true, sep="\t", header=None)

    x_true = np.array(df_data_true.iloc[:, 0] * x_multi)
    y_true = np.array(df_data_true.iloc[:, 1] * y_multi)

    ax.plot(x_true, y_true, marker=markers[0], color=colors[0], label='Real data', linewidth=1, markersize=markersizes[0])

    soil_model = SoilModel()

    if interp_change:
        x_syn = np.linspace(0, -0.25, num=num)

        x_true, y_true, x_syn = map(np.array, (x_true, y_true, x_syn))
        y_pred = soil_model.interpolation_scikit(x_true=x_true, y_true=y_true, x_check=x_syn, degree=degree)

        ax.scatter(x_syn, y_pred, marker=markers[0], color=colors[1], label='Interpolated data', s=1, zorder=10)

    else:
        max_q, min_q = max(x_true), min(x_true)
        max_p, min_p = max(y_true), min(y_true)

        # Reversed arrays
        y_pred = np.linspace(max_q, min_q, num=num)
        x_syn = np.linspace(min_p, max_p, num=num)

        # y_pred = soil_model.interpolation_scikit(x_true=y_true, y_true=x_true, x_check=x_syn, degree=degree, include_bias=True)
        #ax.scatter(y_pred, x_syn, marker=markers[0], color=colors[1], label='Interpolated data', s=1, zorder=10)
        ax.scatter(y_pred, x_syn, marker=markers[0], color=colors[1], label='Interpolated data', s=1, zorder=10)



    ax.grid(True, linewidth=0.5)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    if invert_x:
        ax.invert_xaxis()
    if invert_y:
        ax.invert_yaxis()

    ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return x_syn, y_pred


def plot_oedometer_test(ax, file_true, xlabel, ylabel, ylim, xlim, x_multi, y_multi, degree=4):

    df_data_true = pd.read_csv(file_true, sep="\t", header=None)
    sig_1_true = df_data_true.iloc[:, 0] * x_multi
    eps_1_true = df_data_true.iloc[:, 1] * y_multi

    sig_1_true, eps_1_true = map(np.array, (sig_1_true, eps_1_true))
    min_index = np.argmin(sig_1_true)

    #loading
    eps1_section_1 = eps_1_true[:min_index + 1]
    sig1_section_1 = sig_1_true[:min_index + 1]

    #undloading
    eps1_section_2 = eps_1_true[min_index:]
    sig1_section_2 = sig_1_true[min_index:]

    # Filter for sigma not less than -50
    mask = sig1_section_2 <= -50
    cutoff_indices = np.where(~mask)[0]  # Indices that do not meet the condition


    eps1_section_2 = eps1_section_2[:cutoff_indices[0]]
    sig1_section_2 = sig1_section_2[:cutoff_indices[0]]

    eps1_section_1_syn = np.linspace(0, min(eps1_section_1), num=375)
    eps1_section_2_syn = np.linspace(min(eps1_section_1), max(eps1_section_2), num=375)

    soil_model = SoilModel()
    sig1_pred_section_1 = soil_model.interpolation_scikit(x_true=eps1_section_1, y_true=sig1_section_1, x_check=eps1_section_1_syn, degree=degree)
    sig1_pred_section_2 = soil_model.interpolation_scikit(x_true=eps1_section_2, y_true=sig1_section_2, x_check=eps1_section_2_syn, degree=degree)

    ax.plot(eps_1_true[:min_index + 1], sig_1_true[:min_index + 1], marker=markers[0], color=colors[0], label='Real data', linewidth=1, markersize=markersizes[0])
    ax.plot(eps_1_true[min_index:], sig_1_true[min_index:], marker=markers[0], color=colors[0], linewidth=1, markersize=markersizes[0])

    ax.scatter(eps1_section_1_syn, sig1_pred_section_1, marker=markers[0], color=colors[1], label='Interpolated data', s=1, zorder=10)
    ax.scatter(eps1_section_2_syn, sig1_pred_section_2, marker=markers[0], color=colors[1], s=1, zorder=10)

    ax.grid(True, linewidth=0.5)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))

    ax.invert_xaxis()
    ax.invert_yaxis()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)

    ax.set_xlabel('$\epsilon_{1}$(-)')
    ax.set_ylabel("$\sigma_{1}'$ (kPa)")

    interpolated_data = np.vstack((np.column_stack((eps1_section_1_syn, sig1_pred_section_1)), np.column_stack((eps1_section_2_syn, sig1_pred_section_2))))

    return interpolated_data


folder = r'01_analysis_RealTest_Karlsruhe_200kPa'
#folder = r'02_analysis_RealTest_Karlsruhe_100kPa'
path_real = rf'C:\Users\haris\Documents\GitHub\MLpFEM\data\{folder}\\'

# Plot 1: Triaxial Test - Deviatoric Stress and Eps1
file_true = path_real + r"\01_rawdata_cropped\\" + file_names_real[0]
x_syn, y_pred_q = interpolate_and_plot(axs[0, 0], file_true, '$\epsilon_{1}$(-)', 'q (kPa)',
                                      (0, 1000), (0, -0.30), invert_x=True,
                                      x_multi = 1 / 100 * (-1))
# x_syn, y_pred_q = interpolate_and_plot(axs[0, 0], file_true, '$\epsilon_{1}$(-)', 'q (kPa)',
#                                      (0, 1000), (0, -0.30), invert_x=True,degree=10,
#                                      x_multi = (-1))
# Save the interpolated data
np.savetxt(path_real + rf".\02_rawdata_interpolated\{file_names_real[0]}", np.column_stack((x_syn, y_pred_q)), delimiter='\t')



# Plot 2: Triaxial Test - Volumetric Strain and Eps1
file_true = path_real + r"\01_rawdata_cropped\\" + file_names_real[1]

x_syn, y_pred = interpolate_and_plot(axs[0, 1], file_true, '$\epsilon_{1}$(-)', '$\epsilon_{vol}$ (-)',
                                      (-0.05, 0.10), (0, -0.30), invert_x=True, degree=10,
                                      x_multi = 1/100*(-1), y_multi = (-1)/100)

# x_syn, y_pred = interpolate_and_plot(axs[0, 1], file_true, '$\epsilon_{1}$(-)', '$\epsilon_{vol}$ (-)',
#                                      (-0.05, 0.10), (0, -0.30), invert_x=True, degree=10,
#                                      x_multi = (-1), y_multi = -1)
# Save the interpolated data
np.savetxt(path_real + rf".\02_rawdata_interpolated\{file_names_real[1]}", np.column_stack((x_syn, y_pred)), delimiter='\t')


# Plot 3: Triaxial Test - Deviatoric and isotropic stress
file_true = path_real + r"\01_rawdata_cropped\\" + file_names_real[2]

x_syn, y_pred = interpolate_and_plot(axs[1, 0], file_true, "$p'$ (kPa)", '$q$ (kPa)',
                                      (0, 1000), (0, -600), invert_x=True, degree=10,
                                      x_multi = -1, y_multi = 1, x_syn = y_pred_q, interp_change = False, num = 250)

# x_syn, y_pred = interpolate_and_plot(axs[1, 0], file_true, "$p'$ (kPa)", '$q$ (kPa)',
#                                      (0, 1000), (0, -900), invert_x=True, degree=5,
#                                      x_multi = -1, y_multi = 1, x_syn = y_pred_q, interp_change = False)
# Save the interpolated data
np.savetxt(path_real + rf".\02_rawdata_interpolated\{file_names_real[2]}", np.column_stack((y_pred, x_syn)), delimiter='\t')



# Plot 4: Oedometer test
file_true = path_real + r"\01_rawdata_cropped\\" + file_names_real[-1]
interpolated_data = plot_oedometer_test(axs[1, 1], file_true, xlabel = '$\epsilon_{1}$(-)', ylabel = "$\sigma_{1}'$ (kPa)",
                    ylim = (0, -500), xlim = (0, -0.030),
                    x_multi = -1, y_multi = (-1) / 100)
# interpolated_data = plot_oedometer_test(axs[1, 1], file_true, xlabel = '$\epsilon_{1}$(-)', ylabel = "$\sigma_{1}'$ (kPa)",
#                     ylim = (0, -300), xlim = (0, -0.030),
#                     x_multi = -1, y_multi = (-1))

np.savetxt(path_real + rf".\02_rawdata_interpolated\{file_names_real[-1]}", interpolated_data, delimiter='\t')


plt.legend(loc='upper center', bbox_to_anchor=(-.25, -.38), frameon=False, ncol=3)
plt.subplots_adjust(left=0.12, right=0.97, bottom=0.22, top=0.98, wspace=0.40, hspace=0.5)
plt.savefig(f"RealTest_interpolation_{folder}.png", dpi=500)
