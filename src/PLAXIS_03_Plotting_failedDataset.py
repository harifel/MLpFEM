import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


######################## Define the text size of each plot globally ###########
SMALL_SIZE = 16
MEDIUM_SIZE = 16
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
######################## Define the text size of each plot globally ###########


path = r"..\data\HardeningSoil_Database.npy"
path_save = r'..\graphics\\'
columns = ['E50ref', 'Eoedref','Eurref','phi','cref',
           'psi','m','nu','Rf','K0NC','CellPressure',
           'q','eps_y','eps_v','p','sig_y','eps_y_oed']


path = r"..\data\HardeningSoil_Database_20240502_part1.npy"
df = np.load(path, allow_pickle=True)
df1 = pd.DataFrame(df, columns=columns)

path = r"..\data\HardeningSoil_Database_20240502_part2.npy"
df = np.load(path, allow_pickle=True)
df2 = pd.DataFrame(df, columns=columns)

df = pd.concat([df1, df2], ignore_index=True)

column_name = "q"
# Replace NaN with 0 and non-NaN with 1
df[column_name] = df[column_name].notna().astype(int)

# Define your custom two-color colormap
colors = ['red', 'green']
cmap = ListedColormap(colors, N=2)

# Extracting data columns from DataFrame
E50ref = df.iloc[:, 0]
Eoedref = df.iloc[:, 1]
Eurref = df.iloc[:, 2]
phi = df.iloc[:, 3]
cref = df.iloc[:, 4]
psi = df.iloc[:, 5]
m = df.iloc[:, 6]
nu = df.iloc[:, 7]
Rf = df.iloc[:, 8]
k0 = df.iloc[:, 9]
cellpressure = df.iloc[:, 10]
test = df.iloc[:, 11]

fig, ax = plt.subplots(2, 5, figsize=(30, 25))
s = 5

sc0 = ax[0,0].scatter(E50ref, Eoedref, c=test, cmap=cmap, s=s)
ax[0,0].set_xlabel('E50ref')
ax[0,0].set_ylabel('Eoedref')
ax[0,0].grid()

sc1 = ax[0,1].scatter(E50ref, Eurref, c=test, cmap=cmap, s=s)
ax[0,1].set_xlabel('E50ref')
ax[0,1].set_ylabel('Eurref')
ax[0,1].grid()

sc2 = ax[0,2].scatter(E50ref, m, c=test, cmap=cmap, s=s)
ax[0,2].set_xlabel('E50ref')
ax[0,2].set_ylabel('m')
ax[0,2].grid()

sc3 = ax[0,3].scatter(psi, m, c=test, cmap=cmap, s=s)
ax[0,3].set_xlabel('psi')
ax[0,3].set_ylabel('m')
ax[0,3].grid()

sc4 = ax[0,4].scatter(phi, nu, c=test, cmap=cmap, s=s)
ax[0,4].set_xlabel('phi')
ax[0,4].set_ylabel('nu')
ax[0,4].grid()
fig.colorbar(sc4, ax=ax[0,4], label='Failed test (0 = failed, 1 = passed))',ticks=[0, 1])


sc0 = ax[1,0].scatter(Eoedref, Rf, c=test, cmap=cmap, s=s)
ax[1,0].set_xlabel('Eoedref')
ax[1,0].set_ylabel('Rf')
ax[1,0].grid()

sc1 = ax[1,1].scatter(E50ref, phi, c=test, cmap=cmap, s=s)
ax[1,1].set_xlabel('E50ref')
ax[1,1].set_ylabel('phi')
ax[1,1].grid()

sc2 = ax[1,2].scatter(phi, m, c=test, cmap=cmap, s=s)
ax[1,2].set_xlabel('phi')
ax[1,2].set_ylabel('m')
ax[1,2].grid()

sc3 = ax[1,3].scatter(cref, phi, c=test, cmap=cmap, s=s)
ax[1,3].set_xlabel('cref')
ax[1,3].set_ylabel('phi')
ax[1,3].grid()

sc4 = ax[1,4].scatter(phi, cellpressure, c=test, cmap=cmap, s=s)
ax[1,4].set_xlabel('phi')
ax[1,4].set_ylabel('cellpressure')
ax[1,4].grid()
fig.colorbar(sc4, ax=ax[1,4], label='Failed test (0 = failed, 1 = passed))',ticks=[0, 1])

plt.subplots_adjust(wspace=0.7)
plt.tight_layout()
plt.savefig(path_save + "Failed_dataset.png", dpi = 600)


def custom_parallel_coordinate_plot(df: pd.DataFrame, sep: str , params: list, num: int,
                                    savepath: str = None,
                                    show: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    df = df[df[sep] == 0]

    mins = df[params].min().values
    f = df[params].values-mins
    maxs = np.max(f, axis=0)

    for t in range(num):
        df_temp = df.iloc[t]
        x = np.arange(len(params))
        y = df_temp[params].values

        y = y - mins
        y = y / maxs

        if df.iloc[t, 11] == 0:
            ax.plot(x, y, c='black', alpha=0.3, zorder=10)
        else:
            ax.plot(x, y, c='white', alpha=0.0, zorder=1)

    ax.scatter(x, np.zeros(x.shape), color='black')
    ax.scatter(x, np.ones(x.shape), color='black')

    for i in range(len(x)):
        ax.text(x=x[i], y=-0.01, s=round(df[params[i]].min(), 3),
                horizontalalignment='center', verticalalignment='top')
        ax.text(x=x[i], y=1.01, s=round(df[params[i]].max(), 3),
                horizontalalignment='center', verticalalignment='bottom')

    ax.set_xticks(x)
    ax.set_yticks([0, 1])
    ax.set_xticklabels(params, rotation=45, ha='right')
    ax.set_yticklabels([])
    ax.grid()
    ax.set_title('Failed tests - parameter combinations')

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=200)
    if show is False:
        plt.close()


# Define the parameters for the plot
params = ['E50ref', 'Eoedref', 'Eurref','phi', 'cref', 'm', 'nu', 'Rf', 'CellPressure']

custom_parallel_coordinate_plot(df, 'q', params, 500, savepath=path_save+"parallel_coordinate_plot.png", show=True)
