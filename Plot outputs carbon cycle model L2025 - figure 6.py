#!/usr/bin/env python
# coding: utf-8

# In[8]:


# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import interp1d

# Load model output (assuming the .mat file is in the same directory)

## historical until 1981 + predictions (1982-2016) with 34 set of parameters

## Keeling et al. (2017) simulations
# constant Δ (18 permil) and constant C4 fraction
data1 = loadmat('Simus_constant_delta_constant_fracc4_K2017.mat')  
inputdata1 = data1['inputdata']  # Inputs
d13C1= data1['atmospheric d13C']  # Atmospheric d13CO2

# variable Δ as in K2017 (with CO2 effect from Schubert and Jahren (2015)) and constant C4 fraction
data2 = loadmat('Simus_variable_delta_constant_fracc4_K2017.mat')  
inputdata2 = data2['inputdata']  # Inputs
d13C2= data2['atmospheric d13C']  # Atmospheric d13CO2


## Simulations with 3 boxes and C3/C4 model as in Lavergne et al. 2025 CEE
# box 1: C4 plants / box 2 and 3: C3 plants

# constant Δ with constant C4 fraction
data3 = loadmat('Simus_constant_delta_constant_fracc4_L2025.mat')  
inputdata3 = data3['inputdata']  # Inputs
d13C3= data3['atmospheric - ocean - biosphere d13C']  # Atmospheric d13CO2

# variable Δ with constant C4 fraction and cO2 fertilisation removed for box 1
data4 = loadmat('Simus_variable_delta_constant_fracc4_L2025.mat')  
inputdata4 = data4['inputdata']  # Inputs
d13C4= data4['atmospheric - ocean - biosphere d13C']  # Atmospheric d13CO2

# variable Δ with variable C4 fraction and CUE and cO2 fertilisation removed for box 1
data5 = loadmat('Simus_variable_delta_variale_fracc4_L2025.mat') 
inputdata5 = data5['inputdata']  # Inputs
d13C5= data5['atmospheric - ocean - biosphere d13C']  # Atmospheric d13CO2

# Constants and parameters
C14prog = 0  # Change to zero so that BDprecalc does not attempt to load ex14sourcef input file


# In[9]:


import numpy as np

def BDreaddata(filename):
    """
    Reads a data file with a header ending in '----' and extracts two columns of numerical data.

    Parameters:
        filename (str): Path to the input file.

    Returns:
        tp (numpy.ndarray): Time column data.
        pp (numpy.ndarray): Data column values.

    Raises:
        ValueError: If the file does not contain at least two columns of data or if non-numeric data is encountered.
    """
    with open(filename, 'r') as file:
        # Read the header until a line starting with '----' is found
        while True:
            line = file.readline()
            if line.startswith('----'):
                break

        # Read the rest of the file and filter out non-numeric lines
        data_lines = []
        for line in file:
            # Skip lines that are not numeric (e.g., comments or metadata)
            try:
                # Try to split the line into two floats
                parts = line.strip().split('\t')
                float(parts[0])  # Check if the first part is a float
                float(parts[1])  # Check if the second part is a float
                data_lines.append(line)  # Only add valid numeric lines
            except (ValueError, IndexError):
                continue  # Skip lines that cannot be converted to floats

        # If no valid data lines are found, raise an error
        if not data_lines:
            raise ValueError("No valid numeric data found after the header.")

        # Load the filtered data into a NumPy array
        data = np.loadtxt(data_lines, delimiter='\t')

    # Check if the data has at least two columns
    if data.ndim == 1 or data.shape[1] < 2:
        raise ValueError("The input file must contain at least two columns of data.")

    # Split the data into time (tp) and values (pp)
    tp = data[:, 0]
    pp = data[:, 1]

    return tp, pp


# In[10]:


# Define datafiles used in all simulations
del13atmf = 'c13_cmip6_hist.txt'

pco2atmf = f'co2_hist.txt'

inputdata = {}
inputdata['CO2time'], inputdata['CO2data'] = BDreaddata(pco2atmf)
timespan_CO2 = [inputdata['CO2time'][0], inputdata['CO2time'][-1]]

inputdata['C13time'], inputdata['C13data'] = BDreaddata(del13atmf)
timespan_C13 = [inputdata['C13time'][0], inputdata['C13time'][-1]]

inputdata['C14time'], inputdata['C14data'] = BDreaddata(del14atmf)
timespan_C14 = [inputdata['C14time'][0], inputdata['C14time'][-1]]

# Max, min, mid values
maxd13C1 = np.max(d13C1, axis=2)
mind13C1 = np.min(d13C1, axis=2)
midd13C1 = (maxd13C1 + mind13C1) / 2

maxd13C2 = np.max(d13C2, axis=2)
mind13C2 = np.min(d13C2, axis=2)
midd13C2 = (maxd13C2 + mind13C2) / 2

maxd13C3 = np.max(d13C3, axis=2)
mind13C3 = np.min(d13C3, axis=2)
midd13C3 = (maxd13C3 + mind13C3) / 2

maxd13C4 = np.max(d13C4, axis=2)
mind13C4 = np.min(d13C4, axis=2)
midd13C4 = (maxd13C4 + mind13C4) / 2

maxd13C5 = np.nanmax(d13C5, axis=2)
mind13C5 = np.nanmin(d13C5, axis=2)
midd13C5 = (maxd13C5 + mind13C5) / 2


# In[11]:


# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, rcParams, colors
from matplotlib import gridspec as gspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
import matplotlib.path as mpat
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Define colors 
sspco = np.array([
    [25, 76, 156],
    [62, 153, 17],
    [115, 66, 66],
    [153, 153, 0],
    [153, 51, 0],
    [217, 84, 26]
]) / 255


# In[12]:


# Set up the subplot figure

fig_figure6 = plt.figure(1, figsize=(10,14))

mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'
mpl.rcParams['xtick.top'] = True 
mpl.rcParams['ytick.right'] = True
params = {
    'lines.linewidth':3,
    'axes.facecolor':'white',
    'xtick.color':'k',
    'ytick.color':'k',
    'axes.labelsize': 34,
    'xtick.labelsize':34,
    'ytick.labelsize':34,
    'font.size':34,
    'text.usetex': False,
    "svg.fonttype": 'none'
}
plt.rcParams.update(params)



gs = gspec.GridSpec(3, 3, figure=fig_figure6, hspace=0.4,  width_ratios=[1.4,0.2,1.4],height_ratios=[0.15,0.15,0.15])

k = 0.075

# Figure a
column = 0
row = 0

ax = fig_figure6.add_subplot(gs[row, column])

ax.plot(inputdata['C13time'], inputdata['C13data'],'k-',lw=2.5)
ax.plot(np.arange(1982, 2026), midd13C1[481:, 0, 0]-k, color=sspco[0])
ax.plot(np.arange(1982, 2026), midd13C2[481:, 0, 0]-k, color=sspco[1])

ax.fill_between(np.arange(1982, 2026), mind13C1[481:, 0, 0]-k, maxd13C1[481:, 0, 0]-k, color=sspco[0], alpha=0.5)
ax.fill_between(np.arange(1982, 2026), mind13C2[481:, 0, 0]-k, maxd13C2[481:, 0, 0]-k, color=sspco[1], alpha=0.5)

ax.set_ylabel(u'$\mathregular{\u03B4}^{13}$CO$_{2}$ (‰)',fontsize=14)
ax.set_ylim((-9.0, -7.2))
ax.set_xlim((1970, 2016))
ax.text(0.0, 1.1, '(a) Keeling2017 and Graven2020',transform=ax.transAxes,va = 'top',fontsize=14)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

ax.tick_params(labelsize=12)


# Figure b
column = 2
row = 0

ax = fig_figure6.add_subplot(gs[row, column])

ax.plot(inputdata['C13time'], inputdata['C13data'],'k-',lw=2.5)
ax.plot(np.arange(1982, 2026), midd13C3[481:, 0, 0]-k, color=sspco[0])
ax.plot(np.arange(1982, 2026), midd13C4[481:, 0, 0]-k, color=sspco[1])
ax.plot(np.arange(1982, 2026), midd13C5[481:, 0, 0]-k, color=sspco[2])

ax.fill_between(np.arange(1982, 2026), mind13C3[481:, 0, 0]-k, maxd13C3[481:, 0, 0]-k, color=sspco[0], alpha=0.5)
ax.fill_between(np.arange(1982, 2026), mind13C4[481:, 0, 0]-k, maxd13C4[481:, 0, 0]-k, color=sspco[1], alpha=0.5)
ax.fill_between(np.arange(1982, 2026), mind13C5[481:, 0, 0]-k, maxd13C5[481:, 0, 0]-k, color=sspco[2], alpha=0.5)

ax.set_ylabel(u'$\mathregular{\u03B4}^{13}$CO$_{2}$ (‰)',fontsize=14)
ax.set_ylim((-9, -7.2))
ax.set_xlim((1970, 2016))
ax.text(0.0, 1.1, '(b) This study: C$_{4}$ plants in box 1',transform=ax.transAxes,va = 'top',fontsize=14)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

ax.tick_params(labelsize=12)

ax.legend(['Observed','Constant Δ & F$_{4}$','Variable Δ & constant F$_{4}$','Variable Δ & F$_{4}$'], fontsize=12, loc='center left', bbox_to_anchor= (-1.7, -0.2), ncol=4,
            borderaxespad=0, frameon=False)

fig_figure6.savefig('Figure6.pdf', bbox_inches='tight')

plt.close()


# In[ ]:




