#!/usr/bin/env python
# coding: utf-8

# In[3]:


# This script runs the simple carbon cycle model simulations presented in
# Graven et al. GBC 2020. A spinup and historical simulation is run first,
# then 6 different future scenarios are simulated. The model is run for
# different parameter sets that were selected for their correspondence
# with observations.
# Heather Graven 2020 - modified Python version by Alienor Lavergne 2025

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from datetime import datetime

# Global variables (assumed to be defined elsewhere)
global ak, akam, inputdata, C14prog, C13prog, ndeconv, timespan, ntempfb, epsi, nepsab, pa0, pa013, cm014, pa014, gtc2ppm, nbuff, nfoss, ndifffb, prodoc, rs, R14s, h1, h2, hm, cbio0141, cbio0142, cbio0143, fossfac, akba1, akba2, akba3, akab1, akab2, akab3

# Parameter values
# Keddy : eddy diffusivity (Keddy) of 3,000–6,000 m2·y−1
# τam: atmospheric CO2 residence time with respect to exchange with the mixed layer of 9–11 y (τam, corresponding to piston velocities of 14.8–18.1 cm·h−1)
# β:  CO2 fertilization factor (β) of 0–0.4
# τab1 τab2 τab3 : atmospheric CO2 residence time with respect to biospheric exchange (τab) of 18–25 y,
# τba1 τba2 τba3 : biospheric residence time (τba) of 20–35 y

# only consideration of set of parameters when CO2 fertilisation factor is equal to 0
param = np.array([
    [4000, 10, 0.4, 27.1, 21.6, 210.9, 2.4, 24.8, 580.2],
    [4000, 10, 0.4, 33.8, 25.6, 61.9, 2, 13.4, 212.8],
    [4000, 11, 0.4, 21.2, 31.5, 53.0, 2.6, 24.1, 165.7],
    [3000, 9, 0.4, 33.8, 25.6, 61.9, 2, 13.4, 212.8],
    [3000, 10, 0.4, 33.8, 25.6, 61.9, 2, 13.4, 212.8],
    [3000, 11, 0.4, 33.8, 25.6, 61.9, 2, 13.4, 212.8],
    [3000, 11, 0.4, 21.2, 31.5, 53.0, 2.6, 24.1, 165.7],
    [4000, 9, 0.4, 38.1, 23.0, 107.8, 2, 15.7, 353.2],
    [3000, 9, 0.4, 38.1, 23.0, 107.8, 2, 15.7, 353.2],
    [5000, 11, 0.4, 23.2, 24.4, 105.4, 2.5, 26.2, 299.4],
#    [5000, 10, 0, 33.8, 25.6, 61.9, 2, 13.4, 212.8],
    [4000, 11, 0.4, 23.2, 24.4, 105.4, 2.5, 26.2, 299.4],
 #   [4000, 10, 0, 33.8, 25.6, 61.9, 2, 13.4, 212.8],
#    [4000, 11, 0, 33.8, 25.6, 61.9, 2, 13.4, 212.8],
    [3000, 10, 0.4, 23.2, 24.4, 105.4, 2.5, 26.2, 299.4],
    [3000, 11, 0.4, 23.2, 24.4, 105.4, 2.5, 26.2, 299.4],
 #   [3000, 9, 0, 33.8, 25.6, 61.9, 2, 13.4, 212.8],
#     [3000, 10, 0, 33.8, 25.6, 61.9, 2, 13.4, 212.8],
#     [3000, 11, 0, 33.8, 25.6, 61.9, 2, 13.4, 212.8],
#     [4000, 9, 0, 38.1, 23.0, 107.8, 2, 15.7, 353.2],
#     [4000, 11, 0, 21.2, 31.5, 53.0, 2.6, 24.1, 165.7],
#     [3000, 9, 0, 38.1, 23.0, 107.8, 2, 15.7, 353.2],
#     [3000, 11, 0, 21.2, 31.5, 53.0, 2.6, 24.1, 165.7],
    [3000, 9, 0.4, 27.1, 21.6, 210.9, 2.4, 24.8, 580.2],
    [3000, 10, 0.4, 27.1, 21.6, 210.9, 2.4, 24.8, 580.2],
#     [5000, 11, 0, 23.2, 24.4, 105.4, 2.5, 26.2, 299.4],
#     [4000, 11, 0, 23.2, 24.4, 105.4, 2.5, 26.2, 299.4],
#     [4000, 10, 0, 27.1, 21.6, 210.9, 2.4, 24.8, 580.2],
#     [3000, 10, 0, 23.2, 24.4, 105.4, 2.5, 26.2, 299.4],
#     [3000, 11, 0, 23.2, 24.4, 105.4, 2.5, 26.2, 299.4],
#     [3000, 9, 0, 27.1, 21.6, 210.9, 2.4, 24.8, 580.2],
#     [3000, 10, 0, 27.1, 21.6, 210.9, 2.4, 24.8, 580.2],
    [3000, 11, 0.4, 21.7, 20.1, 232.6, 2.6, 34.7, 502.4] #,
#    [3000, 11, 0, 21.7, 20.1, 232.6, 2.6, 34.7, 502.4]
])


# Model setup
ndeconv = 1  # 0: prognostic, 1: single deconv
nbuff = 1  # 0: constant buffer factor, 1: exact chemistry
nfoss = 1  # 0: exponential emissions, 1: emissions from file
fossfac = 1.0  # fossil fuel emissions scaling factor
ntempfb = 1  # 0: constant T, else variable T from file
ndifffb = 0  # 0: constant diffusivity, else variable diff from file
nepsab = 1 # 0: constant fractionation with land biota, else frac from file
prodoc = 0  # 0: constant marine biosphere pool, else variable mbio from file
fractionc4mode = 0 # 0: constant fraction of C4 plants, else variable from file

# Define datafiles used in all simulations
del13atmf = 'c13_cmip6_hist.txt'
del14atmf = 'c14_cmip6_hist.txt'

# List of future scenario names
SSP = ['119', '126', '245', '3B', '534', '5B']

# Initialize output variable for predictions over 1500-2025
SSPY = np.zeros((len(range(1500,2025)), 3 * 49, len(param), len(SSP)))


# In[6]:


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


# In[7]:


import numpy as np

def cchems(sc, sb, ssi, sp, alk, t, sal, ah):
    # Constants
    nmax = 1000000000000
    eps = 1e-8

    ta = t + 273.15
    tac = ta / 100

    # Reaction constants calculated for given temperature and salinity
    ak1 = 10 ** (13.7201 - 0.031334 * ta - 3235.76 / ta - 1.3e-5 * sal * ta + 0.1032 * np.sqrt(sal))
    a1 = -5371.9645 - 1.671221 * ta + 128375.28 / ta
    a2 = 2194.3055 * np.log10(ta) - 0.22913 * sal - 18.3802 * np.log10(sal)
    a3 = 8.0944e-4 * sal * ta + 5617.11 * np.log10(sal) / ta - 2.136 * sal / ta
    ak2 = 10 ** (a1 + a2 + a3)

    akb = 10 ** (-9.26 + 0.00886 * sal + 0.01 * t)

    aksi = 4e-10

    akp1 = 2e-2
    akp2 = np.exp(-9.039 - 1450 / ta)
    akp3 = np.exp(4.466 - 7276 / ta)

    akw = np.exp(148.9802 - 13847.26 / ta - 23.6521 * np.log(ta) - 0.019813 * sal + np.sqrt(sal) * (-79.2447 + 3298.72 / ta + 12.0408 * np.log(ta)))
    fh = 1.29 - 0.00204 * ta + sal * sal * (4.61e-4 - 1.48e-6 * ta)

    alfa = np.exp(-60.2409 + 9345.17 / ta + 23.3585 * np.log(tac) + sal * (0.023517 - 0.023656 * tac + 0.0047036 * tac * tac))

    n = 0
    a = 1 / ah
    sums = [sc, sb, ssi, sp]
    alka = alk

    x = 0

    while x == 0:
        n += 1
        al = a

        # Calculation of g matrix
        g = np.zeros((4, 3))
        g[0, 0] = 1 + ak1 * a * (1 + ak2 * a)
        g[0, 1] = ak1 * (1 + 2 * ak2 * a)
        g[0, 2] = 2 * ak1 * ak2

        g[1, 0] = 1 + akb * a
        g[1, 1] = akb
        g[1, 2] = 0

        g[2, 0] = 1 + aksi * a
        g[2, 1] = aksi
        g[2, 2] = 0

        g[3, 0] = 1 + akp1 * a * (1 + akp2 * a * (1 + akp3 * a))
        g[3, 1] = akp1 * (1 + akp2 * a * (2 + 3 * akp3 * a))
        g[3, 2] = akp1 * akp2 * (2 + 6 * akp3 * a)

        h = 0
        hs = 0

        # Summation of results
        for w in range(4):
            h += sums[w] * g[w, 1] / g[w, 0]
            hs += sums[w] * (g[w, 2] * g[w, 0] - g[w, 1] * g[w, 1]) / (g[w, 0] * g[w, 0])

        hs += h + a * hs + 1e6 * akw * fh + 1e6 / (a * a * fh)
        h = a * (h + 1e6 * akw * fh) - 1e6 / (a * fh) - alka
        a = a - h / hs

        # Check for convergence
        if n > nmax:
            x = 1
            print(f"No convergence in module CCHEMS, ah0= {ah}")
        elif abs((a - al) / al) < eps:
            x = 1

            ah = 1 / a
            hco3 = sc / (1 + ak2 / ah + ah / ak1)
            co3 = hco3 * ak2 / ah
            co2 = hco3 * ah / ak1
            pco2 = co2 / alfa
            ph = -np.log10(ah / fh)

            # Return the result
            return pco2, ph, co2, hco3, co3

    # If the loop exits without convergence, return None
    return None, None, None, None, None


# In[8]:


import numpy as np

def cchems_co3out(sc, sb, ssi, sp, alk, t, sal, ah):
    nmax = 1000000000000
    eps = 1e-8

    ta = t + 273.15
    tac = ta / 100

    # Reaction constants calculated for given Temp and Salinity
    ak1 = 10**(13.7201 - 0.031334 * ta - 3235.76 / ta - 1.3e-5 * sal * ta + 0.1032 * np.sqrt(sal))
    a1 = -5371.9645 - 1.671221 * ta + 128375.28 / ta
    a2 = 2194.3055 * np.log10(ta) - 0.22913 * sal - 18.3802 * np.log10(sal)
    a3 = 8.0944e-4 * sal * ta + 5617.11 * np.log10(sal) / ta - 2.136 * sal / ta
    ak2 = 10**(a1 + a2 + a3)

    akb = 10**(-9.26 + 0.00886 * sal + 0.01 * t)
    aksi = 4 * 10**-10

    akp1 = 2 * 10**-2
    akp2 = np.exp(-9.039 - 1450 / ta)
    akp3 = np.exp(4.466 - 7276 / ta)

    akw = np.exp(148.9802 - 13847.26 / ta - 23.6521 * np.log(ta) - 0.019813 * sal + np.sqrt(sal) * (-79.2447 + 3298.72 / ta + 12.0408 * np.log(ta)))
    fh = 1.29 - 0.00204 * ta + sal * sal * (4.61e-4 - 1.48e-6 * ta)

    alfa = np.exp(-60.2409 + 9345.17 / ta + 23.3585 * np.log(tac) + sal * (0.023517 - 0.023656 * tac + 0.0047036 * tac * tac))

    n = 0
    a = 1 / ah
    sums = [sc, sb, ssi, sp]
    alka = alk

    x = 0
    while x == 0:
        n += 1
        al = a

        g = np.zeros((4, 3))
        g[0, 0] = 1 + ak1 * a * (1 + ak2 * a)
        g[0, 1] = ak1 * (1 + 2 * ak2 * a)
        g[0, 2] = 2 * ak1 * ak2

        g[1, 0] = 1 + akb * a
        g[1, 1] = akb
        g[1, 2] = 0

        g[2, 0] = 1 + aksi * a
        g[2, 1] = aksi
        g[2, 2] = 0

        g[3, 0] = 1 + akp1 * a * (1 + akp2 * a * (1 + akp3 * a))
        g[3, 1] = akp1 * (1 + akp2 * a * (2 + 3 * akp3 * a))
        g[3, 2] = akp1 * akp2 * (2 + 6 * akp3 * a)

        h = 0
        hs = 0

        for w in range(4):
            h += sums[w] * g[w, 1] / g[w, 0]
            hs += sums[w] * (g[w, 2] * g[w, 0] - g[w, 1] ** 2) / (g[w, 0] ** 2)

        hs = h + a * hs + 1e6 * akw * fh + 1e6 / (a * a * fh)
        h = a * (h + 1e6 * akw * fh) - 1e6 / (a * fh) - alka
        a = a - h / hs

        if n > nmax:
            x = 1
            print(f'No convergence in module CCHEMS, ah0= {ah}')
        elif abs((a - al) / al) < eps:
            x = 1

            ah = 1 / a
            hco3 = sc / (1 + ak2 / ah + ah / ak1)
            co3 = hco3 * ak2 / ah
            co2 = hco3 * ah / ak1
            pco2 = co2 / alfa
            ph = -np.log10(ah / fh)

            fco3 = co3 / (co3 + hco3 + co2)

    return pco2, fco3


# In[9]:


def chemi(cm, sst, pa0):
    # Define the constants
    sb = 409.07  # Borate
    ssi = 46.5   # Silicate
    sp = 1.43    # Phosphate
    alk = 2333.0 # Alkalinity
    sal = 35.0   # Salinity
    ah = 1.e-8   # Ah
    
    cchems_result = cchems(cm, sb, ssi, sp, alk, sst, sal, ah)
   # print(f"cchems result: {cchems_result}")  # Debugging output
    
    # Extract the desired value from the result tuple (e.g., the first element)
    pco2excess = cchems_result[0] - pa0  # Assuming the first value is what you want   
    
  #  print(f"chemi function called with cm={cm}, pco2excess={pco2excess}")
    return pco2excess


# In[10]:


import numpy as np
from scipy.interpolate import interp1d

def RHS(y, time, inputdata, ntempfb, pa0, cm0, cm013, pa013, cm014, pa014, gtc2ppm, nbuff, nfoss, ndifffb, prodoc, rs, R14s, h1, h2, hm, cbio0141, cbio0142, cbio0143, fossfac, akba1, akba2, akba3, akab1, akab2, akab3, sst0, sal, cbio0131, cbio0132, cbio0133, cbio01, cbio02, cbio03, akma, rmbio, rmbio14):    

    """
    Right-hand side function for the ODE system.
    """
    # Initialize array of ODEs
    a = np.zeros(49 * 3)
# % 1: atmosphere / 0 in python
# % 2: mixed layer / 1 in python
# % 3-39: thermocline / 2-38 in python
# % 40-44: deep ocean / 39-43 in python
# % 45-47: biosphere boxes / 44-46 in python
# % 48-49: unused / 47-48 in python
# % First 49 correspond to carbon, next 49 to 13C and last 49 to 14C


    # Set temperature
    if ntempfb == 1:  # if variable SST
        if time < timespan['sst'][0]:
            sstt = inputdata['sstdata'][0]
        elif timespan['sst'][0] <= time <= timespan['sst'][1]:
            sstt = interp1d(inputdata['ssttime'], inputdata['sstdata'], kind='linear')(time)
        else:
            sstt = inputdata['sstdata'][-1]
        ssts = sstt + sst0
    else:  # if constant SST
        ssts = sst0

    # Set carbonate chemistry
    if nbuff == 0:  # if constant Revelle factor
        pm = pa0 * (1. + xi * y[1] / cm0)
    else:  # if chemistry calculated explicitly
        sb = 409.07  # borate
        ssi = 46.5  # silicate
        sp = 1.43  # phosphate
        alk = 2333.  # alkalinity
        ah = 1.e-8  # ah
        pm, fco3 = cchems_co3out(cm0 + y[1], sb, ssi, sp, alk, ssts, sal, ah)

    # Calculate air-sea exchange fractionation factors
    eps_k = -0.86
    eps_aq = 0.0049 * ssts - 1.31  # in deg C
    eps_DIC = 0.014 * ssts * fco3 - 0.105 * ssts + 10.53
    eps_ao = eps_k + eps_aq
    eps_oa = eps_k + eps_aq - eps_DIC
    alfaam = eps_ao / 1000 + 1
    alfama = eps_oa / 1000 + 1
    alfaam14 = (alfaam - 1) * 2 + 1
    alfama14 = (alfama - 1) * 2 + 1

    # Calculate biospheric ratios
    rbio1 = (cbio0131 + y[44 + (2 - 1) * 49]) / (cbio01 + y[44])
    rbio2 = (cbio0132 + y[45 + (2 - 1) * 49]) / (cbio02 + y[45])
    rbio3 = (cbio0133 + y[46 + (2 - 1) * 49]) / (cbio03 + y[46])

    rbio141 = (cbio0141 + y[44 + (3 - 1) * 49]) / (cbio01 + y[44])
    rbio142 = (cbio0142 + y[45 + (3 - 1) * 49]) / (cbio02 + y[45])
    rbio143 = (cbio0143 + y[46 + (3 - 1) * 49]) / (cbio03 + y[46])

    
    # Set fossil fuel emissions
    if nfoss == 0:  # if exponential emissions
        amu = 1. / 34.
        p0 = 5.2
        q = fossfac * p0 * np.exp(amu * (time - 1980.)) * gtc2ppm
    elif nfoss == 1:  # if reported emissions
        if time < timespan['prod'][0]:
            q = fossfac * inputdata['proddata'][0] * gtc2ppm
        elif timespan['prod'][0] <= time <= timespan['prod'][1]:
            q = fossfac * interp1d(inputdata['prodtime'], inputdata['proddata'], kind='linear')(time) * gtc2ppm
        else:
            q = fossfac * inputdata['proddata'][-1] * gtc2ppm

    # Set BECCS
    if time < timespan['beccs'][0]:
        beccs = inputdata['beccsdata'][0] * gtc2ppm
    elif timespan['beccs'][0] <= time <= timespan['beccs'][1]:
        beccs = interp1d(inputdata['beccstime'], inputdata['beccsdata'], kind='linear')(time) * gtc2ppm
    else:
        beccs = inputdata['beccsdata'][-1] * gtc2ppm

    # Set land use emissions
    if time < timespan['prodbio'][0]:
        qbio = inputdata['prodbiodata'][0] * gtc2ppm
    elif timespan['prodbio'][0] <= time <= timespan['prodbio'][1]:
        qbio = interp1d(inputdata['prodbiotime'], inputdata['prodbiodata'], kind='linear')(time) * gtc2ppm
    else:
        qbio = inputdata['prodbiodata'][-1] * gtc2ppm

    # Attribute fraction of reported ff emissions to land use emissions
    if ndeconv == 0:
        qbio = qbio + (1 - fossfac) * q

    # Set marine biosphere change
    if prodoc == 1:
        qmbio = interp1d(inputdata['prodoctime'], inputdata['prodocdata'], kind='linear')(time)
    else:
        qmbio = 0

    # Set air-land exchange fractionation factors
    if nepsab == 1:  # if variable fractionation
        if time < timespan['eps'][0]:
            alfaab_c4_herb = 1. + inputdata['epsdata_c4_herb'][0] / 1000.
            alfaab14_c4_herb = 1. + 2 * inputdata['epsdata_c4_herb'][0] / 1000.
            alfaab_c3_herb = 1. + inputdata['epsdata_c3_herb'][0] / 1000.
            alfaab14_c3_herb = 1. + 2 * inputdata['epsdata_c3_herb'][0] / 1000.
            alfaab_c3_woody = 1. + inputdata['epsdata_c3_woody'][0] / 1000.
            alfaab14_c3_woody = 1. + 2 * inputdata['epsdata_c3_woody'][0] / 1000.

        elif timespan['eps'][0] <= time <= timespan['eps'][1]:
            alfaab_c4_herb = 1. + interp1d(inputdata['epstime_c4_herb'], inputdata['epsdata_c4_herb'], kind='linear')(time) / 1000.
            alfaab14_c4_herb = 1. + 2 * interp1d(inputdata['epstime_c4_herb'], inputdata['epsdata_c4_herb'], kind='linear')(time) / 1000.
            alfaab_c3_herb = 1. + interp1d(inputdata['epstime_c3_herb'], inputdata['epsdata_c3_herb'], kind='linear')(time) / 1000.
            alfaab14_c3_herb = 1. + 2 * interp1d(inputdata['epstime_c3_herb'], inputdata['epsdata_c3_herb'], kind='linear')(time) / 1000.
            alfaab_c3_woody = 1. + interp1d(inputdata['epstime_c3_woody'], inputdata['epsdata_c3_woody'], kind='linear')(time) / 1000.
            alfaab14_c3_woody = 1. + 2 * interp1d(inputdata['epstime_c3_woody'], inputdata['epsdata_c3_woody'], kind='linear')(time) / 1000.

        else:
            alfaab_c4_herb = 1. + inputdata['epsdata_c4_herb'][-1] / 1000.
            alfaab14_c4_herb = 1. + 2 * inputdata['epsdata_c4_herb'][-1] / 1000.
            alfaab_c3_herb = 1. + inputdata['epsdata_c3_herb'][-1] / 1000.
            alfaab14_c3_herb = 1. + 2 * inputdata['epsdata_c3_herb'][-1] / 1000.
            alfaab_c3_woody = 1. + inputdata['epsdata_c3_woody'][-1] / 1000.
            alfaab14_c3_woody = 1. + 2 * inputdata['epsdata_c3_woody'][-1] / 1000.

    else:  # if constant fractionation
        alfaab_c4_herb = 1. + epsab_c4_herb / 1000.
        alfaab14_c4_herb = 1. + 2 * epsab_c4_herb / 1000.
        alfaab_c3_herb = 1. + epsab_c3_herb / 1000.
        alfaab14_c3_herb = 1. + 2 * epsab_c3_herb / 1000.
        alfaab_c3_woody = 1. + epsab_c3_woody / 1000.
        alfaab14_c3_woody = 1. + 2 * epsab_c3_woody / 1000.

        
    # Set C4 and C3 fraction
    if fractionc4mode == 1:  # if variable fraction of C4 plants
        if time < timespan['fracc4'][0]:
            fracc4 = inputdata['fracc4data'][0]
            fracc3 = inputdata['fracc3data'][0]
        elif timespan['fracc4'][0] <= time <= timespan['fracc4'][1]:
            fracc4 = interp1d(inputdata['fracc4time'], inputdata['fracc4data'], kind='linear')(time) 
            fracc3 = interp1d(inputdata['fracc3time'], inputdata['fracc3data'], kind='linear')(time) 

        else:
            fracc4 = inputdata['fracc4data'][-1]
            fracc3 = inputdata['fracc3data'][-1]

    else:  # if constant fraction of C4 plants
        fracc4 = 1
        fracc3 = 1
        
    # Calculate derivatives by linear interpolation with 1-year window
    dt = 0.5

    # Read observed atmospheric CO2 and derivatives
    if time < timespan['CO2'][0] + dt:
        xpa = 0
        dxpadt = 0
    elif timespan['CO2'][0] + dt <= time <= timespan['CO2'][1] - dt:
        xpa = interp1d(inputdata['CO2time'], inputdata['CO2data'], kind='linear')(time) - pa0
        xpa1 = interp1d(inputdata['CO2time'], inputdata['CO2data'], kind='linear')(time - dt) - pa0
        xpa2 = interp1d(inputdata['CO2time'], inputdata['CO2data'], kind='linear')(time + dt) - pa0
        dxpadt = (xpa2 - xpa1) / (2 * dt)
    else:
        xpa = inputdata['CO2data'][-1] - pa0
        dxpadt = 0

    # Read observed atmospheric d13C and derivative
    if time < timespan['CO2'][0] + dt:
        xda = 0
        ddadt = 0
    elif timespan['CO2'][0] + dt <= time < timespan['C13'][0] + dt:
        xda = pa013 / pa0 * (xpa + pa0) - pa013
        xda1 = pa013 / pa0 * (xpa1 + pa0) - pa013
        xda2 = pa013 / pa0 * (xpa2 + pa0) - pa013
        ddadt = (xda2 - xda1) / (2 * dt)
    elif timespan['C13'][0] + dt <= time <= timespan['CO2'][1] - dt:

# Enable extrapolation by setting bounds_error=False and fill_value to a default value
        interp_func = interp1d(inputdata['C13time'], inputdata['C13data'], kind='linear', bounds_error=False, fill_value="extrapolate")
# Now you can call it with time + dt, even if it's outside the bounds
        xda = (xpa + pa0) * rs * (1 + interp_func(time) / 1000) - pa013
        xda1 = (xpa1 + pa0) * rs * (1 + interp_func(time - dt) / 1000) - pa013
        xda2 = (xpa2 + pa0) * rs * (1 + interp_func(time + dt) / 1000) - pa013
        
        ddadt = (xda2 - xda1) / (2 * dt)
    else:
        xda = (xpa + pa0) * rs * (1 + inputdata['C13data'][-1] / 1000) - pa013
        ddadt = 0

    # Read observed atmospheric D14C and derivative
    if time < timespan['C14'][0] + dt:
        xda14 = 0
        ddadt14 = 0
    elif timespan['C14'][0] + dt <= time < timespan['CO2'][0] + dt:
        xda14 = (((interp1d(inputdata['C14time'], inputdata['C14data'], kind='linear')(time) + 2 * ((pa013 / pa0 / rs - 1) * 1000 + 25)) / (1 - 2e-3 * ((pa013 / pa0 / rs - 1) * 1000 + 25))) / 1000 + 1) * pa0 - pa014
        ddadt14 = ((((interp1d(inputdata['C14time'], inputdata['C14data'], kind='linear')(time + dt) + 2 * ((pa013 / pa0 / rs - 1) * 1000 + 25)) / (1 - 2e-3 * ((pa013 / pa0 / rs - 1) * 1000 + 25))) / 1000 + 1) * pa0 - (((interp1d(inputdata['C14time'], inputdata['C14data'], kind='linear')(time - dt) + 2 * ((pa013 / pa0 / rs - 1) * 1000 + 25)) / (1 - 2e-3 * ((pa013 / pa0 / rs - 1) * 1000 + 25))) / 1000 + 1) * pa0) / (2 * dt)
    elif timespan['CO2'][0] + dt <= time <= timespan['C14'][1] - dt:

# Enable extrapolation by setting bounds_error=False and fill_value to a default value
        interp_func = interp1d(inputdata['C14time'], inputdata['C14data'], kind='linear', bounds_error=False, fill_value="extrapolate")
# Now you can call it with time + dt, even if it's outside the bounds
        xda14 = (((interp_func(time) + 2 * (((xda + pa013) / (xpa + pa0) / rs - 1) * 1000 + 25)) / (1 - 2e-3 * (((xda + pa013) / (xpa + pa0) / rs - 1) * 1000 + 25))) / 1000 + 1) * (xpa + pa0) - pa014
        xda141 = (((interp_func(time - dt) + 2 * (((xda1 + pa013) / (xpa1 + pa0) / rs - 1) * 1000 + 25)) / (1 - 2e-3 * (((xda1 + pa013) / (xpa1 + pa0) / rs - 1) * 1000 + 25))) / 1000 + 1) * (xpa1 + pa0) - pa014
        xda142 = (((interp_func(time + dt) + 2 * (((xda2 + pa013) / (xpa2 + pa0) / rs - 1) * 1000 + 25)) / (1 - 2e-3 * (((xda2 + pa013) / (xpa2 + pa0) / rs - 1) * 1000 + 25))) / 1000 + 1) * (xpa2 + pa0) - pa014

        ddadt14 = (xda142 - xda141) / (2 * dt)
    else:
        xda14 = (inputdata['C14data'][-1] + 2 * (((xda + pa013) / (xpa + pa0) / rs - 1) * 1000 + 25)) / (1 - 2e-3 * (((xda + pa013) / (xpa + pa0) / rs - 1) * 1000 + 25)) - pa014
        ddadt14 = 0

    # Read 14C atom data
    if C14prog == 1:
        if time < timespan['ex14'][0] + dt:
            ex14c = inputdata['ex14data'][0]
            dex14cdt = 0
        elif timespan['ex14'][0] + dt <= time <= timespan['ex14'][1] - dt:
            ex14c = interp1d(inputdata['ex14time'], inputdata['ex14data'], kind='linear')(time)
            dex14cdt = (interp1d(inputdata['ex14time'], inputdata['ex14data'], kind='linear')(time + dt) - interp1d(inputdata['ex14time'], inputdata['ex14data'], kind='linear')(time - dt)) / (2 * dt)
        else:
            ex14c = inputdata['ex14data'] #[-1]
            dex14cdt = 0

    # Set 13C/12C ratio in fossil fuel carbon
    if nfoss == 0:
        qf = -25
        rfosstime = rs * (1 + qf / 1000.)
    elif nfoss == 1:
        if time < timespan['C13foss'][0]:
            rfosstime = rs * (1 + inputdata['C13fossdata'][-1] / 1000.)
        elif timespan['C13foss'][0] <= time <= timespan['C13foss'][1]:
            rfosstime = rs * (1 + interp1d(inputdata['C13fosstime'], inputdata['C13fossdata'], kind='linear')(time) / 1000.)
        else:
            rfosstime = rs * (1 + inputdata['C13fossdata'][-1] / 1000.)

    # Constants of discretization scheme in ocean
    if ndifffb == 1:
        if time < timespan['diff'][0]:
            aks = inputdata['diffdata'][0]
        elif timespan['diff'][0] <= time <= timespan['diff'][1]:
            aks = interp1d(inputdata['difftime'], inputdata['diffdata'], kind='linear')(time)
        else:
            aks = inputdata['diffdata'][-1]
    else:
        aks = ak

    ak1 = aks / (h1 * h1)
    ak2 = aks / (h2 * h2)
    akv = 2 * aks / (h1 * (h1 + h2))
    akn = 2 * aks / (h2 * (h1 + h2))
    akmd = 2 * aks / (hm * (hm + h1))
    akdm = 2 * aks / (h1 * (hm + h1))

    
    # Model configuration
    frac_box1 =1
    frac_box2 =1
    frac_box3 =1
#     frac_box1 = fracc4
#     frac_box2 = fracc3
#     frac_box3 = fracc3
    
    alfaab_box1 = alfaab_c4_herb
    alfaab_box2 = alfaab_c3_herb
    alfaab_box3 = alfaab_c3_woody

    alfaab14_box1 = alfaab14_c4_herb
    alfaab14_box2 = alfaab14_c3_herb
    alfaab14_box3 = alfaab14_c3_woody

    epsi_box1 = 0  # no CO2 fertilisation for C4 plants 
    
    # Now program the equations
    if ndeconv == 0:  # Forward Model run

        # Atmosphere
        # 2.12⋅ d13pa/dt = 2.12⋅kam (αoa⋅rm⋅pm – αao⋅ra⋅pa ) 
        #                   + kba1⋅13B1 + kba3⋅13B2 + kba2⋅13B3
        #                   − αab⋅ra NPP + rfos⋅Qfos + (13B2/B2)⋅QLU + αab⋅ra⋅Qres        

        # kam is air-sea exchange rate
        # αoa = 1 + ϵoa and αao = 1+ ϵao are fractionation factors associated with air-sea exchange
        # rm and ra are the 13C/C mole ratios in the ocean mixed layer and atmosphere
        # pm and pa are concentration of CO2 in ppm in the ocean mixed layer and atmosphere
        # kba1, kba2, kba3 are the turnover rates of box 1, 2 and 3, respectively
        # αab = 1 − Δ is the fractionation factor associated with land photosynthesis
        # 13B1, 13B2 and 13B3 are the 13C amounts in the three land boxes
        # NPP = NPP0 (1 + epsi*(pa−pa0)/pa0)
        # Preindustrial values: NPP0 = 46.5 Pg yr -1, pa0 = 278 ppm
        # rfos and Qfos are the isotopic ratio and amount of fossil-fuel combustion
        # B2 is the total amount of carbon in the second land box
        # QLU is the residual land flux from land use
        # Qres is the residual land flux (positive into the air) required to match the observed atmospheric CO2 trend
        
        # Atmospheric CO2
        a[0] = akam * (pm - (pa0 + y[0])) + q + beccs + qbio - akab1 * (pa0 + epsi * y[0]) + akba1 * (y[44] + cbio01) * gtc2ppm - akab2 * (pa0 + epsi * y[0]) + akba2 * (y[45] + cbio02) * gtc2ppm - akab3 * (pa0 + epsi * y[0]) + akba3 * (y[46] + cbio03) * gtc2ppm
 
        # Atmospheric 13C
        a[0+(2-1)*49] = akam * (alfama * ((y[1+(2-1)*49] + cm013) / (y[1] + cm0)) * pm - alfaam * (y[0+(2-1)*49] + pa013)) + (akba1 * (y[44+(2-1)*49] + cbio0131) * gtc2ppm - alfaab_box1 * akab1 * (pa013 + y[0+(2-1)*49]) * (pa0 + epsi * y[0]) / (pa0 + y[0])) + (akba2 * (y[45+(2-1)*49] + cbio0132) * gtc2ppm - alfaab_box2 * akab2 * (pa013 + y[0+(2-1)*49]) * (pa0 + epsi * y[0]) / (pa0 + y[0])) + (akba3 * (y[46+(2-1)*49] + cbio0133) * gtc2ppm - alfaab_box3 * akab3 * (pa013 + y[0+(2-1)*49]) * (pa0 + epsi * y[0]) / (pa0 + y[0])) + (q * rfosstime) + beccs * (alfaab_box2 * (pa013 + y[0+(2-1)*49]) / (pa0 + y[0])) + (qbio >= 0) * (qbio * rbio2) + (qbio < 0) * (alfaab_box2 * qbio * (pa013 + y[0+(2-1)*49]) / (pa0 + y[0]))

        # Atmospheric 14C
        if C14prog == 0:
            a[0+(3-1)*49] = ddadt14
        elif C14prog == 1:
            a[0+(3-1)*49] = akam * (alfama14 * ((y[1+(3-1)*49] + cm014) / (y[1] + cm0)) * pm - alfaam14 * (y[0+(3-1)*49] + pa014)) + (akba1 * (y[44+(3-1)*49] + cbio0141) * gtc2ppm - alfaab14_box1 * akab1 * (y[0+(3-1)*49] + pa014) * (pa0 + epsi * y[0]) / (pa0 + y[0])) + (akba2 * (y[45+(3-1)*49] + cbio0142) * gtc2ppm - alfaab14_box2 * akab2 * (y[0+(3-1)*49] + pa014) * (pa0 + epsi * y[0]) / (pa0 + y[0])) + (akba3 * (y[46+(3-1)*49] + cbio0143) * gtc2ppm - alfaab14_box3 * akab3 * (y[0+(3-1)*49] + pa014) * (pa0 + epsi * y[0]) / (pa0 + y[0])) - (y[0+(3-1)*49] + pa014) / 8267. + (qbio >= 0) * (qbio * rbio142) + (qbio < 0) * (alfaab14_box2 * qbio * (y[0+(3-1)*49] + pa014) / (pa0 + y[0])) + beccs * (alfaab14_box2 * (y[0+(3-1)*49] + pa014) / (pa0 + y[0])) + ex14c * gtc2ppm / 1e15 * 12 / R14s / 6.02e-3 / 8267 + 0.30852866
        
        # Biosphere
        # Land Biosphere Total Carbon - Box 1
        a[44] = akab1 * (pa0 + epsi * y[0]) / gtc2ppm - akba1 * (y[44] + cbio01)
        # Land Biosphere Total Carbon - Box 2 - Land use fluxes go here
        a[45] = akab2 * (pa0 + epsi * y[0]) / gtc2ppm - akba2 * (y[45] + cbio02) - qbio / gtc2ppm
        # Land Biosphere Total Carbon - Box 3
        a[46] = akab3 * (pa0 + epsi * y[0]) / gtc2ppm - akba3 * (y[46] + cbio03)
         
        # Land Biosphere 13C - box 1
        a[44+(2-1)*49] = -(akba1 * (y[44+(2-1)*49] + cbio0131) - alfaab_box1 * akab1 * (pa013 + y[0+(2-1)*49]) * (pa0 + epsi * y[0]) / (pa0 + y[0]) / gtc2ppm)
        # Land Biosphere 13C - box 2 - land use fluxes go here
        a[45+(2-1)*49] = -(akba2 * (y[45+(2-1)*49] + cbio0132) - alfaab_box2 * akab2 * (pa013 + y[0+(2-1)*49]) * (pa0 + epsi * y[0]) / (pa0 + y[0]) / gtc2ppm) - (qbio >= 0) * (qbio / gtc2ppm * rbio2) - (qbio < 0) * (alfaab_box2 * qbio / gtc2ppm * (pa013 + y[0+(2-1)*49]) / (pa0 + y[0]))
        # Land Biosphere 13C - box 3
        a[46+(2-1)*49] = -(akba3 * (y[46+(2-1)*49] + cbio0133) - alfaab_box3 * akab3 * (pa013 + y[0+(2-1)*49]) * (pa0 + epsi * y[0]) / (pa0 + y[0]) / gtc2ppm)

        # Land Biosphere 14C - box 1
        a[44+(3-1)*49] = -(akba1 * (y[44+(3-1)*49]+ cbio0141) - alfaab14_box1 * akab1 * (y[0+(3-1)*49] + pa014) * (pa0 + epsi * y[0]) / (pa0 + y[0]) / gtc2ppm) - (y[44+(3-1)*49] + cbio0141) / 8267.
        # Land Biosphere 14C - box 2
        a[45+(3-1)*49] = -(akba2 * (y[45+(3-1)*49] + cbio0142) - alfaab14_box2 * akab2 * (y[0+(3-1)*49] + pa014) * (pa0 + epsi * y[0]) / (pa0 + y[0]) / gtc2ppm) - (qbio >= 0) * (qbio / gtc2ppm * rbio142) - (qbio < 0) * (alfaab14_box2 * qbio * (y[45+(2-1)*49] + pa014) / (pa0 + y[0])) - (y[45+(3-1)*49] + cbio0142) / 8267.
        # Land Biosphere 14C - box 3
        a[46+(3-1)*49] = -(akba3 * (y[46+(3-1)*49] + cbio0143) - alfaab14_box3 * akab3 * (y[0+(3-1)*49] + pa014) * (pa0 + epsi * y[0]) / (pa0 + y[0]) / gtc2ppm) - (y[46+(3-1)*49] + cbio0143) / 8267.

        # Mixed Layer
        # Mixed Layer total carbon
        a[1] = -akma * (pm - (pa0 + y[0])) + akmd * (-3 * y[1] + 4 * y[2] - y[3]) + akma / akam * qmbio * gtc2ppm

        # Mixed Layer 13C
        a[1+(2-1)*49] = -akma * (alfama * ((y[1+(2-1)*49] + cm013) / (y[1] + cm0)) * pm - alfaam * (y[0+(2-1)*49] + pa013)) + akmd * (-3 * y[1+(2-1)*49] + 4 * y[2+(2-1)*49] - y[3+(2-1)*49]) + akma / akam * qmbio * gtc2ppm * rmbio

        # Mixed Layer 14C
        a[1+(3-1)*49] = -akma * (alfama14 * ((y[1+(3-1)*49] + cm014) / (y[1] + cm0)) * pm - alfaam14 * (y[0+(3-1)*49] + pa014)) + akmd * (-3 * y[1+(3-1)*49] + 4 * y[2+(3-1)*49] - y[3+(3-1)*49]) + akma / akam * qmbio * gtc2ppm * rmbio14 - (y[1+(3-1)*49] + cm014) / 8267.

    elif ndeconv == 1:  # Single Deconvolution

        # Atmospheric CO2
        a[0] = dxpadt
        sdbioflux = (akam * (pm - (pa0 + y[0])) + q + beccs + qbio - akab1 * (pa0 + epsi_box1 * y[0])*frac_box1 + akba1 * (y[44] + cbio01) * gtc2ppm *frac_box1 - akab2 * (pa0 + epsi * y[0])*frac_box2 + akba2 * (y[45] + cbio02) * gtc2ppm*frac_box2 - akab3 * (pa0 + epsi * y[0])*frac_box3 + akba3 * (y[46] + cbio03) * gtc2ppm*frac_box3) - dxpadt

        # Land Biosphere Total Carbon - Box 1
        a[44] = (akab1 * (pa0 + epsi_box1 * y[0]) / gtc2ppm - akba1 * (y[44] + cbio01))*frac_box1

        # Land Biosphere Total Carbon - Box 2 - Land use and residual fluxes go here
        a[45] = (akab2 * (pa0 + epsi * y[0]) / gtc2ppm - akba2 * (y[45] + cbio02))*frac_box2 - qbio / gtc2ppm + sdbioflux / gtc2ppm

        # Land Biosphere Total Carbon - Box 3
        a[46] = (akab3 * (pa0 + epsi * y[0]) / gtc2ppm - akba3 * (y[46] + cbio03))*frac_box3

        if sdbioflux > 0:  # flux is into biosphere
            sdbio13flux = sdbioflux * alfaab_box2 * (pa013 + y[0+(2-1)*49]) / (pa0 + y[0])
            sdbio14flux = sdbioflux * alfaab14_box2 * (pa014 + y[0+(3-1)*49]) / (pa0 + y[0])
        else:  # flux is out of biosphere
            sdbio13flux = sdbioflux * rbio2
            sdbio14flux = sdbioflux * rbio142

        if C13prog == 0:
            a[0+(2-1)*49] = ddadt
        else:
            # Atmospheric d13C
            a[0+(2-1)*49] = akam * (alfama * ((y[1+(2-1)*49] + cm013) / (y[1] + cm0)) * pm - alfaam * (y[0+(2-1)*49] + pa013)) + (akba1 * (y[44+(2-1)*49] + cbio0131) * gtc2ppm - alfaab_box1 * akab1 * (pa013 + y[0+(2-1)*49]) * (pa0 + epsi_box1 * y[0]) / (pa0 + y[0]))*frac_box1 + (akba2 * (y[45+(2-1)*49] + cbio0132) * gtc2ppm - alfaab_box2 * akab2 * (pa013 + y[0+(2-1)*49]) * (pa0 + epsi * y[0]) / (pa0 + y[0]))*frac_box2 + (akba3 * (y[46+(2-1)*49] + cbio0133) * gtc2ppm - alfaab_box3 * akab3 * (pa013 + y[0+(2-1)*49])* (pa0 + epsi * y[0]) / (pa0 + y[0]))*frac_box3 - sdbio13flux + (q * rfosstime) + beccs * (alfaab_box2 * (pa013 + y[0+(2-1)*49]) / (pa0 + y[0]))*frac_box2 + (qbio >= 0) * (qbio * rbio2) + (qbio < 0) * (alfaab_box2 * qbio * (pa013 + y[0+(2-1)*49]) / (pa0 + y[0]))*frac_box2

        # Atmospheric 14C
        if C14prog == 0:
            a[0+(3-1)*49] = ddadt14
        elif C14prog == 1:
            a[0+(3-1)*49] = akam * (alfama14 * ((y[1+(3-1)*49] + cm014) / (y[1] + cm0)) * pm - alfaam14 * (y[0+(3-1)*49] + pa014)) + (akba1 * (y[44+(3-1)*49] + cbio0141) * gtc2ppm - alfaab14_box1 * akab1 * (y[0+(3-1)*49] + pa014) * (pa0 + epsi_box1 * y[0]) / (pa0 + y[0]))*frac_box1 + (akba2 * (y[45+(3-1)*49] + cbio0142) * gtc2ppm - alfaab14_box2 * akab2 * (y[0+(3-1)*49] + pa014) * (pa0 + epsi * y[0]) / (pa0 + y[0]))*frac_box2 + (akba3 * (y[46+(3-1)*49] + cbio0143) * gtc2ppm - alfaab14_box3 * akab3 * (y[0+(3-1)*49] + pa014) * (pa0 + epsi * y[0]) / (pa0 + y[0]))*frac_box3 - (y[0+(3-1)*49] + pa014) / 8267. - sdbio14flux + (qbio >= 0) * (qbio * rbio142) + (qbio < 0) * (alfaab14_box2 * qbio * (y[0+(3-1)*49] + pa014) / (pa0 + y[0])) + beccs * (alfaab14_box2 * (y[0+(3-1)*49] + pa014) / (pa0 + y[0]))*frac_box2 + ex14c * gtc2ppm / 1e15 * 12 / R14s / 6.02e-3 / 8267 + 0.30852866

        # Land Biosphere 13C - box 1
        a[44+(2-1)*49] = -(akba1 * (y[44+(2-1)*49] + cbio0131) - alfaab_box1 * akab1 * (pa013 + y[0+(2-1)*49]) * (pa0 + epsi_box1 * y[0]) / (pa0 + y[0]) / gtc2ppm)*frac_box1
        # Land Biosphere 13C - box 2 - land use and residual flux go here
        a[45+(2-1)*49] = -(akba2 * (y[45+(2-1)*49] + cbio0132) - alfaab_box2 * akab2 * (pa013 + y[0+(2-1)*49]) * (pa0 + epsi * y[0]) / (pa0 + y[0]) / gtc2ppm)*frac_box2 + sdbio13flux / gtc2ppm - (qbio >= 0) * (qbio / gtc2ppm * rbio2) - (qbio < 0) * (alfaab_box2 * qbio / gtc2ppm * (pa013 + y[0+(2-1)*49]) / (pa0 + y[0]))*frac_box2
        # Land Biosphere 13C - box 3
        a[46+(2-1)*49] = -(akba3 * (y[46+(2-1)*49] + cbio0133) - alfaab_box3 * akab3 * (pa013 + y[0+(2-1)*49]) * (pa0 + epsi * y[0]) / (pa0 + y[0]) / gtc2ppm)*frac_box3

        # Land Biosphere 14C - box 1
        a[44+(3-1)*49] = -(akba1 * (y[44+(3-1)*49] + cbio0141) - alfaab14_box1 * akab1 * (y[0+(3-1)*49] + pa014) * (pa0 + epsi_box1 * y[0]) / (pa0 + y[0]) / gtc2ppm)*frac_box1 - (y[44+(3-1)*49] + cbio0141) / 8267.
        # Land Biosphere 14C - box 2
        a[45+(3-1)*49] = -(akba2 * (y[45+(3-1)*49] + cbio0142) - alfaab14_box2 * akab2 * (y[0+(3-1)*49] + pa014) * (pa0 + epsi * y[0]) / (pa0 + y[0]) / gtc2ppm)*frac_box2 + sdbio14flux / gtc2ppm - (qbio >= 0) * (qbio / gtc2ppm * rbio142) - (qbio < 0) * (alfaab14_box2 * qbio * (y[0+(3-1)*49] + pa014) / (pa0 + y[0]))*frac_box2 - (y[45+(3-1)*49] + cbio0142) / 8267.
        # Land Biosphere 14C - box 3
        a[46+(3-1)*49] = -(akba3 * (y[46+(3-1)*49] + cbio0143) - alfaab14_box3 * akab3 * (y[0+(3-1)*49] + pa014) * (pa0 + epsi * y[0]) / (pa0 + y[0]) / gtc2ppm)*frac_box3 - (y[46+(3-1)*49] + cbio0143) / 8267.

        # Mixed Layer total carbon
        a[1] = -akma * (pm - (pa0 + y[0])) + akmd * (-3 * y[1] + 4 * y[2] - y[3]) + akma / akam * qmbio * gtc2ppm

        # Mixed Layer 13C
        a[1+(2-1)*49] = -akma * (alfama * ((y[1+(2-1)*49] + cm013) / (y[1] + cm0)) * pm - alfaam * (y[0+(2-1)*49] + pa013)) + akmd * (-3 * y[1+(2-1)*49] + 4 * y[2+(2-1)*49] - y[3+(2-1)*49]) + akma / akam * qmbio * gtc2ppm * rmbio

        # Mixed Layer 14C
        a[1+(3-1)*49] = -akma * (alfama14 * ((y[1+(3-1)*49] + cm014) / (y[1] + cm0)) * pm - alfaam14 * (y[0+(3-1)*49] + pa014)) + akmd * (-3 * y[1+(3-1)*49] + 4 * y[2+(3-1)*49] - y[3+(3-1)*49]) + akma / akam * qmbio * gtc2ppm * rmbio14 - (y[1+(3-1)*49] + cm014) / 8267.

    # First thermocline box
    a[2] = -akdm * (-3 * y[1] + 4 * y[2] - y[3]) + ak1 * (y[3] - y[2])
    a[2+(2-1)*49] = -akdm * (-3 * y[1+(2-1)*49] + 4 * y[2+(2-1)*49] - y[3+(2-1)*49]) + ak1 * (y[3+(2-1)*49] - y[2+(2-1)*49])
    a[2+(3-1)*49] = -akdm * (-3 * y[1+(3-1)*49] + 4 * y[2+(3-1)*49] - y[3+(3-1)*49]) + ak1 * (y[3+(3-1)*49] - y[2+(3-1)*49]) - (y[2+(3-1)*49] + cm014) / 8267.

    # Thermocline
    for i in range(3, 38):
        a[i] = ak1 * (y[i - 1] - 2 * y[i] + y[i + 1])  # Total Carbon
        a[i + 0+(2-1)*49] = ak1 * (y[i - 1 + 0+(2-1)*49] - 2 * y[i + 0+(2-1)*49] + y[i + 1 + 0+(2-1)*49])  # 13C
        a[i + 0+(3-1)*49] = ak1 * (y[i - 1 + 0+(3-1)*49] - 2 * y[i + 0+(3-1)*49] + y[i + 1 + 0+(3-1)*49]) - (y[i + 0+(3-1)*49] + cm014) / 8267.  # 14C

    a[38] = ak1 * (y[37] - y[38]) + akv * (y[39] - y[38])
    a[38+(2-1)*49] = ak1 * (y[37+(2-1)*49] - y[38+(2-1)*49]) + akv * (y[39+(2-1)*49] - y[38+(2-1)*49])
    a[38+(3-1)*49] = ak1 * (y[37+(3-1)*49] - y[38+(3-1)*49]) + akv * (y[39+(3-1)*49] - y[38+(3-1)*49]) - (y[38+(3-1)*49] + cm014) / 8267.

    # Deep Sea
    a[39] = akn * (y[38] - y[39]) + ak2 * (y[40] - y[39])  # Total Carbon
    a[39+(2-1)*49] = akn * (y[38+(2-1)*49] - y[39+(2-1)*49]) + ak2 * (y[40+(2-1)*49] - y[39+(2-1)*49])  # 13C
    a[39+(3-1)*49] = akn * (y[38+(3-1)*49] - y[39+(3-1)*49]) + ak2 * (y[40+(3-1)*49] - y[39+(3-1)*49]) - (y[39+(3-1)*49] + cm014) / 8267.  # 14C

    for i in range(40, 43):
        a[i] = ak2 * (y[i - 1] - 2 * y[i] + y[i + 1])  # Total Carbon
        a[i + 0+(2-1)*49] = ak2 * (y[i - 1 + 0+(2-1)*49] - 2 * y[i + 0+(2-1)*49] + y[i + 1 + 0+(2-1)*49])  # 13C
        a[i + 0+(3-1)*49] = ak2 * (y[i - 1 + 0+(3-1)*49] - 2 * y[i + 0+(3-1)*49] + y[i + 1 + 0+(3-1)*49]) - (y[i + 0+(3-1)*49] + cm014) / 8267.  # 14C

    a[43] = ak2 * (y[42] - y[43])  # Total Carbon
    a[43+(2-1)*49] = ak2 * (y[42+(2-1)*49] - y[43+(2-1)*49])  # 13C
    a[43+(3-1)*49] = ak2 * (y[42+(3-1)*49] - y[43+(3-1)*49]) - (y[43+(3-1)*49] + cm014) / 8267.  # 14C

    return a


# In[11]:


from scipy.optimize import root_scalar

# Run the model for each parameter set
for t in range(len(param)):
#for t in range(1):
    # Specify parameters
    ak = param[t, 0]
    akam = 1 / param[t, 1]
    epsi = param[t, 2]  # beta, fertilisation factor
    akab1 = 1 / param[t, 3]
    akab2 = 1 / param[t, 4]
    akab3 = 1 / param[t, 5]
    akba1 = 1 / param[t, 6]
    akba2 = 1 / param[t, 7]
    akba3 = 1 / param[t, 8]
    
    print("Param line t:", t)

#    for rc in range(6):
    for rc in range(1):
        # Define datafiles used in different scenarios
        prodco2f = f'fossil_SSP{SSP[rc]}.txt'
        prodbiof = f'landuse_SSP{SSP[rc]}.txt'
        pco2atmf = f'co2_SSP{SSP[rc]}.txt'
        beccsf = f'beccs_SSP{SSP[rc]}.txt'

        fepsabf_c3_herb = f'epsab_lavergne_historical_C3_herbaceous.txt' ## land fractionation for C3 plants - Lavergne et al. 2025 CEE
        fepsabf_c3_woody = f'epsab_lavergne_historical_C3_woody.txt' ## land fractionation for C3 plants - Lavergne et al. 2025 CEE
        fepsabf_c4_herb = f'epsab_lavergne_historical_C4_herbaceous.txt' ## land fractionation for C4 plants - Lavergne et al. 2025 CEE

        sstdatf = f'sst_SSP{SSP[rc]}.txt'
        c13foss = 'c13foss_constantpost2013.txt'

        C14prog = 0  # diagnose atm D14C for historical period
        C13prog = 0  # diagnose atm d13C for historical period

        # new data
        fractionc4 = f'fraction_c4_lavergne_v2.txt' ## modified fraction C4 - Lavergne et al. 2025 CEE
        fractionc3 = f'fraction_c3_lavergne_v2.txt' ## modified fraction C3 - Lavergne et al. 2025 CEE

    
        # BDprecalc
        
        # Perform preliminary calculations to begin carbon cycle simulation
        # Heather Graven 2020


        # Constants and variables
        gtc2ppm = 290 / 615.6  # conversion from GtC to ppm
        aoc = 361419000e6  # square meter area of the ocean
        hm = 75.0  # mixed layer depth
        h1 = 25.0  # thermocline layers depth
        h2 = 545.8  # deep layers depth
        rs = 0.0112372  # abundance of C13 vs C12
        R14s = 1.176e-12  # abundance of C14 vs C
        sal = 35
#        epsab = -18.0  # land biospheric fractionation (permil) - negative of carbon discrimination
        epsab_c3 = -18.0  # land biospheric fractionation (permil) - negative of carbon discrimination
        epsab_c4 = -6.0  # land biospheric fractionation (permil) - negative of carbon discrimination

        epsmbio = -18.0  # ocean biospheric fractionation (permil)
        xi = 10.0  # buffer factor (ignored if nbuff=1)

        # Placeholder for arrays and variables
        y0 = np.zeros(3 * 49)

        # Constants from the script
        akma = akam / gtc2ppm / aoc * 1.e18 / 12 / hm  # units (1/yr)(mmol/m3)(1/ppm)


        # Read in atmospheric data and define initial atmospheric values / reference values
        inputdata = {}
        inputdata['CO2time'], inputdata['CO2data'] = BDreaddata(pco2atmf)
        timespan_CO2 = [inputdata['CO2time'][0], inputdata['CO2time'][-1]]

        inputdata['C13time'], inputdata['C13data'] = BDreaddata(del13atmf)
        timespan_C13 = [inputdata['C13time'][0], inputdata['C13time'][-1]]

        inputdata['C14time'], inputdata['C14data'] = BDreaddata(del14atmf)
        timespan_C14 = [inputdata['C14time'][0], inputdata['C14time'][-1]]

        pa0 = inputdata['CO2data'][0]
        da0 = inputdata['C13data'][0]
        Da140 = inputdata['C14data'][0]
        pa013 = pa0 * rs * (1. + da0 / 1000.)  # 13CO2 approximated using 13C/C rather than 13C/12C
        da014 = (Da140 + 2 * (da0 + 25)) / (1 - 2e-3 * (da0 + 25))
        pa014 = pa0 * (1. + da014 / 1000.)

        print("13CO2:", pa013)
                
        ## Read C3 and C4 fraction data
        inputdata['fracc4time'], inputdata['fracc4data'] = BDreaddata(fractionc4)
        timespan_fracc4 = [inputdata['fracc4time'][0], inputdata['fracc4time'][-1]]
        
        fracc4 = inputdata['fracc4data'] #[0]

        inputdata['fracc3time'], inputdata['fracc3data'] = BDreaddata(fractionc3)
        timespan_fracc3 = [inputdata['fracc3time'][0], inputdata['fracc3time'][-1]]
        
        fracc3 = inputdata['fracc3data'] #[0]
        
        
        # Read sst data
        sst0 = 18  # set reference value at 18C
        if ntempfb == 1:
            inputdata['ssttime'], inputdata['sstdata'] = BDreaddata(sstdatf)
            sst = sst0 + inputdata['sstdata'][0]
            timespan_sst = [inputdata['ssttime'][0], inputdata['ssttime'][-1]]
        else:
            sst = sst0

        # Find DIC (mmol/m3) where pCO2 excess is zero: atm in eq with ocean
        cm1 = 1800.
        cm2 = 2200.
        tol = 1e-9

        result = root_scalar(chemi, args=(sst, pa0), bracket=[cm1, cm2], xtol=tol)
        cm0 = result.root

        print("Optimized DIC value:", cm0)
        
        sb = 409.07  # borate
        ssi = 46.5  # silicate
        sp = 1.43  # phosphate
        alk = 2333.  # alkalinity
        ah = 1.e-8  # ah
        pm, fco3 = cchems_co3out(cm0, sb, ssi, sp, alk, sst, sal, ah)


        # Calculate initial/ref fractionation factors and C13 in ocean
        eps_k = -0.86
        eps_aq = +0.0049 * sst - 1.31  # in deg C
        eps_DIC = 0.014 * sst * fco3 - 0.105 * sst + 10.53
        eps_ao = eps_k + eps_aq
        eps_oa = eps_k + eps_aq - eps_DIC
        alfaam = eps_ao / 1000 + 1
        alfama = eps_oa / 1000 + 1
        cm013 = pa013 * (alfaam / alfama) * cm0 / pa0
        dm0 = (cm013 / cm0 / rs - 1.) * 1000.

        # Calculate initial/ref fractionation factors and C14 in ocean
        alfaam14 = (alfaam - 1) * 2 + 1
        alfama14 = (alfama - 1) * 2 + 1
        Dm014 = Da140 - 50  # use atm-50 per mil as D14C reference value for surface ocean
        dm014 = (Dm014 + 2 * (dm0 + 25)) / (1 - 2e-3 * (dm0 + 25))
        cm014 = cm0 * (1. + dm014 / 1000.)

        # Calculate fractionation factors and isotopic ratios in marine biota
        alfamb = 1. + epsmbio / 1000.
        alfamb14 = 1 + 2 * epsmbio / 1000
        rmbio = cm013 / cm0 * alfamb
        rmbio14 = cm014 / cm0 * alfamb14

        # Initial/ref biospheric carbon, C13, and C14 content
        
        cbio01 = pa0 * akab1 / (akba1 * gtc2ppm)  # initial biospheric carbon in GtC
        cbio02 = pa0 * akab2 / (akba2 * gtc2ppm)  # initial biospheric carbon in GtC
        cbio03 = pa0 * akab3 / (akba3 * gtc2ppm)  # initial biospheric carbon in GtC

        print("initial biospheric carbon C4 box 1 (GtC):", cbio01)
        print("initial biospheric carbon C3 box 2 (GtC):", cbio02)
        print("initial biospheric carbon C3 box 3 (GtC):", cbio03)

        print("land carbon discrimination (0 = constant, 1 = vary):", nepsab)
        print("fraction of C4 (0 = constant, 1 = vary):", fractionc4mode)

        if nepsab == 1:
            inputdata['epstime_c3_herb'], inputdata['epsdata_c3_herb'] = BDreaddata(fepsabf_c3_herb)
            timespan_eps_c3_herb = [inputdata['epstime_c3_herb'][0], inputdata['epstime_c3_herb'][-1]]
            inputdata['epstime_c4_herb'], inputdata['epsdata_c4_herb'] = BDreaddata(fepsabf_c4_herb)
            timespan_eps_c4_herb = [inputdata['epstime_c4_herb'][0], inputdata['epstime_c4_herb'][-1]]
            inputdata['epstime_c3_woody'], inputdata['epsdata_c3_woody'] = BDreaddata(fepsabf_c3_woody)
            timespan_eps_c3_woody = [inputdata['epstime_c3_woody'][0], inputdata['epstime_c3_woody'][-1]]
                                
            alfaab_c3_herb = 1 + inputdata['epsdata_c3_herb'][0] / 1000
            alfaab14_c3_herb = (alfaab_c3_herb - 1) * 2 + 1
            
            alfaab_c4_herb = 1 + inputdata['epsdata_c4_herb'][0] / 1000
            alfaab14_c4_herb = (alfaab_c4_herb - 1) * 2 + 1

            alfaab_c3_woody = 1 + inputdata['epsdata_c3_woody'][0] / 1000
            alfaab14_c3_woody = (alfaab_c3_woody - 1) * 2 + 1

        else:    
            alfaab_c3_herb = 1. + epsab_c3_herb / 1000.
            alfaab14_c3_herb = 1. + 2 * epsab_c3_herb / 1000.
            alfaab_c4_herb = 1. + epsab_c4_herb / 1000.
            alfaab14_c4_herb = 1. + 2 * epsab_c4_herb / 1000.
            alfaab_c3_woody = 1. + epsab_c3_woody / 1000.
            alfaab14_c3_woody = 1. + 2 * epsab_c3_woody / 1000.
                   
        cbio0131 = cbio01 * alfaab_c4_herb * pa013 / pa0  # 13C in biosphere
        cbio0132 = cbio02 * alfaab_c3_herb * pa013 / pa0  # 13C in biosphere
        cbio0133 = cbio03 * alfaab_c3_woody * pa013 / pa0  # 13C in biosphere
        cbio0141 = cbio01 * alfaab14_c4_herb * pa014 / pa0  # 14C in biosphere
        cbio0142 = cbio02 * alfaab14_c3_herb * pa014 / pa0  # 14C in biosphere
        cbio0143 = cbio03 * alfaab14_c3_woody * pa014 / pa0  # 14C in biosphere

        cbio01_all = cbio0131 + cbio0141  # 13C + 14C in box 1
        cbio02_all = cbio0132 + cbio0142  # 13C + 14C in box 2
        cbio03_all = cbio0133 + cbio0143  # 13C + 14C in box 3
        
        print("Initial biospheric carbon of C4 box 1:", cbio01_all)
        print("Initial biospheric carbon of C3 box 2:", cbio02_all)
        print("Initial biospheric carbon of C3 box 3:", cbio03_all)
        
        # Read in the rest of the data
        inputdata['prodtime'], inputdata['proddata'] = BDreaddata(prodco2f)
        timespan_prod = [inputdata['prodtime'][0], inputdata['prodtime'][-1]]

        inputdata['prodbiotime'], inputdata['prodbiodata'] = BDreaddata(prodbiof)
        timespan_prodbio = [inputdata['prodbiotime'][0], inputdata['prodbiotime'][-1]]

        inputdata['C13fosstime'], inputdata['C13fossdata'] = BDreaddata(c13foss)
        timespan_C13foss = [inputdata['C13fosstime'][0], inputdata['C13fosstime'][-1]]

        inputdata['beccstime'], inputdata['beccsdata'] = BDreaddata(beccsf)
        timespan_beccs = [inputdata['beccstime'][0], inputdata['beccstime'][-1]]

        if ndifffb == 1:
            inputdata['difftime'], inputdata['diffdata'] = BDreaddata(diffdatf)
            timespan_diff = [inputdata['difftime'][0], inputdata['difftime'][-1]]

        if prodoc == 1:
            inputdata['prodoctime'], inputdata['prodocdata'] = BDreaddata(prodocf)
            timespan_prodoc = [inputdata['prodoctime'][0], inputdata['prodoctime'][-1]]

        if C14prog == 1:
            inputdata['ex14time'], inputdata['ex14data'] = BDreaddata(ex14sourcef)
            timespan_ex14 = [inputdata['ex14time'][0], inputdata['ex14time'][-1]]

    
        # Run spinup and historical period to 1981, annual output
        time = np.arange(-10000, 1982, 1)
        y0 = np.zeros(3 * 49)  # Initial conditions (assumed to be defined)
        timespan = {
            'sst': [1900, 2025],  # Example values for SST time range
            'prod': [1900, 2025],  # Example values for fossil fuel emissions time range
            'beccs': [1900, 2025],  # Example values for BECCS time range
            'prodbio': [1900, 2025],  # Example values for land use emissions time range
            'eps': [1900, 2025],  # Example values for air-land exchange fractionation time range
            'CO2': [1900, 2025],  # Example values for atmospheric CO2 time range
            'C14': [1900, 2025],  # Example values for atmospheric D14C time range
            'C13': [1900, 2025],  # Example values for atmospheric d13C time range
            'C13foss': [1900, 2025],  # Example values for fossil fuel 13C/12C ratio time range
            'fracc4': [1900, 2025],  # Example values for fraction of C4 plants time range
            'fracc3': [1900, 2025],  # Example values for fraction of C3 plants time range
        }
        # # Pack additional arguments into a tuple
        args = (inputdata, ntempfb, pa0, cm0, cm013, pa013, cm014, pa014, gtc2ppm, nbuff, nfoss, ndifffb, prodoc, rs, R14s, h1, h2, hm, cbio0141, cbio0142, cbio0143, fossfac, akba1, akba2, akba3, akab1, akab2, akab3, sst0, sal, cbio0131, cbio0132, cbio0133, cbio01, cbio02, cbio03, akma, rmbio, rmbio14) 

        Y = odeint(RHS, y0, time, args=args)

        # Calculate steady state total atoms for production term
        atoms14 = np.zeros(47)
        atoms14[0] = (Y[-1, 0 + (3 - 1) * 49] + pa014) / gtc2ppm * 1e15 / 12 * R14s * 6.02e-3
        atoms14[1] = (Y[-1, 1 + (3 - 1) * 49] + cm014) * R14s * hm * aoc / 1000 * 6.02e-3
        for i in range(2, 39):
            atoms14[i] = (Y[-1, i + (3 - 1) * 49] + cm014) * R14s * h1 * aoc / 1000 * 6.02e-3
        for i in range(39, 44):
            atoms14[i] = (Y[-1, i + (3 - 1) * 49] + cm014) * R14s * h2 * aoc / 1000 * 6.02e-3
        atoms14[44] = (Y[-1, 44 + (3 - 1) * 49] + cbio0141) * R14s * 1e15 / 12 * 6.02e-3
        atoms14[45] = (Y[-1, 45 + (3 - 1) * 49] + cbio0142) * R14s * 1e15 / 12 * 6.02e-3
        atoms14[46] = (Y[-1, 46 + (3 - 1) * 49] + cbio0143) * R14s * 1e15 / 12 * 6.02e-3

        # Save number of 14C atoms after spinup for cosmogenic production term
        inputdata['ex14time'] = 1500
        inputdata['ex14data'] = np.sum(atoms14)
        timespan['ex14'] = [1500, 1500]

        C14prog = 1  # prognostic atm D14C for 2005-2100 period
        C13prog = 1  # prognostic atm d13C for 2005-2100 period

         # Run predictions 1982-2025, annual output
        ftime = np.arange(1981, 2026, 1)
        fY = odeint(RHS, Y[-1, :], ftime, args=args)

        # Combine historical simulations in output variable SSPY
        SSPY[:481, :, t, rc] = Y[11501:, :]
        SSPY[481:, :, t, rc] = fY[1:, :]

        del Y 
        del fY


# Calculate atmospheric/ocean/biosphere values to check simulation
d14C = np.full_like(SSPY, np.nan)
D14C = np.full_like(SSPY, np.nan)
d13C = np.full_like(SSPY, np.nan)

d14C[:, 0, :, :] = ((SSPY[:, 0 + (3 - 1) * 49, :, :] + pa014) / (SSPY[:, 0, :, :] + pa0) - 1) * 1000
d13C[:, 0, :, :] = ((SSPY[:, 0 + (2 - 1) * 49, :, :] + pa013) / (SSPY[:, 0, :, :] + pa0) / rs - 1) * 1000
D14C[:, 0, :, :] = d14C[:, 0, :, :] - 2 * (d13C[:, 0, :, :] + 25) * (1 + d14C[:, 0, :, :] / 1000)

# Ocean (similar for the other pools)
for p in range(1, 44):
    d14C[:, p, :, :] = ((SSPY[:, p + (3 - 1) * 49, :, :] + cm014) /
                         (SSPY[:, p, :, :] + cm0) - 1) * 1000
    d13C[:, p, :, :] = ((SSPY[:, p + (2 - 1) * 49, :, :] + cm013) /
                         (SSPY[:, p, :, :] + cm0) / rs - 1) * 1000
    D14C[:, p, :, :] = d14C[:, p, :, :] - 2 * (d13C[:, p, :, :] + 25) * (1 + d14C[:, p, :, :] / 1000)

# Biosphere pools
d14C[:, 44, :, :] = ((SSPY[:, 44 + (3 - 1) * 49, :, :] + cbio0141) /                        (SSPY[:, 44, :, :] + cbio01) - 1) * 1000
d13C[:, 44, :, :] = ((SSPY[:, 44 + (2 - 1) * 49, :, :] + cbio0131) /                        (SSPY[:, 44, :, :] + cbio01) / rs - 1) * 1000
d14C[:, 45, :, :] = ((SSPY[:, 45 + (3 - 1) * 49, :, :] + cbio0142) /                        (SSPY[:, 45, :, :] + cbio02) - 1) * 1000
d13C[:, 45, :, :] = ((SSPY[:, 45 + (2 - 1) * 49, :, :] + cbio0132) /                        (SSPY[:, 45, :, :] + cbio02) / rs - 1) * 1000
d14C[:, 46, :, :] = ((SSPY[:, 46 + (3 - 1) * 49, :, :] + cbio0143) /                        (SSPY[:, 46, :, :] + cbio03) - 1) * 1000
d13C[:, 46, :, :] = ((SSPY[:, 46 + (2 - 1) * 49, :, :] + cbio0133) /                        (SSPY[:, 46, :, :] + cbio03) / rs - 1) * 1000
for p in range(44, 47):
    D14C[:, p, :, :] = d14C[:, p, :, :] - 2 * (d13C[:, p, :, :] + 25) * (1 + d14C[:, p, :, :] / 1000)


# Plot atmospheric CO2
#simtime = np.arange(1500, 2101)
simtime = np.arange(1500, 2025)
plt.figure()
plt.plot(simtime, SSPY[:, 0, :, 0] + pa0, 'g-')
plt.plot(simtime, SSPY[:, 0, :, 1] + pa0, 'm-')
plt.plot(simtime, SSPY[:, 0, :, 2] + pa0, 'b-')
plt.plot(simtime, SSPY[:, 0, :, 3] + pa0, 'y-')
plt.plot(simtime, SSPY[:, 0, :, 4] + pa0, 'r-')
plt.plot(simtime, SSPY[:, 0, :, 5] + pa0, 'k-')
plt.show()

# Plot atmospheric d13C
plt.figure()
plt.plot(simtime, d13C[:, 0, :, 0], 'g-')
plt.plot(simtime, d13C[:, 0, :, 1], 'm-')
plt.plot(simtime, d13C[:, 0, :, 2], 'b-')
plt.plot(simtime, d13C[:, 0, :, 3], 'y-')
plt.plot(simtime, d13C[:, 0, :, 4], 'r-')
plt.plot(simtime, d13C[:, 0, :, 5], 'k-')
plt.plot(inputdata['C13time'], inputdata['C13data'], 'c-')
plt.show()

# Plot atmospheric D14C
plt.figure()
plt.plot(simtime, D14C[:, 0, :, 0], 'g-')
plt.plot(simtime, D14C[:, 0, :, 1], 'm-')
plt.plot(simtime, D14C[:, 0, :, 2], 'b-')
plt.plot(simtime, D14C[:, 0, :, 3], 'y-')
plt.plot(simtime, D14C[:, 0, :, 4], 'r-')
plt.plot(simtime, D14C[:, 0, :, 5], 'k-')
plt.plot(inputdata['C14time'], inputdata['C14data'], 'c-')
plt.show()

# Save output
from scipy.io import savemat
from datetime import datetime

timestamp = datetime.now().strftime('%y%m%d%H%M%S')
#np.save(f'SSPsims_{timestamp}.npy', SSPY)


# Generate a timestamp in the format 'yymmddHHMMSS'
timestamp = datetime.now().strftime('%y%m%d%H%M%S')

# Define the filename with the timestamp
filename = f'SSPsims_{timestamp}.mat'

# Save variables to the .mat file
# Replace `your_variables` with the actual variables you want to save
your_variables = {
    'SSPY': SSPY,  # Outputs
    'param': param,  # Parameters
    'inputdata': inputdata,  # Inputs
    'atmospheric - ocean - biosphere d13C' :d13C,  # d13CO2
    'atmospheric - ocean - biosphere D14C' :D14C,  # D14CO2
    'biospheric carbon C4 herbaceous - box 1 (GtC)' :cbio01_all,  # 
    'biospheric carbon C3 herbaceous - box 2 (GtC)' :cbio02_all,  # 
    'biospheric carbon C3 woody - box 3 (GtC)' :cbio03_all  # 
}

# Save the variables to the .mat file
savemat(filename, your_variables)

print(f"Variables saved to {filename}")


# In[ ]:




