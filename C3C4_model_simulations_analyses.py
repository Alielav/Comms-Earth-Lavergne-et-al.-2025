#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import OrderedDict


# Analysis imports
import numpy.ma as ma
import csv
import netCDF4
from netCDF4 import Dataset
import glob
import warnings

import pandas as pd
from sklearn import linear_model

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import gls

from numpy import NaN
from patsy import dmatrices

import seaborn as sns

import scipy.stats

## Import PYREALM
from pyrealm import pmodel
from pyrealm import C3C4model


## Figures color/style

# color = ['darkblue', 'dodgerblue', '#80b1d3', 'darkcyan', '#8dd3c7', 'darkseagreen', 'darkgreen', 'olive', 'gold', 
#          'orange', 'peachpuff', '#fb8072', 'red', 'hotpink', '#fccde5','#bebada' ]

# These are the "Tableau 20" colors as RGB.    
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]    
  
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)    
  
plt.style.use('default')
plt.rcParams.update({'font.family':'Helvetica'})

colourWheel =['#329932',
            '#ff6961',
            'b',
            '#6a3d9a',
            '#fb9a99',
            '#e31a1c',
            '#fdbf6f',
            '#ff7f00',
            '#cab2d6',
            '#6a3d9a',
            '#ffff99',
            '#b15928',
            '#67001f',
            '#b2182b',
            '#d6604d',
            '#f4a582',
            '#fddbc7',
            '#f7f7f7',
            '#d1e5f0',
            '#92c5de',
            '#4393c3',
            '#2166ac',
            '#053061']
dashesStyles = [[3,1],
            [1000,1],
            [2,1,10,1],
            [4, 1, 1, 1, 1, 1]]

# combine them and build a new colormap
import matplotlib.colors as mcolors
colors1 = plt.cm.Greens(np.linspace(0, 1, 128))
colors2 = plt.cm.Reds_r(np.linspace(0, 1, 128))
#colors3 = plt.cm.Reds(np.linspace(0, 1, 128))
#colors4 = plt.cm.Blues_r(np.linspace(0, 1, 128))

# combine them and build a new colormap
colors = np.vstack((colors2, colors1))
mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)


# In[2]:


# Load climate/remote sensing dataset

### CLIMATE

## Daytime temperature (estimated from WATCH WFDEI product)
ds1 = netCDF4.Dataset('Data/splash_1979_2018_WATCH_WFDEI/1979_2018_Tdaytime.nc')
ds1.set_auto_mask(False)
temp = ds1['Tdtime'][:]
## 1982:2016 period
temp = temp[36:456,:,:]
temp = temp[:,::-1,:]
ds1.close()


## Atmospheric CO2 concentration
ds2 = netCDF4.Dataset('~/Data/co2_annmean_0.5_1982_2016.nc')
ds2.set_auto_mask(False)
co2 = ds2['co2'][:]         # Note - spatially constant but mapped.
ds2.close()


# Elevation (WATCH WFDEI product)
ds3 = netCDF4.Dataset('~/Data/elev.nc')
ds3.set_auto_mask(False)
elev = ds3['elev'][:]  # Note - temporally constant but repeated
ds3.close()


## Vapour pressure deficit (estimated from WATCH WFDEI product)
ds4 = netCDF4.Dataset('~/Data/splash_1979_2018_WATCH_WFDEI/1979_2018_vpd.nc')
ds4.set_auto_mask(False)
vpd = ds4['vpd'][:]
## 1982:2016 period
vpd = vpd[36:456,:,:]
vpd = vpd[:,::-1,:]
ds4.close()


## fraction of absorbed photosynthetic active radiation (fAPAR3g product)
ds5 = netCDF4.Dataset('~/Data/fAPAR/fAPAR3g_v2/fAPAR3g_v2_1982_2016_FILLED.nc')
ds5.set_auto_mask(False)
fapar = ds5['FAPAR_FILLED'][:]
ds5.close()


## Photosynthetic Photon Flux Density (WATCH WFDEI product)
ds6 = netCDF4.Dataset('~/Data/watch_wfdei/PPFD_monthly/1982_2016.ppfd.nc')
ds6.set_auto_mask(False)
ppfd = ds6['ppfd'][:]
ppfd = ppfd[:,::-1,:]

lat1=ds6.variables['lat'][:]
lon1=ds6.variables['lon'][:]
lat1 = lat1[::-1]

ds6.close()


## Soil moisture from SPLASH (estimated from WATCH WFDEI product)
ds7 = netCDF4.Dataset('~/Data/splash_1979_2018_WATCH_WFDEI/1979_2018.theta.nc')
ds7.set_auto_mask(False)
theta = ds7['theta'][:]
## 1982:2016 period
theta = theta[36:456,:,:]
theta = theta[:,::-1,:]

ds7.close()


## Atmospheric d13CO2 

import numpy as geek
ds10 = netCDF4.Dataset('~/Data/d13co2_annmean_0.5_1982_2016.nc')
ds10.set_auto_mask(False)
d13CO2 = ds10['d13CO2'][:]


# Convert elevation to atmospheric pressure
patm = pmodel.calc_patm(elev)



## LANDCOVER/LAND-USE

## MODIS tree cover
ds6 = netCDF4.Dataset('~/Data/MODIS_VCF_Tree_Cover/VCF5KYR_1982_2016_05d.nc')
ds6.set_auto_mask(False)

## Percent Tree Cover: Percent of pixel covered by tree canopy
treecover = ds6['treecover'][:]
ds6.close()


## MODIS land cover
ds6 = netCDF4.Dataset('~/Data/landcover/MODIS/modis_snowandice_0.5d-2010.nc')
ds6.set_auto_mask(False)

snowandice = ds6['snowandice'][:]
ds6.close()

ds6 = netCDF4.Dataset('~/Data/landcover/MODIS/modis_barren_sparsely_vegetated_0.5d-2010.nc')
ds6.set_auto_mask(False)

barren_sparsely_vegetated = ds6['barren_sparsely_vegetated'][:]
ds6.close()


### Calculation of areas to remove
areas_to_remove = (barren_sparsely_vegetated+snowandice)

## mask
a = np.nanmean(temp,axis=0)
a[a >= -19] = 1

## assume when F4 is NA in lands it is instead equal to 0
areas_to_remove = np.nan_to_num(areas_to_remove)
areas_to_remove = areas_to_remove*a

areas_to_remove[areas_to_remove < 100] = 1
areas_to_remove[areas_to_remove >= 100] = np.nan


## LUH2 land cover
# C4 crops
ds6 = netCDF4.Dataset('~/Data/landcover/LUH2/luh2_c4crops.nc')
ds6.set_auto_mask(False)

luh2_c4crops = ds6['c4crops'][:]

# C3 crops
ds6 = netCDF4.Dataset('~/Data/landcover/LUH2/luh2_c3crops.nc')
ds6.set_auto_mask(False)

luh2_c3crops = ds6['c3crops'][:]

# total crops c3 + c4: 
luh2_crops = luh2_c3crops+luh2_c4crops


## urban areas
ds6 = netCDF4.Dataset('~/Data/landcover/LUH2/luh2_urban.nc')
ds6.set_auto_mask(False)

luh2_urban = ds6['urban'][:]

ds6.close()

## Data filtering
temp[temp < -25] = np.nan
temp[temp > 80] = np.nan

fapar[fapar < 0] = np.nan
fapar=fapar[:,0,:,:]

ppfd[ppfd < 0] = np.nan

vpd[vpd < 0] = np.nan

theta[theta < 0] = np.nan


# In[3]:


## Outputs modern simulations
ds6 = netCDF4.Dataset('~/Outputs/pmodel_outputs_modern.nc')

ds6.set_auto_mask(False)
temp_weight = ds6['temp'][:,:,:]
finald13CO2 = ds6['d13co2'][:,:,:]
finalgppc3 = ds6['gppc3'][:,:,:]
finalgppc4 = ds6['ggpc4'][:,:,:]
D13Cplant_C3 = ds6['D13Cc3'][:,:,:]
D13Cplant_C4 = ds6['D13Cc4'][:,:,:]
ds6.close()

D13Cplant_C4[D13Cplant_C4 < 0] = 0
finalgppc3[finalgppc3 > 100000] = np.nan


### Outputs for attribution analysis

## Outputs modern simulations without co2 effect
ds7 = netCDF4.Dataset('~/Outputs/pmodel_outputs_modern_co2_effect.nc')
ds7.set_auto_mask(False)
finalgppc3_co2 = ds7['gppc3'][:,:,:]
finalgppc4_co2 = ds7['ggpc4'][:,:,:]
D13Cplant_C3_co2 = ds7['D13Cc3'][:,:,:]
D13Cplant_C4_co2 = ds7['D13Cc4'][:,:,:]
ds7.close()

## Outputs modern simulations without temp effect
ds7 = netCDF4.Dataset('~/Outputs/ppmodel_outputs_modern_temp_effect.nc')
ds7.set_auto_mask(False)
finalgppc3_temp = ds7['gppc3'][:,:,:]
finalgppc4_temp = ds7['ggpc4'][:,:,:]
D13Cplant_C3_temp = ds7['D13Cc3'][:,:,:]
D13Cplant_C4_temp = ds7['D13Cc4'][:,:,:]
ds7.close()

## Outputs modern simulations without vpd effect
ds7 = netCDF4.Dataset('~/Outputs/pmodel_outputs_modern_vpd_effect.nc')
ds7.set_auto_mask(False)
finalgppc3_vpd = ds7['gppc3'][:,:,:]
finalgppc4_vpd = ds7['ggpc4'][:,:,:]
D13Cplant_C3_vpd = ds7['D13Cc3'][:,:,:]
D13Cplant_C4_vpd = ds7['D13Cc4'][:,:,:]
ds7.close()

finalgppc4_vpd[finalgppc4_vpd > 100000] = np.nan

## Outputs modern simulations without theta effect
ds7 = netCDF4.Dataset('/~/Outputs/pmodel_outputs_modern_theta_effect.nc')
ds7.set_auto_mask(False)
finalgppc3_theta = ds7['gppc3'][:,:,:]
finalgppc4_theta = ds7['ggpc4'][:,:,:]
D13Cplant_C3_theta = ds7['D13Cc3'][:,:,:]
D13Cplant_C4_theta = ds7['D13Cc4'][:,:,:]
ds7.close()


# In[4]:


## Calculation of share of C4 plant in GPP (F4)

crops = 0 ## no region with crops are removed from the calculations of F4

Adv4,Sh4 = C3C4model.c4fraction(temp_weight,finalgppc3,finalgppc4,treecover,crops)

## Conversion of share of C4 plant in GPP into fraction of C4 plant
## Emergent constraint from Luo et al.: share of C4 plant in GPP = (1.11+1.10+1.16+1.15)/4*fraction of C4 plant
## so fraction of C4 plant = share of C4 plant in GPP/1.13
F4 = Sh4/1.13

## Fraction of C4 plants in natural ecosystems

natural = 1 - luh2_crops - luh2_urban
F4_natural = F4*natural


## Fraction of C4 plants with crops
F4_withcrops = luh2_c4crops + F4_natural


## mask to correct map
# When F4 is NA in lands it is instead equal to 0
a = np.nanmean(temp,axis=0)
thresh = -19
a[a >= thresh] = 1
a[a < thresh] = np.nan

## F4 natural ecosystems + crops corrected
F4_withcrops = np.nan_to_num(F4_withcrops)
F4_withcrops = F4_withcrops*a

F4_withcrops = F4_withcrops*areas_to_remove


## F4 for natural ecosystems corrected
F4_natural = np.nan_to_num(F4_natural)
F4_natural = F4_natural*a

F4_natural = F4_natural*areas_to_remove
# F4_natural[F4_natural<0] = 0


## F3 for natural ecosystems

F3_natural = 1 - F4_natural - luh2_urban

## F3 natural ecosystems + crops

F3_withcrops = 1 - F4_withcrops 


## Calculation total GPP 

# Natural ecosystems only
gpp_c3 = finalgppc3*F3_natural
gpp_c4 = finalgppc4*F4_natural
gpp_tot = gpp_c3 + gpp_c4


# Natural ecosystems + crops
gpp_c3_crops = finalgppc3*F3_withcrops
gpp_c4_crops = finalgppc4*F4_withcrops
gpp_tot_crops = gpp_c3_crops + gpp_c4_crops


# In[5]:


np.nanmean(F4_natural), np.nanmean(F4_withcrops), np.nanmean(natural)


# In[6]:


## Parameter values from Graven et al. (2020) to use for turnover time

# Keddy : eddy diffusivity (Keddy) of 3,000–6,000 m2·y−1
# τam: atmospheric CO2 residence time with respect to exchange with the mixed layer of 9–11 y (τam, corresponding to piston velocities of 14.8–18.1 cm·h−1)
# β:  CO2 fertilization factor (β) of 0–0.4
# τab1 τab2 τab3 : atmospheric CO2 residence time with respect to biospheric exchange (τab) of 18–25 y,
# τba1 τba2 τba3 : biospheric residence time (τba) of 20–35 y
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


# In[7]:


### Three boxes: C4 herbaceous, C3 herbaceous and C3 woody

## Turnover time from Graven et al. 2020 

# tau1 = 2.4 # box 1
# tau2 = 24.1 # box 2
# tau3 = 299.1 # box 3

tau1 = param[:,6] # box 1
tau2 = param[:,7] # box 2
tau3 = param[:,8] # box 3


## Carbon use efficiency (CUE) - from Lu et al. (2025)

cue_c4_herb = 0.41 # box 1
cue_c3_herb = 0.46 # box 2
cue_c3_woody = 0.40 # box 3


## Discrimination with C3 and C4 plants 

# ## calculation of F3/F4 index to account for differential effect of C3/C4 on land biomass/discrimination
F3_natural_herb_index = F3_natural[0,:,:]-(F3_natural[0,:,:]-F3_natural)*cue_c3_herb
F3_natural_woody_index = F3_natural[0,:,:]-(F3_natural[0,:,:]-F3_natural)*cue_c3_woody
F4_natural_herb_index = F4_natural[0,:,:]-(F4_natural[0,:,:]-F4_natural)*cue_c4_herb

F4_withcrops_herb_index = F4_withcrops[0,:,:]-(F4_withcrops[0,:,:]-F4_withcrops)*cue_c4_herb
F3_withcrops_herb_index = F3_withcrops[0,:,:]-(F3_withcrops[0,:,:]-F3_withcrops)*cue_c3_herb
F3_withcrops_woody_index = F3_withcrops[0,:,:]-(F3_withcrops[0,:,:]-F3_withcrops)*cue_c3_herb

D13C_c4_herb = D13Cplant_C4*F4_natural_herb_index  
D13C_c3_herb = D13Cplant_C3*F3_natural_herb_index  
D13C_c3_woody = D13Cplant_C3*F3_natural_woody_index  


D13C_tot = (D13C_c4_herb*np.nanmedian(tau1) + D13C_c3_herb*np.nanmedian(tau2) + D13C_c3_woody*np.nanmedian(tau3))/(np.nanmedian(tau1)+np.nanmedian(tau2)+np.nanmedian(tau3))

D13C_c4_herb_crops = D13Cplant_C4*F4_withcrops_herb_index  
D13C_c3_herb_crops = D13Cplant_C3*F3_withcrops_herb_index  
D13C_c3_woody_crops = D13Cplant_C3*F3_withcrops_woody_index 

D13C_tot_crops = (D13C_c4_herb_crops*np.nanmedian(tau1) + D13C_c3_herb_crops*np.nanmedian(tau2) + D13C_c3_woody_crops*np.nanmedian(tau3))/(np.nanmedian(tau1)+np.nanmedian(tau2)+np.nanmedian(tau3))


# In[8]:


### Map from Still et al. (2009)

C4_frac_still = netCDF4.Dataset('~/Data/C3_C4_fraction/ISLSCP_C4_1DEG_932/c4_percent_05d.nc')
C4_frac_still.set_auto_mask(False)
C4_frac_still = C4_frac_still['C4frac'][:]

C4_frac_still[C4_frac_still < 0] = np.nan
C4_frac_still = C4_frac_still/100
C4_frac_still[C4_frac_still > 1] = 1

## mask
a = np.nanmean(temp,axis=0)
a[a >= thresh] = 1
a[a < thresh] = np.nan

## assume when F4 is NA in lands it is instead equal to 0
C4_frac_still = np.nan_to_num(C4_frac_still)
C4_frac_still = C4_frac_still*a*areas_to_remove


## Calculation total GPP

C3_frac_still = 1-C4_frac_still 

gpp_c3_still = finalgppc3*C3_frac_still
gpp_c4_still = finalgppc4*C4_frac_still
gpp_tot_still = gpp_c3_still + gpp_c4_still


## Discrimination with C3 and C4 plants

C4_frac_still_herb_index = C4_frac_still-(C4_frac_still-C4_frac_still)*cue_c4_herb
C3_frac_still_herb_index = C3_frac_still-(C3_frac_still-C3_frac_still)*cue_c3_herb
C3_frac_still_woody_index = C3_frac_still-(C3_frac_still-C3_frac_still)*cue_c3_woody

D13C_c4_herb_still = D13Cplant_C4*C4_frac_still_herb_index 
D13C_c3_herb_still = D13Cplant_C3*C3_frac_still_herb_index 
D13C_c3_woody_still = D13Cplant_C3*C3_frac_still_woody_index 

D13C_tot_still = (D13C_c4_herb_still*np.nanmedian(tau1) + D13C_c3_herb_still*np.nanmedian(tau2) + D13C_c3_woody_still*np.nanmedian(tau3))/(np.nanmedian(tau1)+np.nanmedian(tau2)+np.nanmedian(tau3))



### Map from Luo et al. (2024, Nat. comms.)

C4_frac_luo1 = netCDF4.Dataset('~/Data/C3_C4_fraction/Luoetal/C4_distribution_NUS_v2.2.nc')
C4_frac_luo1.set_auto_mask(False)
C4_frac_luo1 = C4_frac_luo1['C4_area'][:]


C4_frac_luo1 = C4_frac_luo1/100
C4_frac_luo1 = np.transpose(C4_frac_luo1)
C4_frac_luo1 = C4_frac_luo1[::-1,:,0:16]  ## for period 2001-2016 in common


C4_frac_luo = np.empty((16,360,720))
C4_frac_luo[:] = np.nan

for x in range(360):
    for y in range(720):
        C4_frac_luo[:,x,y] = C4_frac_luo1[x,y,:]
  
C4_frac_luo = np.array(C4_frac_luo)

## assume when F4 is NA in lands it is instead equal to 0
## mask
a = np.nanmean(temp,axis=0)
a[a >= thresh] = 1
a[a < thresh] = np.nan

C4_frac_luo = np.nan_to_num(C4_frac_luo)
C4_frac_luo = C4_frac_luo*a*areas_to_remove


### Calculation range of values
p = 0.95
uncert= 2/100


n = [0 for i in range(16)]
F4_luo_std = [0 for i in range(16)]
F4_luo_ci = [0 for i in range(16)]

for i in range(16):
    n[i] = np.count_nonzero(~np.isnan(C4_frac_luo[i,:,:]))
    F4_luo_std[i] = np.nanstd(C4_frac_luo,axis=(1,2))[i]/ np.sqrt(n[i])
    F4_luo_ci[i] = F4_luo_std[i] * scipy.stats.t.ppf((1 + p) / 2., n[i] -1) + np.nanmean(C4_frac_luo,axis=(1,2))[i]*uncert


## Calculation total GPP with Luo et al. map
C3_frac_luo = 1 - C4_frac_luo 

gpp_c3_luo = finalgppc3[19:35,:,:]*C3_frac_luo
gpp_c4_luo = finalgppc4[19:35,:,:]*C4_frac_luo
gpp_tot_luo = gpp_c3_luo + gpp_c4_luo


## Discrimination with C3 and C4 plants with Luo et al. map
C4_frac_luo_herb_index = C4_frac_luo[0,:,:]-(C4_frac_luo[0,:,:]-C4_frac_luo)*cue_c4_herb
C3_frac_luo_herb_index = C3_frac_luo[0,:,:]-(C3_frac_luo[0,:,:]-C3_frac_luo)*cue_c3_herb
C3_frac_luo_woody_index = C3_frac_luo[0,:,:]-(C3_frac_luo[0,:,:]-C3_frac_luo)*cue_c3_woody


D13C_c4_herb_luo = D13Cplant_C4[19:35,:,:]*C4_frac_luo_herb_index 
D13C_c3_herb_luo = D13Cplant_C3[19:35,:,:]*C3_frac_luo_herb_index
D13C_c3_woody_luo = D13Cplant_C3[19:35,:,:]*C3_frac_luo_woody_index

D13C_tot_luo = (D13C_c4_herb_luo*np.nanmedian(tau1) + D13C_c3_herb_luo*np.nanmedian(tau2) + D13C_c3_woody_luo*np.nanmedian(tau3))/(np.nanmedian(tau1)+np.nanmedian(tau2)+np.nanmedian(tau3))


# In[9]:


## Comparison model-isotopic data ##

# Find grid points when latitude are the closest

def get_data(lat_input, long_input):

    lat_index  = np.nanargmin((np.array(lat1)-lat_input)**2)
    long_index = np.nanargmin((np.array(lon1)-long_input)**2)
    return lat_index,long_index


## Extraction soil isotopic data from Ning Dong et al. compilation
data_isotope_ning = pd.read_csv('~/Data/Glob_Soil_δ13C.csv', index_col=0, na_values=['(NA)'])

data_isotope_ning['Lat'] = data_isotope_ning['Lat'].apply(pd.to_numeric, errors='coerce')
data_isotope_ning['Lon'] = data_isotope_ning['Lon'].apply(pd.to_numeric, errors='coerce')
data_isotope_ning = data_isotope_ning.dropna(subset=["Lat","Lon"])


## Extraction leaf isotopic data from Cornwell et al. compilation

data_isotope_corn = pd.read_csv('~/Data/leaf13C.csv', index_col=0, na_values=['(NA)'])

data_isotope_corn['latitude'] = data_isotope_corn['latitude'].apply(pd.to_numeric, errors='coerce')
data_isotope_corn['longitude'] = data_isotope_corn['longitude'].apply(pd.to_numeric, errors='coerce')
data_isotope_corn = data_isotope_corn.dropna(subset=["latitude","longitude"])


## Extraction data for C3 and C4 plants

## 1- Ning Dong compilation soil
## All
coord_ning_site = [[0 for i in range(2)] for i in range(len(data_isotope_ning.Lon))]  

for x in range(len(data_isotope_ning.Lat)):
    coord_ning_site[x] = get_data(data_isotope_ning.Lat[x], data_isotope_ning.Lon[x])
coord_ning_site = np.squeeze(pd.DataFrame(coord_ning_site))    


## Separated
# C3
data_isotope_ning_C3 = data_isotope_ning[data_isotope_ning.Type == "C3"]
data_isotope_ning_C4 = data_isotope_ning[data_isotope_ning.Type == "C4"]

coord_ning_site_C3 = [[0 for i in range(2)] for i in range(len(data_isotope_ning_C3.Lon))]  

for x in range(len(data_isotope_ning_C3.Lat)):
    coord_ning_site_C3[x] = get_data(data_isotope_ning_C3.Lat[x], data_isotope_ning_C3.Lon[x])
coord_ning_site_C3 = np.squeeze(pd.DataFrame(coord_ning_site_C3))    

# C4
coord_ning_site_C4 = [[0 for i in range(2)] for i in range(len(data_isotope_ning_C4.Lon))]  

for x in range(len(data_isotope_ning_C4.Lat)):
    coord_ning_site_C4[x] = get_data(data_isotope_ning_C4.Lat[x], data_isotope_ning_C4.Lon[x])
coord_ning_site_C4 = np.squeeze(pd.DataFrame(coord_ning_site_C4))    


## 2- Cornwell compilation leaf material
## All
coord_corn_site = [[0 for i in range(2)] for i in range(len(data_isotope_corn.longitude))]  

for x in range(len(data_isotope_corn.latitude)):
    coord_corn_site[x] = get_data(data_isotope_corn.latitude[x], data_isotope_corn.longitude[x])
coord_corn_site = np.squeeze(pd.DataFrame(coord_corn_site))    


## Separated
# C3
data_isotope_corn_C3 = data_isotope_corn[data_isotope_corn.Type == "C3"]
data_isotope_corn_C4 = data_isotope_corn[data_isotope_corn.Type == "C4"]

coord_corn_site_C3 = [[0 for i in range(2)] for i in range(len(data_isotope_corn_C3.longitude))]  

for x in range(len(data_isotope_corn_C3.latitude)):
    coord_corn_site_C3[x] = get_data(data_isotope_corn_C3.latitude[x], data_isotope_corn_C3.longitude[x])
coord_corn_site_C3 = np.squeeze(pd.DataFrame(coord_corn_site_C3))    

# C4
coord_corn_site_C4 = [[0 for i in range(2)] for i in range(len(data_isotope_corn_C4.longitude))]  

for x in range(len(data_isotope_corn_C4.latitude)):
    coord_corn_site_C4[x] = get_data(data_isotope_corn_C4.latitude[x], data_isotope_corn_C4.longitude[x])
coord_corn_site_C4 = np.squeeze(pd.DataFrame(coord_corn_site_C4))    


## Extracting modelled vD13C for each site from the isotopic network of Cornwell
D13Cplant_corn_sites_C3_weight = D13Cplant_C3[:,coord_corn_site_C3[0],coord_corn_site_C3[1]]
D13Cplant_corn_sites_C4_weight = D13Cplant_C4[:,coord_corn_site_C4[0],coord_corn_site_C4[1]]

D13Cplant_corn_sites_C3_weight_yr = np.nanmean(D13Cplant_corn_sites_C3_weight,axis=0)
D13Cplant_corn_sites_C4_weight_yr = np.nanmean(D13Cplant_corn_sites_C4_weight,axis=0)


## Extraction D13C data for given year
year = np.arange(0,33,1) #+1982


## Cornwell compilation
## C3 only        
for x in range(len(data_isotope_corn_C3.year)):
    if (data_isotope_corn_C3.year[x] > 1982) & (data_isotope_corn_C3.year[x] < 2015):
        yr = data_isotope_corn_C3.year[x]-1982
        D13Cplant_corn_sites_C3_weight_yr[x] = D13Cplant_corn_sites_C3_weight[int(yr),x]
    else:
        D13Cplant_corn_sites_C3_weight_yr[x] = np.nanmean(D13Cplant_corn_sites_C3_weight[:,x],axis=0)

## C4 only        
for x in range(len(data_isotope_corn_C4.year)):
    if (data_isotope_corn_C4.year[x] > 1982) & (data_isotope_corn_C4.year[x] < 2015):
        yr = data_isotope_corn_C4.year[x]-1982
        D13Cplant_corn_sites_C4_weight_yr[x] = D13Cplant_corn_sites_C4_weight[int(yr),x]
    else:
        D13Cplant_corn_sites_C4_weight_yr[x] = np.nanmean(D13Cplant_corn_sites_C4_weight[:,x],axis=0)

        
## C3 + C4
D13Cplant_corn_sites_weight_yr = np.concatenate((D13Cplant_corn_sites_C3_weight_yr,D13Cplant_corn_sites_C4_weight_yr),axis=0)

data_leafD13C_corn_sites_weight_yr = np.concatenate((data_isotope_corn_C3.leaf_D13C,data_isotope_corn_C4.leaf_D13C),axis=0)

D13Cplant_corn_sites_weight_yr[D13Cplant_corn_sites_weight_yr == 0] = np.nan



## Calculation modelled d13Cplant for C3 and C4 plants

d13Cplant_model = [[0 for i in range(len(lon1))] for j in range(len(lat1))] 
d13Cplant_model_crops = [[0 for i in range(len(lon1))] for j in range(len(lat1))] 
d13Cplant_still = [[0 for i in range(len(lon1))] for j in range(len(lat1))] 
d13Cplant_luo = [[0 for i in range(len(lon1))] for j in range(len(lat1))] 

for x in range(len(lat1)):
    for y in range(len(lon1)):
            ## without carbon turnover added
            d13Cplant_model[x][y] = (finald13CO2[:,x,y] - D13C_tot[:,x,y])/(1 + D13C_tot[:,x,y]/1000) 
            d13Cplant_model_crops[x][y] = (finald13CO2[:,x,y] - D13C_tot_crops[:,x,y])/(1 + D13C_tot_crops[:,x,y]/1000)  
            d13Cplant_still[x][y] = (finald13CO2[:,x,y] - D13C_tot_still[:,x,y])/(1 + D13C_tot_still[:,x,y]/1000)
            d13Cplant_luo[x][y] = (finald13CO2[19:35,x,y] - D13C_tot_luo[:,x,y])/(1 + D13C_tot_luo[:,x,y]/1000) 
            

d13Cplant_model = np.squeeze(np.asarray(d13Cplant_model))
d13Cplant_model = np.transpose(d13Cplant_model, (2, 0, 1))

d13Cplant_model_crops = np.squeeze(np.asarray(d13Cplant_model_crops))
d13Cplant_model_crops = np.transpose(d13Cplant_model_crops, (2, 0, 1))

d13Cplant_still = np.squeeze(np.asarray(d13Cplant_still))
d13Cplant_still = np.transpose(d13Cplant_still, (2, 0, 1))

d13Cplant_luo = np.squeeze(np.asarray(d13Cplant_luo))
d13Cplant_luo = np.transpose(d13Cplant_luo, (2, 0, 1))


## Extraction of modelled d13C plant at sites with soil d13C data
d13Cplant_model_ning_sites = d13Cplant_model[:,coord_ning_site[0],coord_ning_site[1]]
d13Cplant_model_ning_sites = np.nanmean(d13Cplant_model_ning_sites,axis=0)

d13Cplant_model_ning_sites_crops = d13Cplant_model_crops[:,coord_ning_site[0],coord_ning_site[1]]
d13Cplant_model_ning_sites_crops = np.nanmean(d13Cplant_model_ning_sites_crops,axis=0)


## Soil d13C data Still
d13Cplant_still_ning_sites = d13Cplant_still[:,coord_ning_site[0],coord_ning_site[1]]
d13Cplant_still_ning_sites = np.nanmean(d13Cplant_still_ning_sites,axis=0)

## Soil d13C data Luo
d13Cplant_luo_ning_sites = d13Cplant_luo[:,coord_ning_site[0],coord_ning_site[1]]
d13Cplant_luo_ning_sites = np.nanmean(d13Cplant_luo_ning_sites,axis=0)


## Leaf data for C3 and C4 plants with Cornwell
## for C3 plants

Discr_model_corn_sites_C3_weight = D13Cplant_C3[:,coord_corn_site_C3[0],coord_corn_site_C3[1]]
Discr_model_corn_sites_C3_weight_yr = np.nanmean(Discr_model_corn_sites_C3_weight,axis=0)

for x in range(len(data_isotope_corn_C3.year)):
    if (data_isotope_corn_C3.year[x] > 1982) & (data_isotope_corn_C3.year[x] < 2015):
        yr = data_isotope_corn_C3.year[x]-1982
        Discr_model_corn_sites_C3_weight_yr[x] = Discr_model_corn_sites_C3_weight[int(yr),x]
    else:
        Discr_model_corn_sites_C3_weight_yr[x] = np.nanmean(Discr_model_corn_sites_C3_weight[:,x],axis=0)

Discr_model_corn_sites_C3 = Discr_model_corn_sites_C3_weight_yr
Discr_model_corn_sites_C3 = np.nanmean(Discr_model_corn_sites_C3_weight,axis=0)
Discr_model_corn_sites_C3 = D13Cplant_corn_sites_C3_weight_yr


## for C4 plants
Discr_model_corn_sites_C4_weight = D13Cplant_C4[:,coord_corn_site_C4[0],coord_corn_site_C4[1]]
Discr_model_corn_sites_C4_weight_yr = np.nanmean(Discr_model_corn_sites_C4_weight,axis=0)

for x in range(len(data_isotope_corn_C4.year)):
    if (data_isotope_corn_C4.year[x] > 1982) & (data_isotope_corn_C4.year[x] < 2015):
        yr = data_isotope_corn_C4.year[x]-1982
        Discr_model_corn_sites_C4_weight_yr[x] = Discr_model_corn_sites_C4_weight[int(yr),x]
    else:
        Discr_model_corn_sites_C4_weight_yr[x] = np.nanmean(Discr_model_corn_sites_C4_weight[:,x],axis=0)

Discr_model_corn_sites_C4 = Discr_model_corn_sites_C4_weight_yr
Discr_model_corn_sites_C4 = np.nanmean(Discr_model_corn_sites_C4_weight,axis=0)
Discr_model_corn_sites = np.concatenate((Discr_model_corn_sites_C3,Discr_model_corn_sites_C4),axis=0)
           
data_isotope_ning.soil_d13C[data_isotope_ning.soil_d13C < -32] = np.nan

## Calculation Pearson's correlations

df_model_reg1 = pd.DataFrame({'model': Discr_model_corn_sites_C3, 'leaf':data_isotope_corn_C3.leaf_D13C}) 
df_model_reg1.corr(method ='pearson')

df_model_reg2 = pd.DataFrame({'model': Discr_model_corn_sites_C4, 'leaf':data_isotope_corn_C4.leaf_D13C}) 
df_model_reg2.corr(method ='pearson')

df_model_reg3 = pd.DataFrame({'model': Discr_model_corn_sites, 'leaf':data_leafD13C_corn_sites_weight_yr}) 
df_model_reg3.corr(method ='pearson')

mdf01 = gls('leaf ~ model', data=df_model_reg1).fit()
print('model R2: ', round(mdf01.rsquared,2), 'params', mdf01.params, 'pval', mdf01.pvalues)

mdf02 = gls('leaf ~ model', data=df_model_reg2).fit()
print('model R2: ', round(mdf02.rsquared,2), 'params', mdf02.params, 'pval', mdf02.pvalues)

mdf03 = gls('leaf ~ model', data=df_model_reg3).fit()
print('model R2: ', round(mdf03.rsquared,2), 'params', mdf03.params, 'pval', mdf03.pvalues)


df_model_reg5 = pd.DataFrame({'model': d13Cplant_model_ning_sites, 'model_crops': d13Cplant_model_ning_sites_crops, 'still': d13Cplant_still_ning_sites, 'soil':data_isotope_ning.soil_d13C}) 
df_model_reg5.corr(method ='pearson')

mdf04 = gls('soil ~ model', data=df_model_reg5).fit()
print('model R2: ', round(mdf04.rsquared,2), 'params', mdf04.params, 'pval', mdf04.pvalues)

mdf04 = gls('soil ~ model_crops', data=df_model_reg5).fit()
print('model R2: ', round(mdf04.rsquared,2), 'params', mdf04.params, 'pval', mdf04.pvalues)


mdf05 = gls('soil ~ still', data=df_model_reg5).fit()
print('model R2: ', round(mdf05.rsquared,2), 'params', mdf05.params, 'pval', mdf05.pvalues)


df_model_reg6 = pd.DataFrame({'model': d13Cplant_model_ning_sites, 'luo': d13Cplant_luo_ning_sites, 'soil':data_isotope_ning.soil_d13C}) 
df_model_reg6.corr(method ='pearson')

mdf06 = gls('soil ~ luo', data=df_model_reg6).fit()
print('model R2: ', round(mdf06.rsquared, 2), 'params', mdf06.params, 'pval', mdf06.pvalues)


# In[10]:


## Figure 1

# Set up the subplot figure

fig_figure1 = plt.figure(1, figsize=(17,18))

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


gs = gspec.GridSpec(5, 7, figure=fig_figure1, hspace=0.2,  width_ratios=[0.9, 0.2, 0.9, 0.05, 0.9, 0.05, 0.9])

# Figure a
column = 0
row = 0

ax = fig_figure1.add_subplot(gs[row, column])

ax.plot([0,34], [0,34], color='grey', label='fitted line', ls='dotted')

ax.plot(D13Cplant_corn_sites_weight_yr, mdf03.params.Intercept + mdf03.params.model*(D13Cplant_corn_sites_weight_yr), color='black', label='fitted line', ls='dotted')

ax.scatter((D13Cplant_corn_sites_C3_weight_yr), data_isotope_corn_C3.leaf_D13C, color=tableau20[9])

ax.scatter((Discr_model_corn_sites_C4), data_isotope_corn_C4.leaf_D13C, color=tableau20[10],marker="^")

ax.set_xlabel(u'$\mathregular{\u0394_{pmodel}}$ (‰)',fontsize=18)
ax.set_ylabel(u'$\mathregular{\u0394_{leaf}}$ (‰)',fontsize=18)
ax.set_ylim((0,34))
ax.set_xlim((0,34))
ax.set_yticks([5, 10, 15, 20, 25,30],fontsize=16)
ax.set_xticks([5, 10, 15, 20, 25,30],fontsize=16)

ax.tick_params(labelsize=16)

ax.text(1.5, 24, u'$R^2$ = O.50', color=tableau20[9],fontsize=16)
ax.text(10, 5, u'$R^2$ = O.23', color=tableau20[10],fontsize=16)
ax.text(20, 1, u'$R^2$ = O.92', color='black',fontsize=16)

ax.text(0.05, 0.95, u'(a)',transform=ax.transAxes,va = 'top',fontsize=18)


# Figure b
column = 2

ax = fig_figure1.add_subplot(gs[row, column])

ax.plot([-35,-5], [-35,-5], color='grey', label='fitted line', ls='dotted')

ax.plot(df_model_reg5.model_crops, df_model_reg5.soil,'o', color="grey")
sns.regplot(x = "model_crops",y = "soil", 
            data = df_model_reg5,order=1,color="black")

ax.set_xlabel(u'$\mathregular{\u03B4^{13}C_{pmodel}}$ (‰)',fontsize=18)
ax.set_ylabel(u'$\mathregular{\u03B4^{13}C_{soil}}$ (‰)',fontsize=18)
ax.set_ylim((-33,-5))
ax.set_xlim((-33,-5))
ax.set_yticks([-30, -25, -20,-15, -10, -5],fontsize=16)
ax.set_xticks([-30,-25, -20, -15, -10, -5],fontsize=16)

ax.tick_params(labelsize=16)

ax.text(0.05, 0.95, '(b) Our model',transform=ax.transAxes,va = 'top',fontsize=16)
ax.text(0.58, 0.12, u'$R^2$ = O.58',transform=ax.transAxes,va = 'top',fontsize=16)


# Figure c
column = 4

ax = fig_figure1.add_subplot(gs[row, column])

ax.plot([-35,-5], [-35,-5], color='grey', label='fitted line', ls='dotted')

ax.plot(df_model_reg5.still, df_model_reg5.soil,'o', color="grey")
sns.regplot(x = "still",y = "soil", 
            data = df_model_reg5,order=1,color="black")

ax.set_xlabel(u'$\mathregular{\u03B4^{13}C_{pmodel}}$ (‰)',fontsize=18)
ax.set_ylabel(u' ',fontsize=18)
ax.set_ylim((-33,-5))
ax.set_xlim((-33,-5))
ax.set_yticks([-30, -25, -20,-15, -10, -5],fontsize=16)
ax.set_xticks([-30,-25, -20, -15, -10, -5],fontsize=16)

ax.tick_params(labelsize=16)

ax.text(0.05, 0.95, '(c) Still2009',transform=ax.transAxes,va = 'top',fontsize=16)
ax.text(0.58, 0.12, u'$R^2$ = O.32',transform=ax.transAxes,va = 'top',fontsize=16)


# Figure d
column = 6

ax = fig_figure1.add_subplot(gs[row, column])

ax.plot([-35,-5], [-35,-5], color='grey', label='fitted line', ls='dotted')
ax.plot(df_model_reg6.luo, df_model_reg6.soil,'o', color="gray")
sns.regplot(x = "luo",y = "soil", 
           data = df_model_reg6,order=1,color="black")

ax.set_xlabel(u'$\mathregular{\u03B4^{13}C_{pmodel}}$ (‰)',fontsize=18)
ax.set_ylabel(u' ',fontsize=18)
ax.set_ylim((-33,-5))
ax.set_xlim((-33,-5))
ax.set_yticks([-30, -25, -20,-15, -10, -5],fontsize=16)
ax.set_xticks([-30,-25, -20, -15, -10, -5],fontsize=16)

ax.tick_params(labelsize=16)

ax.text(0.05, 0.95, '(d) Luo2024',transform=ax.transAxes,va = 'top',fontsize=16)
ax.text(0.58, 0.12, u'$R^2$ = O.37',transform=ax.transAxes,va = 'top',fontsize=16)


fig_figure1.savefig('~/Figure1.pdf', bbox_inches='tight')

plt.close()


# In[11]:


## Figure 2

# Set up the subplot figure

fig_figure2 = plt.figure(1, figsize=(20,35))
gs = gspec.GridSpec(4, 3, figure=fig_figure2, width_ratios=[0.15, 1, 0.05], hspace=0.3)
# set rows and column
column = 0
row = 0
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

#%%

# Figure a

column += 1
ax = fig_figure2.add_subplot(gs[row, column], projection=ccrs.Robinson())


# Add lat/lon grid lines to the figure
gl = ax.gridlines(linestyle='solid', color='white', linewidth=0.5, alpha=0.5)
gl.yformatter=LATITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator([-60, -40, -20, 0, 20, 40, 60, 80])
gl.ylabels_left=True
gl = ax.gridlines(linestyle='solid', color='white', linewidth=0.5, alpha=0.5)
gl.xformatter=LONGITUDE_FORMATTER
gl.xlabels_bottom=True
ax.coastlines()

line = np.arange(0, 1.000001, 0.05)
diff = plt.contourf(lon1, lat1, (np.nanmean(F4_withcrops[19:35,:,:],axis=0)),line, cmap2 = 'YIOrRd', extend='neither', transform=ccrs.PlateCarree(central_longitude=0))

ax.text(-0.1, 1.15, '(a) Fraction of $C_4$ plants with crops from our model',transform=ax.transAxes,va = 'top',fontweight = 'bold',fontsize=30)

ax=fig_figure2.add_subplot(gs[row,2])
ax=plt.gca()
fig_figure2.colorbar(diff, ax, orientation='vertical').set_label(u'$F_4$')


# Figure b
column = 1
row = 1
ax = fig_figure2.add_subplot(gs[row, column], projection=ccrs.Robinson())


# Add lat/lon grid lines to the figure
gl = ax.gridlines(linestyle='solid', color='white', linewidth=0.5, alpha=0.5)
gl.yformatter=LATITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator([-60, -40, -20, 0, 20, 40, 60, 80])
gl.ylabels_left=True
gl = ax.gridlines(linestyle='solid', color='white', linewidth=0.5, alpha=0.5)
gl.xformatter=LONGITUDE_FORMATTER
gl.xlabels_bottom=True
ax.coastlines()

line = np.arange(0, 1.000001, 0.05)
diff = plt.contourf(lon1, lat1, C4_frac_still,line, cmap2 = 'YIOrRd', extend='neither', transform=ccrs.PlateCarree(central_longitude=0))

ax.text(-0.1, 1.15, '(b) Fraction of $C_4$ plants from Still2009',transform=ax.transAxes,va = 'top',fontweight = 'bold',fontsize=30)

ax=fig_figure2.add_subplot(gs[row,2])
ax=plt.gca()
fig_figure2.colorbar(diff, ax, orientation='vertical').set_label(u'$F_4$')


# Figure c

column = 1
row = 2
ax = fig_figure2.add_subplot(gs[row, column], projection=ccrs.Robinson())


# Add lat/lon grid lines to the figure
gl = ax.gridlines(linestyle='solid', color='white', linewidth=0.5, alpha=0.5)
gl.yformatter=LATITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator([-60, -40, -20, 0, 20, 40, 60, 80])
gl.ylabels_left=True
gl = ax.gridlines(linestyle='solid', color='white', linewidth=0.5, alpha=0.5)
gl.xformatter=LONGITUDE_FORMATTER
gl.xlabels_bottom=True
ax.coastlines()

line = np.arange(0, 1.000001, 0.05)

diff = plt.contourf(lon1, lat1, (np.nanmean(C4_frac_luo,axis=0)),line, cmap2 = 'YIOrRd', extend='neither', transform=ccrs.PlateCarree(central_longitude=0))

ax.text(-0.1, 1.15, '(c) Fraction of $C_4$ plants from Luo2024',transform=ax.transAxes,va = 'top',fontweight = 'bold',fontsize=30)


ax=fig_figure2.add_subplot(gs[row,2])
ax=plt.gca()
fig_figure2.colorbar(diff, ax, orientation='vertical').set_label(u'$F_4$')


fig_figure2.savefig('~/Figure2.pdf', bbox_inches='tight')

plt.close()


# In[ ]:


## Calculation trends

import statsmodels.api as sm
import statsmodels.formula.api as smf

from numpy import NaN
from patsy import dmatrices

## Calculation temporal trend for F4
## 1982-2016

trendF4 = [[0 for i in range(len(lon1))] for j in range(len(lat1))] 

for x in range(len(lat1)):
    for y in range(len(lon1)):
        F4_withcrops[:,x,y][F4_withcrops[:,x,y]<=0] = np.nan
        md2 = sm.GLS(F4_withcrops[:,x,y],sm.add_constant(range(35))).fit()
        if md2.pvalues[1]>0.05:
            trendF4[x][y] = np.nan
        else:
            trendF4[x][y] = md2.params[1]     
            
trendF4 = np.squeeze(np.asarray(trendF4))


## 2001-2016: for comparison with Luo et al. 2024

trendF4_short = [[0 for i in range(len(lon1))] for j in range(len(lat1))] 
trendF4_luo = [[0 for i in range(len(lon1))] for j in range(len(lat1))] 

for x in range(len(lat1)):
    for y in range(len(lon1)):
        F4_withcrops[19:35,x,y][F4_withcrops[19:35,x,y]<=0] = np.nan
        md2 = sm.GLS(F4_withcrops[19:35,x,y],sm.add_constant(range(16))).fit()
        if md2.pvalues[1]>0.05:
            trendF4_short[x][y] = np.nan
        else:
            trendF4_short[x][y] = md2.params[1] 
            
        C4_frac_luo[:,x,y][C4_frac_luo[:,x,y]<=0] = np.nan
        md2 = sm.GLS(C4_frac_luo[:,x,y],sm.add_constant(range(16))).fit()
        if md2.pvalues[1]>0.05:
            trendF4_luo[x][y] = np.nan
        else:
            trendF4_luo[x][y] = md2.params[1] 
            
trendF4_short = np.squeeze(np.asarray(trendF4_short))
trendF4_luo = np.squeeze(np.asarray(trendF4_luo))


# In[ ]:


## Figure 3

# Set up the subplot figure

fig_figure3 = plt.figure(1, figsize=(20,35))
gs = gspec.GridSpec(4, 3, figure=fig_figure3, width_ratios=[0.15, 1, 0.05], hspace=0.3)
# set rows and column
column = 0
row = 0
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

#%%

# Figure a

column += 1
ax = fig_figure3.add_subplot(gs[row, column], projection=ccrs.Robinson())


# Add lat/lon grid lines to the figure
gl = ax.gridlines(linestyle='solid', color='white', linewidth=0.5, alpha=0.5)
gl.yformatter=LATITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator([-60, -40, -20, 0, 20, 40, 60, 80])
gl.ylabels_left=True
gl = ax.gridlines(linestyle='solid', color='white', linewidth=0.5, alpha=0.5)
gl.xformatter=LONGITUDE_FORMATTER
gl.xlabels_bottom=True
ax.coastlines()

line = np.arange(-0.012, 0.013, 0.001)
line = np.arange(-0.009, 0.010, 0.001)

diff = plt.contourf(lon1, lat1, trendF4, line, cmap = mymap, extend='neither', transform=ccrs.PlateCarree(central_longitude=0))

ax.text(-0.1, 1.15, '(a) Our model over 1982-2016',transform=ax.transAxes,va = 'top',fontweight = 'bold',fontsize=34)

ax=fig_figure3.add_subplot(gs[row,2])
ax=plt.gca()
fig_figure3.colorbar(diff, ax, orientation='vertical').set_label(u'$F_{4}$ trend (yr$^{-1}$)')


# Figure b

column = 1
row +=1
ax = fig_figure3.add_subplot(gs[row, column], projection=ccrs.Robinson())


# Add lat/lon grid lines to the figure
gl = ax.gridlines(linestyle='solid', color='white', linewidth=0.5, alpha=0.5)
gl.yformatter=LATITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator([-60, -40, -20, 0, 20, 40, 60, 80])
gl.ylabels_left=True
gl = ax.gridlines(linestyle='solid', color='white', linewidth=0.5, alpha=0.5)
gl.xformatter=LONGITUDE_FORMATTER
gl.xlabels_bottom=True
ax.coastlines()

diff = plt.contourf(lon1, lat1, trendF4_short, line, cmap = mymap, extend='neither', transform=ccrs.PlateCarree(central_longitude=0))

ax.text(-0.1, 1.15, '(b) Our model over 2001-2016',transform=ax.transAxes,va = 'top',fontweight = 'bold',fontsize=34)

ax=fig_figure3.add_subplot(gs[row,2])
ax=plt.gca()
fig_figure3.colorbar(diff, ax, orientation='vertical').set_label(u'$F_{4}$ trend (yr$^{-1}$)')


# Figure c

column = 1
row = 2
ax = fig_figure3.add_subplot(gs[row, column], projection=ccrs.Robinson())


# Add lat/lon grid lines to the figure
gl = ax.gridlines(linestyle='solid', color='white', linewidth=0.5, alpha=0.5)
gl.yformatter=LATITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator([-60, -40, -20, 0, 20, 40, 60, 80])
gl.ylabels_left=True
gl = ax.gridlines(linestyle='solid', color='white', linewidth=0.5, alpha=0.5)
gl.xformatter=LONGITUDE_FORMATTER
gl.xlabels_bottom=True
ax.coastlines()

diff = plt.contourf(lon1, lat1, trendF4_luo, line,cmap = mymap, extend='neither', transform=ccrs.PlateCarree(central_longitude=0))

ax.text(-0.1, 1.15, '(c) Luo2024 map over 2001-2016',transform=ax.transAxes,va = 'top',fontweight = 'bold',fontsize=34)

ax=fig_figure3.add_subplot(gs[row,2])
ax=plt.gca()
fig_figure3.colorbar(diff, ax, orientation='vertical').set_label(u'$F_{4}$ trend (yr$^{-1}$)')


fig_figure3.savefig('~/Figure3.pdf', bbox_inches='tight')

plt.close()


# In[12]:


## Standard error and confidence interval at 95% + 5% uncertainty due to observations
p = 0.95

uncert = 2/100

n = [0 for i in range(35)]
F4_std = [0 for i in range(35)]
F4_ci = [0 for i in range(35)]

for i in range(35):
    n[i] = np.count_nonzero(~np.isnan(F4[i,:,:]))
    F4_std[i] = np.nanstd(F4,axis=(1,2))[i]/ np.sqrt(n[i])
    F4_ci[i] = F4_std[i] * scipy.stats.t.ppf((1 + p) / 2., n[i] -1) + np.nanmean(F4,axis=(1,2))[i]*uncert

n = [0 for i in range(35)]
F4_natural_std = [0 for i in range(35)]
F4_natural_ci = [0 for i in range(35)]

for i in range(35):
    n[i] = np.count_nonzero(~np.isnan(F4_natural[i,:,:]))
    F4_natural_std[i] = np.nanstd(F4_natural,axis=(1,2))[i]/ np.sqrt(n[i])
    F4_natural_ci[i] = F4_natural_std[i] * scipy.stats.t.ppf((1 + p) / 2., n[i] -1)  + np.nanmean(F4_natural,axis=(1,2))[i]*uncert

n = [0 for i in range(35)]
luh2_c4crops_std = [0 for i in range(35)]
luh2_c4crops_ci = [0 for i in range(35)]

for i in range(35):
    n[i] = np.count_nonzero(~np.isnan(luh2_c4crops[i,:,:]))
    luh2_c4crops_std[i] = np.nanstd(luh2_c4crops,axis=(1,2))[i]/ np.sqrt(n[i])
    luh2_c4crops_ci[i] = luh2_c4crops_std[i] * scipy.stats.t.ppf((1 + p) / 2., n[i] -1) + np.nanmean(luh2_c4crops,axis=(1,2))[i]*uncert
    
n = [0 for i in range(35)]
F4_std_crops = [0 for i in range(35)]
F4_ci_crops = [0 for i in range(35)]

for i in range(35):
    n[i] = np.count_nonzero(~np.isnan(F4_withcrops[i,:,:]))
    F4_std_crops[i] = np.nanstd(F4_withcrops,axis=(1,2))[i]/ np.sqrt(n[i])
    F4_ci_crops[i] = F4_std_crops[i] * scipy.stats.t.ppf((1 + p) / 2., n[i] -1) + np.nanmean(F4_withcrops,axis=(1,2))[i]*uncert


# In[13]:


## Weighted by gridcell area & Conversion gC m-2 yr-1 to PgC yr-1

ds5 = netCDF4.Dataset('/Users/alienorlavergne/Desktop/Research and Teaching/Data/watch_wfdei/gridarea_0.5.nc')
ds5.set_auto_mask(False)
gridarea = ds5['cell_area'][:]
ds5.close()

gpp_tot_final = gpp_tot*gridarea*1e-15
gpp_c3_final = gpp_c3*gridarea*1e-15
gpp_c4_final = gpp_c4*gridarea*1e-15

gpp_tot_final_crops = gpp_tot_crops*gridarea*1e-15
gpp_c3_final_crops = gpp_c3_crops*gridarea*1e-15
gpp_c4_final_crops = gpp_c4_crops*gridarea*1e-15

gpp_tot_final_still = gpp_tot_still*gridarea*1e-15
gpp_c3_final_still = gpp_c3_still*gridarea*1e-15
gpp_c4_final_still = gpp_c4_still*gridarea*1e-15

gpp_tot_final_luo = gpp_tot_luo*gridarea*1e-15
gpp_c3_final_luo = gpp_c3_luo*gridarea*1e-15
gpp_c4_final_luo = gpp_c4_luo*gridarea*1e-15


## Global temporal variations

gpp_tot_final_all = np.nansum(gpp_tot_final,axis=(1,2))
gpp_c3_final_all = np.nansum(gpp_c3_final,axis=(1,2))
gpp_c4_final_all = np.nansum(gpp_c4_final,axis=(1,2))

gpp_tot_final_all_crops = np.nansum(gpp_tot_final_crops,axis=(1,2))
gpp_c3_final_all_crops = np.nansum(gpp_c3_final_crops,axis=(1,2))
gpp_c4_final_all_crops = np.nansum(gpp_c4_final_crops,axis=(1,2))

gpp_tot_final_all_still = np.nansum(gpp_tot_final_still,axis=(1,2))
gpp_c3_final_all_still = np.nansum(gpp_c3_final_still,axis=(1,2))
gpp_c4_final_all_still = np.nansum(gpp_c4_final_still,axis=(1,2))


gpp_tot_final_all_luo = np.nansum(gpp_tot_final_luo,axis=(1,2))
gpp_c3_final_all_luo = np.nansum(gpp_c3_final_luo,axis=(1,2))
gpp_c4_final_all_luo = np.nansum(gpp_c4_final_luo,axis=(1,2))


## Standard error for sum: sqrt(Sample Size) * Standard Deviation

## Standard error and confidence interval at 95%

uncert = 2/100


n = [0 for i in range(35)]
gpp_tot_final_all_std = [0 for i in range(35)]
gpp_tot_final_all_ci = [0 for i in range(35)]
gpp_c3_final_all_std = [0 for i in range(35)]
gpp_c3_final_all_ci = [0 for i in range(35)]
gpp_c4_final_all_std = [0 for i in range(35)]
gpp_c4_final_all_ci = [0 for i in range(35)]

for i in range(35):
    n[i] = np.count_nonzero(~np.isnan(gpp_tot_final[i,:,:]))
    gpp_tot_final_all_std[i] = np.nanstd(gpp_tot_final,axis=(1,2))[i]* np.sqrt(n[i])
    gpp_tot_final_all_ci[i] = gpp_tot_final_all_std[i] * scipy.stats.t.ppf((1 + 0.95) / 2., n[i] -1) + np.nansum(gpp_tot_final,axis=(1,2))[i]*uncert
    gpp_c3_final_all_std[i] = np.nanstd(gpp_c3_final,axis=(1,2))[i]* np.sqrt(n[i])
    gpp_c3_final_all_ci[i] = gpp_c3_final_all_std[i] * scipy.stats.t.ppf((1 + 0.95) / 2., n[i] -1) + np.nansum(gpp_c3_final,axis=(1,2))[i]*uncert
    gpp_c4_final_all_std[i] = np.nanstd(gpp_c4_final,axis=(1,2))[i]* np.sqrt(n[i])
    gpp_c4_final_all_ci[i] = gpp_c4_final_all_std[i] * scipy.stats.t.ppf((1 + 0.95) / 2., n[i] -1) + np.nansum(gpp_c4_final,axis=(1,2))[i]*uncert

    
n = [0 for i in range(35)]
gpp_tot_final_all_std_crops = [0 for i in range(35)]
gpp_tot_final_all_ci_crops = [0 for i in range(35)]
gpp_c3_final_all_std_crops = [0 for i in range(35)]
gpp_c3_final_all_ci_crops = [0 for i in range(35)]
gpp_c4_final_all_std_crops = [0 for i in range(35)]
gpp_c4_final_all_ci_crops = [0 for i in range(35)]

for i in range(35):
    n[i] = np.count_nonzero(~np.isnan(gpp_tot_final_crops[i,:,:]))
    gpp_tot_final_all_std_crops[i] = np.nanstd(gpp_tot_final_crops,axis=(1,2))[i]* np.sqrt(n[i])
    gpp_tot_final_all_ci_crops[i] = gpp_tot_final_all_std_crops[i] * scipy.stats.t.ppf((1 + 0.95) / 2., n[i] -1) + np.nansum(gpp_tot_final_crops,axis=(1,2))[i]*uncert
    gpp_c3_final_all_std_crops[i] = np.nanstd(gpp_c3_final_crops,axis=(1,2))[i]* np.sqrt(n[i])
    gpp_c3_final_all_ci_crops[i] = gpp_c3_final_all_std_crops[i] * scipy.stats.t.ppf((1 + 0.95) / 2., n[i] -1) + np.nansum(gpp_c3_final_crops,axis=(1,2))[i]*uncert
    gpp_c4_final_all_std_crops[i] = np.nanstd(gpp_c4_final_crops,axis=(1,2))[i]* np.sqrt(n[i])
    gpp_c4_final_all_ci_crops[i] = gpp_c4_final_all_std_crops[i] * scipy.stats.t.ppf((1 + 0.95) / 2., n[i] -1) + np.nansum(gpp_c4_final_crops,axis=(1,2))[i]*uncert


n = [0 for i in range(35)]
gpp_tot_final_all_still_std = [0 for i in range(35)]
gpp_tot_final_all_still_ci = [0 for i in range(35)]
gpp_c3_final_all_still_std = [0 for i in range(35)]
gpp_c3_final_all_still_ci = [0 for i in range(35)]
gpp_c4_final_all_still_std = [0 for i in range(35)]
gpp_c4_final_all_still_ci = [0 for i in range(35)]

for i in range(35):
    n[i] = np.count_nonzero(~np.isnan(gpp_tot_final_still[i,:,:]))
    gpp_tot_final_all_still_std[i] = np.nanstd(gpp_tot_final_still,axis=(1,2))[i]* np.sqrt(n[i])
    gpp_tot_final_all_still_ci[i] = gpp_tot_final_all_still_std[i] * scipy.stats.t.ppf((1 + 0.95) / 2., n[i] -1) + np.nansum(gpp_tot_final_still,axis=(1,2))[i]*uncert
    gpp_c3_final_all_still_std[i] = np.nanstd(gpp_c3_final_still,axis=(1,2))[i]* np.sqrt(n[i])
    gpp_c3_final_all_still_ci[i] = gpp_c3_final_all_still_std[i] * scipy.stats.t.ppf((1 + 0.95) / 2., n[i] -1) + np.nansum(gpp_c3_final_still,axis=(1,2))[i]*uncert
    gpp_c4_final_all_still_std[i] = np.nanstd(gpp_c4_final_still,axis=(1,2))[i]* np.sqrt(n[i])
    gpp_c4_final_all_still_ci[i] = gpp_c4_final_all_still_std[i] * scipy.stats.t.ppf((1 + 0.95) / 2., n[i] -1) + np.nansum(gpp_c4_final_still,axis=(1,2))[i]*uncert


n = [0 for i in range(16)]
gpp_tot_final_all_luo_std = [0 for i in range(16)]
gpp_tot_final_all_luo_ci = [0 for i in range(16)]
gpp_c3_final_all_luo_std = [0 for i in range(16)]
gpp_c3_final_all_luo_ci = [0 for i in range(16)]
gpp_c4_final_all_luo_std = [0 for i in range(16)]
gpp_c4_final_all_luo_ci = [0 for i in range(16)]

for i in range(16):
    n[i] = np.count_nonzero(~np.isnan(gpp_tot_final_luo[i,:,:]))
    gpp_tot_final_all_luo_std[i] = np.nanstd(gpp_tot_final_luo,axis=(1,2))[i]* np.sqrt(n[i])
    gpp_tot_final_all_luo_ci[i] = gpp_tot_final_all_luo_std[i] * scipy.stats.t.ppf((1 + 0.95) / 2., n[i] -1) + np.nansum(gpp_tot_final_luo,axis=(1,2))[i]*uncert
    gpp_c3_final_all_luo_std[i] = np.nanstd(gpp_c3_final_luo,axis=(1,2))[i]* np.sqrt(n[i])
    gpp_c3_final_all_luo_ci[i] = gpp_c3_final_all_luo_std[i] * scipy.stats.t.ppf((1 + 0.95) / 2., n[i] -1) + np.nansum(gpp_c3_final_luo,axis=(1,2))[i]*uncert
    gpp_c4_final_all_luo_std[i] = np.nanstd(gpp_c4_final_luo,axis=(1,2))[i]* np.sqrt(n[i])
    gpp_c4_final_all_luo_ci[i] = gpp_c4_final_all_luo_std[i] * scipy.stats.t.ppf((1 + 0.95) / 2., n[i] -1) + np.nansum(gpp_c4_final_luo,axis=(1,2))[i]*uncert


# In[14]:


D13Cplant_C3_yr = np.nanmean(D13Cplant_C3,axis=(1,2))
D13Cplant_C4_yr = np.nanmean(D13Cplant_C4,axis=(1,2))

D13C_c3_herb_yr = np.nanmean(D13C_c3_herb,axis=(1,2))
D13C_c3_woody_yr = np.nanmean(D13C_c3_woody,axis=(1,2))
D13C_c4_herb_yr = np.nanmean(D13C_c4_herb,axis=(1,2))
D13C_tot_yr = np.nanmean(D13C_tot,axis=(1,2))


# standard error of the mean: np.std(data)/np.sqrt(np.size(data))
# confidence interval at 95%

uncert = 2/100


## natural
n = [0 for i in range(35)]
D13C_tot_yr_std = [0 for i in range(35)]
D13C_tot_yr_ci = [0 for i in range(35)]
D13C_c4_herb_yr_std = [0 for i in range(35)]
D13C_c4_herb_yr_ci = [0 for i in range(35)]

D13C_c3_herb_yr_std = [0 for i in range(35)]
D13C_c3_herb_yr_ci = [0 for i in range(35)]
D13C_c3_woody_yr_std = [0 for i in range(35)]
D13C_c3_woody_yr_ci = [0 for i in range(35)]


for i in range(35):
    n[i] = np.count_nonzero(~np.isnan(D13C_tot[i,:,:]))
    D13C_tot_yr_std[i] = np.nanstd(D13C_tot,axis=(1,2))[i]/ np.sqrt(n[i])
    D13C_tot_yr_ci[i] = D13C_tot_yr_std[i] * scipy.stats.t.ppf((1 + 0.95) / 2., n[i] -1) + np.nanmean(D13C_tot,axis=(1,2))[i]*uncert
    D13C_c4_herb_yr_std[i] = np.nanstd(D13C_c4_herb,axis=(1,2))[i]/ np.sqrt(n[i])
    D13C_c4_herb_yr_ci[i] = D13C_c4_herb_yr_std[i] * scipy.stats.t.ppf((1 + 0.95) / 2., n[i] -1) + np.nanmean(D13C_c4_herb,axis=(1,2))[i]*uncert
    D13C_c3_herb_yr_std[i] = np.nanstd(D13C_c3_herb,axis=(1,2))[i]/ np.sqrt(n[i])
    D13C_c3_herb_yr_ci[i] = D13C_c3_herb_yr_std[i] * scipy.stats.t.ppf((1 + 0.95) / 2., n[i] -1) + np.nanmean(D13C_c3_herb,axis=(1,2))[i]*uncert
    D13C_c3_woody_yr_std[i] = np.nanstd(D13C_c3_woody,axis=(1,2))[i]/ np.sqrt(n[i])
    D13C_c3_woody_yr_ci[i] = D13C_c3_woody_yr_std[i] * scipy.stats.t.ppf((1 + 0.95) / 2., n[i] -1) + np.nanmean(D13C_c3_woody,axis=(1,2))[i]*uncert


## with crops
D13C_c3_herb_yr_crops = np.nanmean(D13C_c3_herb_crops,axis=(1,2))
D13C_c3_woody_yr_crops = np.nanmean(D13C_c3_woody_crops,axis=(1,2))
D13C_c4_herb_yr_crops = np.nanmean(D13C_c4_herb_crops,axis=(1,2))
D13C_tot_yr_crops = np.nanmean(D13C_tot_crops,axis=(1,2))


D13C_tot_yr_crops_std = [0 for i in range(35)]
D13C_tot_yr_crops_ci = [0 for i in range(35)]
D13C_c4_herb_yr_crops_std = [0 for i in range(35)]
D13C_c4_herb_yr_crops_ci = [0 for i in range(35)]

D13C_c3_herb_yr_crops_std = [0 for i in range(35)]
D13C_c3_herb_yr_crops_ci = [0 for i in range(35)]
D13C_c3_woody_yr_crops_std = [0 for i in range(35)]
D13C_c3_woody_yr_crops_ci = [0 for i in range(35)]


for i in range(35):
    n[i] = np.count_nonzero(~np.isnan(D13C_tot_crops[i,:,:]))
    D13C_tot_yr_crops_std[i] = np.nanstd(D13C_tot_crops,axis=(1,2))[i]/ np.sqrt(n[i])
    D13C_tot_yr_crops_ci[i] = D13C_tot_yr_crops_std[i] * scipy.stats.t.ppf((1 + 0.95) / 2., n[i] -1) + np.nanmean(D13C_tot_crops,axis=(1,2))[i]*uncert
    D13C_c4_herb_yr_crops_std[i] = np.nanstd(D13C_c4_herb_crops,axis=(1,2))[i]/ np.sqrt(n[i])
    D13C_c4_herb_yr_crops_ci[i] = D13C_c4_herb_yr_crops_std[i] * scipy.stats.t.ppf((1 + 0.95) / 2., n[i] -1) + np.nanmean(D13C_c4_herb_crops,axis=(1,2))[i]*uncert
    D13C_c3_herb_yr_crops_std[i] = np.nanstd(D13C_c3_herb_crops,axis=(1,2))[i]/ np.sqrt(n[i])
    D13C_c3_herb_yr_crops_ci[i] = D13C_c3_herb_yr_crops_std[i] * scipy.stats.t.ppf((1 + 0.95) / 2., n[i] -1) + np.nanmean(D13C_c3_herb_crops,axis=(1,2))[i]*uncert
    D13C_c3_woody_yr_crops_std[i] = np.nanstd(D13C_c3_woody_crops,axis=(1,2))[i]/ np.sqrt(n[i])
    D13C_c3_woody_yr_crops_ci[i] = D13C_c3_woody_yr_crops_std[i] * scipy.stats.t.ppf((1 + 0.95) / 2., n[i] -1) + np.nanmean(D13C_c3_woody_crops,axis=(1,2))[i]*uncert


## still

D13C_tot_still_yr = np.nanmean(D13C_tot_still,axis=(1,2))
D13C_c4_herb_still_yr = np.nanmean(D13C_c4_herb_still,axis=(1,2))
D13C_c3_herb_still_yr = np.nanmean(D13C_c3_herb_still,axis=(1,2))
D13C_c3_woody_still_yr = np.nanmean(D13C_c3_woody_still,axis=(1,2))


n = [0 for i in range(35)]
D13C_tot_still_yr_std = [0 for i in range(35)]
D13C_tot_still_yr_ci = [0 for i in range(35)]
D13C_c4_herb_still_yr_std = [0 for i in range(35)]
D13C_c4_herb_still_yr_ci = [0 for i in range(35)]
D13C_c3_herb_still_yr_std = [0 for i in range(35)]
D13C_c3_herb_still_yr_ci = [0 for i in range(35)]
D13C_c3_woody_still_yr_std = [0 for i in range(35)]
D13C_c3_woody_still_yr_ci = [0 for i in range(35)]



for i in range(35):
    n[i] = np.count_nonzero(~np.isnan(D13C_tot_still[i,:,:]))
    D13C_tot_still_yr_std[i] = np.nanstd(D13C_tot_still,axis=(1,2))[i]/ np.sqrt(n[i])
    D13C_tot_still_yr_ci[i] = D13C_tot_still_yr_std[i] * scipy.stats.t.ppf((1 + 0.95) / 2., n[i] -1) + np.nanmean(D13C_tot_still,axis=(1,2))[i]*uncert
    D13C_c4_herb_still_yr_std[i] = np.nanstd(D13C_c4_herb_still,axis=(1,2))[i]/ np.sqrt(n[i])
    D13C_c4_herb_still_yr_ci[i] = D13C_c4_herb_still_yr_std[i] * scipy.stats.t.ppf((1 + 0.95) / 2., n[i] -1) + np.nanmean(D13C_c4_herb_still,axis=(1,2))[i]*uncert
    D13C_c3_herb_still_yr_std[i] = np.nanstd(D13C_c3_herb_still,axis=(1,2))[i]/ np.sqrt(n[i])
    D13C_c3_herb_still_yr_ci[i] = D13C_c3_herb_still_yr_std[i] * scipy.stats.t.ppf((1 + 0.95) / 2., n[i] -1) + np.nanmean(D13C_c3_herb_still,axis=(1,2))[i]*uncert
    D13C_c3_woody_still_yr_std[i] = np.nanstd(D13C_c3_woody_still,axis=(1,2))[i]/ np.sqrt(n[i])
    D13C_c3_woody_still_yr_ci[i] = D13C_c3_woody_still_yr_std[i] * scipy.stats.t.ppf((1 + 0.95) / 2., n[i] -1) + np.nanmean(D13C_c3_woody_still,axis=(1,2))[i]*uncert


    
## Luo

D13C_tot_luo_yr = np.nanmean(D13C_tot_luo,axis=(1,2))
D13C_c4_herb_luo_yr = np.nanmean(D13C_c4_herb_luo,axis=(1,2))
D13C_c3_herb_luo_yr = np.nanmean(D13C_c3_herb_luo,axis=(1,2))
D13C_c3_woody_luo_yr = np.nanmean(D13C_c3_woody_luo,axis=(1,2))


n = [0 for i in range(16)]
D13C_tot_luo_yr_std = [0 for i in range(16)]
D13C_tot_luo_yr_ci = [0 for i in range(16)]
D13C_c4_herb_luo_yr_std = [0 for i in range(16)]
D13C_c4_herb_luo_yr_ci = [0 for i in range(16)]
D13C_c3_herb_luo_yr_std = [0 for i in range(16)]
D13C_c3_herb_luo_yr_ci = [0 for i in range(16)]
D13C_c3_woody_luo_yr_std = [0 for i in range(16)]
D13C_c3_woody_luo_yr_ci = [0 for i in range(16)]

for i in range(16):
    n[i] = np.count_nonzero(~np.isnan(D13C_tot_luo[i,:,:]))
    D13C_tot_luo_yr_std[i] = np.nanstd(D13C_tot_luo,axis=(1,2))[i]/ np.sqrt(n[i])
    D13C_tot_luo_yr_ci[i] = D13C_tot_luo_yr_std[i] * scipy.stats.t.ppf((1 + 0.95) / 2., n[i] -1) + np.nanmean(D13C_tot_luo,axis=(1,2))[i]*uncert
    D13C_c4_herb_luo_yr_std[i] = np.nanstd(D13C_c4_herb_luo,axis=(1,2))[i]/ np.sqrt(n[i])
    D13C_c4_herb_luo_yr_ci[i] = D13C_c4_herb_luo_yr_std[i] * scipy.stats.t.ppf((1 + 0.95) / 2., n[i] -1) + np.nanmean(D13C_c4_herb_luo,axis=(1,2))[i]*uncert
    D13C_c3_herb_luo_yr_std[i] = np.nanstd(D13C_c3_herb_luo,axis=(1,2))[i]/ np.sqrt(n[i])
    D13C_c3_herb_luo_yr_ci[i] = D13C_c3_herb_luo_yr_std[i] * scipy.stats.t.ppf((1 + 0.95) / 2., n[i] -1) + np.nanmean(D13C_c3_herb_luo,axis=(1,2))[i]*uncert
    D13C_c3_woody_luo_yr_std[i] = np.nanstd(D13C_c3_woody_luo,axis=(1,2))[i]/ np.sqrt(n[i])
    D13C_c3_woody_luo_yr_ci[i] = D13C_c3_woody_luo_yr_std[i] * scipy.stats.t.ppf((1 + 0.95) / 2., n[i] -1) + np.nanmean(D13C_c3_woody_luo,axis=(1,2))[i]*uncert


# In[16]:


## Figure 4

## Upload Keeling et al. data

Keeling_data = pd.read_csv('~/Data/Isotopes/D13Catm/Keeling2017data.csv', header=0, names=['Year', 'D13Catm']) #, index_col=0, na_values=['(NA)'])

year = np.arange(0,35)+1982

# Set up the subplot figure

fig_figure4 = plt.figure(1, figsize=(10,14))

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


## F4

# Figure a
gs = gspec.GridSpec(5, 3, figure=fig_figure4, hspace=0.3,  width_ratios=[1,0,1],height_ratios=[0.3,0.2,0.2,0.3,0.2])

column = 0
row = 0

ax = fig_figure4.add_subplot(gs[row, column])
        
ax.plot(year,np.nanmean(F4_natural,axis=(1,2)),color=colourWheel[0],lw=2.5)
ax.plot(year,np.nanmean(luh2_c4crops,axis=(1,2)),color=colourWheel[7],lw=2.5)
ax.plot(year,np.nanmean(F4_withcrops,axis=(1,2)),color=colourWheel[9],lw=2.5)
ax.plot(year,np.repeat(np.nanmean(C4_frac_still),len(year))*100,colourWheel[12],lw=2.5) #,ls="dotted")
ax.plot(year[19:35],np.nanmean(C4_frac_luo,axis=(1,2))*100,colourWheel[4],lw=2.5) #,ls="dashed")


error = 0

under_line     = (np.nanmean(F4_natural,axis=(1,2))-F4_natural_ci -error) #[0]
over_line      = (np.nanmean(F4_natural,axis=(1,2))+F4_natural_ci+error) #[0]
ax.fill_between(year, under_line, over_line, color=colourWheel[0],alpha=.1) #std curves

under_line     = (np.nanmean(luh2_c4crops,axis=(1,2))[0:34]-luh2_c4crops_ci[0:34] - np.nanmean(luh2_c4crops[0:34],axis=(1,2))*10/100) #[0]
over_line      = (np.nanmean(luh2_c4crops,axis=(1,2))[0:34]+luh2_c4crops_ci[0:34] + np.nanmean(luh2_c4crops[0:34],axis=(1,2))*10/100) #[0]
ax.fill_between(year[0:34], under_line, over_line,color=colourWheel[7], alpha=.1) #std curves

under_line     = (np.nanmean(F4_withcrops,axis=(1,2))-F4_ci_crops-error) #[0]
over_line      = (np.nanmean(F4_withcrops,axis=(1,2))+F4_ci_crops+error) #[0]
ax.fill_between(year, under_line, over_line,color=colourWheel[9], alpha=.1) #std curves


ax.set_ylabel(u'$F_{4}$ (-)',fontsize=14)
ax.set_ylim((0,0.18))
ax.set_xlim((1981,2017))
ax.text(0.05, 0.99, '(a)',transform=ax.transAxes,va = 'top',fontsize=14)

ax.set_xticks([1985,1990,1995,2000,2005,2010,2015],fontsize=12)
ax.set_yticks([0,0.05, 0.10, 0.15,0.20],fontsize=12)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

ax.tick_params(labelsize=12)


# Figure b
column = 2
row = 0

ax = fig_figure4.add_subplot(gs[row, column])

ax.plot(year,np.repeat(np.nanmean(C4_frac_still),len(year)),color=colourWheel[12],lw=2.5) #,ls="dotted")
ax.plot(year[19:35],np.nanmean(C4_frac_luo,axis=(1,2)),color=colourWheel[4],lw=2.5) #,ls="dashed")
ax.plot(year,np.nanmean(F4_withcrops,axis=(1,2)),color=colourWheel[9],lw=2.5)


error = 0

under_line     = (np.nanmean(C4_frac_luo,axis=(1,2))-F4_luo_ci-error) #[0]
over_line      = (np.nanmean(C4_frac_luo,axis=(1,2))+F4_luo_ci+error) #[0]
ax.fill_between(year[19:35], under_line, over_line, color=colourWheel[4], alpha=.1) #std curves

under_line     = (np.nanmean(luh2_c4crops,axis=(1,2))[0:34] - np.nanstd(luh2_c4crops,axis=(1,2))[0:34]*10/100) #[0]
over_line      = (np.nanmean(luh2_c4crops,axis=(1,2))[0:34] + np.nanstd(luh2_c4crops,axis=(1,2))[0:34]*10/100) #[0]
ax.fill_between(year[0:34], under_line, over_line, color=colourWheel[7], alpha=.1) #std curves

under_line     = (np.nanmean(F4_withcrops,axis=(1,2))-F4_ci_crops-error) #[0]
over_line      = (np.nanmean(F4_withcrops,axis=(1,2))+F4_ci_crops+error) #[0]
ax.fill_between(year, under_line, over_line, color=colourWheel[9], alpha=.1) #std curves


ax.set_ylabel(u'$F_{4}$ (-)',fontsize=14)
ax.set_ylim((0.1125,0.17))
ax.set_xlim((1981,2017))
ax.text(0.05, 0.99, '(b)',transform=ax.transAxes,va = 'top',fontsize=14)

ax.set_xticks([1985,1990,1995,2000,2005,2010,2015],fontsize=12)
ax.set_yticks([0.12, 0.13, 0.14, 0.15, 0.16, 0.17],fontsize=12)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

ax.tick_params(labelsize=12)


## GPP

# Figure c
row = 1
column = 0

ax = fig_figure4.add_subplot(gs[row, column])

ax.plot(year,gpp_c3_final_all_crops,color=colourWheel[9])
ax.plot(year,gpp_c3_final_all,color=colourWheel[0])
ax.plot(year,gpp_c3_final_all_still,color=colourWheel[12])
ax.plot(year[19:35],gpp_c3_final_all_luo,color=colourWheel[4])

error = 0

under_line     = (gpp_c3_final_all_still-gpp_c3_final_all_still_ci-error) #[0]
over_line      = (gpp_c3_final_all_still+gpp_c3_final_all_still_ci+error) #[0]
ax.fill_between(year, under_line, over_line, color=colourWheel[12], alpha=.1) #std curves

under_line     = (gpp_c3_final_all_luo-gpp_c3_final_all_luo_ci-error) #[0]
over_line      = (gpp_c3_final_all_luo+gpp_c3_final_all_luo_ci+error) #[0]
ax.fill_between(year[19:35], under_line, over_line, color=colourWheel[4], alpha=.1) #std curves

under_line     = (gpp_c3_final_all-gpp_c3_final_all_ci-error) #[0]
over_line      = (gpp_c3_final_all+gpp_c3_final_all_ci+error) #[0]
ax.fill_between(year, under_line, over_line, color=colourWheel[0], alpha=.1) #std curves

under_line     = (gpp_c3_final_all_crops-gpp_c3_final_all_ci_crops-error) #[0]
over_line      = (gpp_c3_final_all_crops+gpp_c3_final_all_ci_crops+error) #[0]
ax.fill_between(year, under_line, over_line, color=colourWheel[9], alpha=.1) #std curves

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_ylabel(u'GPP$_{C3}$ $F_{3}$ (PgC yr$^{-1}$)',fontsize=14)

ax.set_ylim((100,150))

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.set_xticks([1985,1990,1995,2000,2005,2010,2015],fontsize=12)

ax.set_xlim((1981,2017))
ax.text(0.05, 1, '(c)',transform=ax.transAxes,va = 'top',fontsize=14)

ax.tick_params(labelsize=12)


# Figure c
row = 2
column = 0

ax = fig_figure4.add_subplot(gs[row, column])
ax.plot(year,gpp_c4_final_all_crops,color=colourWheel[9])
ax.plot(year,gpp_c4_final_all,color=colourWheel[0])
ax.plot(year,gpp_c4_final_all_still,color=colourWheel[12])
ax.plot(year[19:35],gpp_c4_final_all_luo,color=colourWheel[4])

error = 0

under_line     = (gpp_c4_final_all_still-gpp_c4_final_all_still_ci-error) #[0]
over_line      = (gpp_c4_final_all_still+gpp_c4_final_all_still_ci+error) #[0]
ax.fill_between(year, under_line, over_line, color=colourWheel[12], alpha=.1) #std curves

under_line     = (gpp_c4_final_all_luo-gpp_c4_final_all_luo_ci-error) #[0]
over_line      = (gpp_c4_final_all_luo+gpp_c4_final_all_luo_ci+error) #[0]
ax.fill_between(year[19:35], under_line, over_line, color=colourWheel[4], alpha=.1) #std curves

under_line     = (gpp_c4_final_all-gpp_c4_final_all_ci-error) #[0]
over_line      = (gpp_c4_final_all+gpp_c4_final_all_ci+error) #[0]
ax.fill_between(year, under_line, over_line, color=colourWheel[0], alpha=.1) #std curves

under_line     = (gpp_c4_final_all_crops-gpp_c4_final_all_ci_crops-error) #[0]
over_line      = (gpp_c4_final_all_crops+gpp_c4_final_all_ci_crops+error) #[0]
ax.fill_between(year, under_line, over_line, color=colourWheel[9], alpha=.1) #std curves

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_ylabel(u'GPP$_{C4}$ $F_{4}$ (PgC yr$^{-1}$)',fontsize=14)

ax.set_ylim((20,55))
ax.set_yticks([20,30,40,50])

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.set_xticks([1985,1990,1995,2000,2005,2010,2015],fontsize=12)

ax.set_xlim((1981,2017))

ax.tick_params(labelsize=12)


# Figure d
row = 3
column = 0

ax = fig_figure4.add_subplot(gs[row, column])

ax.plot(year,np.nanmean(F4_natural,axis=(1,2)),color=colourWheel[0],lw=2.5)
ax.plot(year,np.nanmean(luh2_c4crops,axis=(1,2)),color=colourWheel[7],lw=2.5)
ax.plot(year,np.nanmean(F4_withcrops,axis=(1,2)),color=colourWheel[9],lw=2.5)
ax.plot(year,np.repeat(np.nanmean(C4_frac_still),len(year))*100,colourWheel[12],lw=2.5) #,ls="dotted")
ax.plot(year[19:35],np.nanmean(C4_frac_luo,axis=(1,2))*100,colourWheel[4],lw=2.5) #,ls="dashed")

ax.legend(['natural grasslands','crops','natural and crops','Still2009','Luo2024'], fontsize=12, loc='lower left', bbox_to_anchor= (-0.05, -0.35), ncol=5,
            borderaxespad=0, frameon=False)

ax.plot(year,gpp_tot_final_all_crops,color=colourWheel[9])
ax.plot(year,gpp_tot_final_all_still,color=colourWheel[12]) #,linestyle="dotted")
ax.plot(year[19:35],gpp_tot_final_all_luo,color=colourWheel[4]) #,linestyle="dashed")

error = 0

under_line     = (gpp_tot_final_all_still-gpp_tot_final_all_still_ci-error) #[0]
over_line      = (gpp_tot_final_all_still+gpp_tot_final_all_still_ci+error) #[0]
ax.fill_between(year, under_line, over_line, color=colourWheel[12], alpha=.1) #std curves

under_line     = (gpp_tot_final_all_luo-gpp_tot_final_all_luo_ci-error) #[0]
over_line      = (gpp_tot_final_all_luo+gpp_tot_final_all_luo_ci+error) #[0]
ax.fill_between(year[19:35], under_line, over_line, color=colourWheel[4], alpha=.1) #std curves

under_line     = (gpp_tot_final_all_crops-gpp_tot_final_all_ci_crops-error) #[0]
over_line      = (gpp_tot_final_all_crops+gpp_tot_final_all_ci_crops+error) #[0]
ax.fill_between(year, under_line, over_line, color=colourWheel[9], alpha=.1) #std curves


ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_ylabel(u'GPP (PgC yr$^{-1}$)',fontsize=14)

ax.set_ylim((140,180))

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.set_xticks([1985,1990,1995,2000,2005,2010,2015],fontsize=12)

ax.set_xlim((1981,2017))
ax.text(0.05, 0.99, '(d)',transform=ax.transAxes,va = 'top',fontsize=14)

ax.tick_params(labelsize=12)


## D13C

k = 1

# Figure e
row = 1
column = 2

ax = fig_figure4.add_subplot(gs[row, column])
ax.plot(year,D13C_c3_herb_still_yr*k,color=colourWheel[12])
ax.plot(year[19:35],D13C_c3_herb_luo_yr*k,color=colourWheel[4])
ax.plot(year,D13C_c3_herb_yr_crops*k,color=colourWheel[9])
ax.plot(year,D13C_c3_herb_yr*k,color=colourWheel[0])

error = 0

under_line     = (D13C_c3_herb_still_yr-D13C_c3_herb_still_yr_ci-error)*k #[0]
over_line      = (D13C_c3_herb_still_yr+D13C_c3_herb_still_yr_ci+error)*k #[0]
ax.fill_between(year, under_line, over_line, color=colourWheel[12], alpha=.1) #std curves

under_line     = (D13C_c3_herb_luo_yr-D13C_c3_herb_luo_yr_ci-error)*k #[0]
over_line      = (D13C_c3_herb_luo_yr+D13C_c3_herb_luo_yr_ci+error)*k #[0]
ax.fill_between(year[19:35], under_line, over_line, color=colourWheel[4], alpha=.1) #std curves

under_line     = (D13C_c3_herb_yr-D13C_c3_herb_yr_ci-error)*k #[0]
over_line      = (D13C_c3_herb_yr+D13C_c3_herb_yr_ci+error)*k #[0]
ax.fill_between(year, under_line, over_line, color=colourWheel[0], alpha=.1) #std curves

under_line     = (D13C_c3_herb_yr_crops-D13C_c3_herb_yr_crops_ci-error)*k #[0]
over_line      = (D13C_c3_herb_yr_crops+D13C_c3_herb_yr_crops_ci+error)*k #[0]
ax.fill_between(year, under_line, over_line, color=colourWheel[9], alpha=.1) #std curves

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_ylabel(u'$\mathregular{\u0394}^{13}$C$_{C3}$ $f_{3}$ (‰)',fontsize=14)

ax.set_ylim((14.5,17))
ax.set_yticks([15,16,17])

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.set_xticks([1985,1990,1995,2000,2005,2010,2015],fontsize=12)

ax.set_xlim((1981,2017))
ax.text(0.05, 0.99, '(e)',transform=ax.transAxes,va = 'top',fontsize=14)

ax.tick_params(labelsize=12)


# Figure e
row = 2
column = 2

ax = fig_figure4.add_subplot(gs[row, column])

ax.plot(year,D13C_c4_herb_still_yr*k,color=colourWheel[12])
ax.plot(year[19:35],D13C_c4_herb_luo_yr*k,color=colourWheel[4])
ax.plot(year,D13C_c4_herb_yr_crops*k,color=colourWheel[9])
ax.plot(year,D13C_c4_herb_yr*k,color=colourWheel[0])

error = 0

under_line     = (D13C_c4_herb_still_yr-D13C_c4_herb_still_yr_ci-error)*k #[0]
over_line      = (D13C_c4_herb_still_yr+D13C_c4_herb_still_yr_ci+error)*k #[0]
ax.fill_between(year, under_line, over_line, color=colourWheel[12], alpha=.1) #std curves

under_line     = (D13C_c4_herb_luo_yr-D13C_c4_herb_luo_yr_ci-error)*k #[0]
over_line      = (D13C_c4_herb_luo_yr+D13C_c4_herb_luo_yr_ci+error)*k #[0]
ax.fill_between(year[19:35], under_line, over_line, color=colourWheel[4], alpha=.1) #std curves

under_line     = (D13C_c4_herb_yr-D13C_c4_herb_yr_ci-error)*k #[0]
over_line      = (D13C_c4_herb_yr+D13C_c4_herb_yr_ci+error)*k #[0]
ax.fill_between(year, under_line, over_line, color=colourWheel[0], alpha=.1) #std curves

under_line     = (D13C_c4_herb_yr_crops-D13C_c4_herb_yr_crops_ci-error)*k #[0]
over_line      = (D13C_c4_herb_yr_crops+D13C_c4_herb_yr_crops_ci+error)*k #[0]
ax.fill_between(year, under_line, over_line, color=colourWheel[9], alpha=.1) #std curves


ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_ylabel(u'$\mathregular{\u0394}^{13}$C$_{C4}$ $f_{4}$ (‰)',fontsize=14)

ax.set_ylim((0.65,1))
ax.set_yticks([0.7,0.8,0.9,1])

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.set_xticks([1985,1990,1995,2000,2005,2010,2015],fontsize=12)

ax.set_xlim((1981,2017))

ax.tick_params(labelsize=12)



# Figure f
row = 3
column = 2

ax = fig_figure4.add_subplot(gs[row, column])

ax.plot(year,D13C_tot_still_yr*k,color=colourWheel[12])
ax.plot(year[19:35],D13C_tot_luo_yr*k,color=colourWheel[4])
ax.plot(year,D13C_tot_yr_crops*k,color=colourWheel[9])



error = 0

under_line     = (D13C_tot_still_yr-D13C_tot_still_yr_ci-error)*k #[0]
over_line      = (D13C_tot_still_yr+D13C_tot_still_yr_ci+error)*k #[0]
ax.fill_between(year, under_line, over_line, color=colourWheel[12], alpha=.1) #std curves

under_line     = (D13C_tot_luo_yr-D13C_tot_luo_yr_ci-error)*k #[0]
over_line      = (D13C_tot_luo_yr+D13C_tot_luo_yr_ci+error)*k #[0]
ax.fill_between(year[19:35], under_line, over_line, color=colourWheel[4], alpha=.1) #std curves

under_line     = (D13C_tot_yr_crops-D13C_tot_yr_crops_ci-error)*k #[0]
over_line      = (D13C_tot_yr_crops+D13C_tot_yr_crops_ci+error)*k #[0]
ax.fill_between(year, under_line, over_line, color=colourWheel[9], alpha=.1) #std curves


ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_ylabel(u'$\mathregular{\u0394}^{13}$C (‰)',fontsize=14)

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.set_xticks([1985,1990,1995,2000,2005,2010,2015],fontsize=12)

ax.set_xlim((1981,2017))
ax.text(0.05, 0.99, '(f)',transform=ax.transAxes,va = 'top',fontsize=14)

ax.tick_params(labelsize=12)


fig_figure4.savefig('~/Figure4.pdf', bbox_inches='tight')

plt.close()


# In[ ]:


## Attribution analysis  ##

# Sensitivity test of effect of CO2, temp, VPD and soil moisture on F4

## Removing effect of CO2
Adv4_co2,Sh4_co2 = C3C4model.c4fraction(temp_weight,finalgppc3_co2,finalgppc4_co2,treecover,crops)

F4_co2 = Sh4_co2/1.13

natural = 1 - luh2_crops - luh2_urban
F4_natural_co2 = F4_co2*natural

F4_withcrops_co2 = luh2_c4crops + F4_natural_co2

a = np.nanmean(temp,axis=0)
thresh = -19
a[a >= thresh] = 1
a[a < thresh] = np.nan

F4_withcrops_co2 = np.nan_to_num(F4_withcrops_co2)
F4_withcrops_co2 = F4_withcrops_co2*a

F4_withcrops_co2 = F4_withcrops_co2*areas_to_remove

F4_natural_co2 = np.nan_to_num(F4_natural_co2)
F4_natural_co2 = F4_natural_co2*a

F4_natural_co2 = F4_natural_co2*areas_to_remove

F3_natural_co2 = 1 - F4_natural_co2 - luh2_urban
F4_natural_co2[F4_natural_co2<0] = 0

F3_withcrops_co2 = 1 - F4_withcrops_co2 


## Removing effect of temp
Adv4_temp,Sh4_temp = C3C4model.c4fraction(temp_weight,finalgppc3_temp,finalgppc4_temp,treecover,crops)

F4_temp = Sh4_temp/1.13

natural = 1 - luh2_crops - luh2_urban
F4_natural_temp = F4_temp*natural

F4_withcrops_temp = luh2_c4crops + F4_natural_temp

a = np.nanmean(temp,axis=0)
thresh = -19
a[a >= thresh] = 1
a[a < thresh] = np.nan

F4_withcrops_temp = np.nan_to_num(F4_withcrops_temp)
F4_withcrops_temp = F4_withcrops_temp*a

F4_withcrops_temp = F4_withcrops_temp*areas_to_remove

F4_natural_temp = np.nan_to_num(F4_natural_temp)
F4_natural_temp = F4_natural_temp*a

F4_natural_temp = F4_natural_temp*areas_to_remove

F3_natural_temp = 1 - F4_natural_temp - luh2_urban
F4_natural_temp[F4_natural_temp<0] = 0

F3_withcrops_temp = 1 - F4_withcrops_temp 


## Removing effect of vpd
Adv4_vpd,Sh4_vpd = C3C4model.c4fraction(temp_weight,finalgppc3_vpd,finalgppc4_vpd,treecover,crops)

F4_vpd = Sh4_vpd/1.13

natural = 1 - luh2_crops - luh2_urban
F4_natural_vpd = F4_vpd*natural

F4_withcrops_vpd = luh2_c4crops + F4_natural_vpd

a = np.nanmean(temp,axis=0)
thresh = -19
a[a >= thresh] = 1
a[a < thresh] = np.nan

F4_withcrops_vpd = np.nan_to_num(F4_withcrops_vpd)
F4_withcrops_vpd = F4_withcrops_vpd*a

F4_withcrops_vpd = F4_withcrops_vpd*areas_to_remove

F4_natural_vpd = np.nan_to_num(F4_natural_vpd)
F4_natural_vpd = F4_natural_vpd*a

F4_natural_vpd = F4_natural_vpd*areas_to_remove

F3_natural_vpd = 1 - F4_natural_vpd - luh2_urban
F4_natural_vpd[F4_natural_vpd<0] = 0

F3_withcrops_vpd = 1 - F4_withcrops_vpd


## Removing effect of theta
Adv4_theta,Sh4_theta = C3C4model.c4fraction(temp_weight,finalgppc3_theta,finalgppc4_theta,treecover,crops)

F4_theta = Sh4_theta/1.13

natural = 1 - luh2_crops - luh2_urban
F4_natural_theta = F4_theta*natural

F4_withcrops_theta = luh2_c4crops + F4_natural_theta

a = np.nanmean(temp,axis=0)
thresh = -19
a[a >= thresh] = 1
a[a < thresh] = np.nan

F4_withcrops_theta = np.nan_to_num(F4_withcrops_theta)
F4_withcrops_theta = F4_withcrops_theta*a

F4_withcrops_theta = F4_withcrops_theta*areas_to_remove

F4_natural_theta = np.nan_to_num(F4_natural_theta)
F4_natural_theta = F4_natural_theta*a

F4_natural_theta = F4_natural_temp*areas_to_remove

F3_natural_theta = 1 - F4_natural_theta - luh2_urban
F4_natural_theta[F4_natural_theta<0] = 0

F3_withcrops_theta = 1 - F4_withcrops_theta 


## Difference between original simulation and constant simulation for each variable

## C4
delta_co2 = (F4_withcrops - F4_withcrops_co2) #/F4_withcrops_co2*100
delta_temp = (F4_withcrops - F4_withcrops_temp) #/F4_withcrops_temp*100
delta_vpd = (F4_withcrops - F4_withcrops_vpd) #/F4_withcrops_vpd*100
delta_theta = (F4_withcrops - F4_withcrops_theta) #/F4_withcrops_theta*100

## GPP
delta_gpp_c3_co2 = (finalgppc3 - finalgppc3_co2) #/F4_withcrops_co2*100
delta_gpp_c3_temp = (finalgppc3 - finalgppc3_temp) #/F4_withcrops_temp*100
delta_gpp_c3_vpd = (finalgppc3 - finalgppc3_vpd) #/F4_withcrops_vpd*100
delta_gpp_c3_theta = (finalgppc3 - finalgppc3_theta) #/F4_withcrops_theta*100

delta_gpp_c4_co2 = (finalgppc4 - finalgppc4_co2) #/F4_withcrops_co2*100
delta_gpp_c4_temp = (finalgppc4 - finalgppc4_temp) #/F4_withcrops_temp*100
delta_gpp_c4_vpd = (finalgppc4 - finalgppc4_vpd) #/F4_withcrops_vpd*100
delta_gpp_c4_theta = (finalgppc4 - finalgppc4_theta) #/F4_withcrops_theta*100


## D13C
delta_D13C_c3_co2 = (D13Cplant_C3 - D13Cplant_C3_co2) #/F4_withcrops_co2*100
delta_D13C_c3_temp = (D13Cplant_C3 - D13Cplant_C3_temp) #/F4_withcrops_temp*100
delta_D13C_c3_vpd = (D13Cplant_C3 - D13Cplant_C3_vpd) #/F4_withcrops_vpd*100
delta_D13C_c3_theta = (D13Cplant_C3 - D13Cplant_C3_theta) #/F4_withcrops_theta*100

delta_D13C_c4_co2 = (D13Cplant_C4 - D13Cplant_C4_co2) #/F4_withcrops_co2*100
delta_D13C_c4_temp = (D13Cplant_C4 - D13Cplant_C4_temp) #/F4_withcrops_temp*100
delta_D13C_c4_vpd = (D13Cplant_C4 - D13Cplant_C4_vpd) #/F4_withcrops_vpd*100
delta_D13C_c4_theta = (D13Cplant_C4 - D13Cplant_C4_theta) #/F4_withcrops_theta*100


# In[ ]:


## Figure 5: temporal variations of effects of CO2, temp, VPD and theta on F4

year = np.arange(0,35)+1982

# Set up the subplot figure

fig_figure5 = plt.figure(1, figsize=(10,14))

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


## F4

# Figure a
gs = gspec.GridSpec(5, 3, figure=fig_figure5, hspace=0.3,  width_ratios=[1,0,1],height_ratios=[0.15,0.15,0.15,0.15,0.15])

column = 0
row = 0

ax = fig_figure5.add_subplot(gs[row, column])

        
ax.plot(year,np.nanmean(delta_co2,axis = (1,2) ),color=colourWheel[0],lw=2.5)
ax.plot(year,np.nanmean(delta_temp,axis=(1,2)),color=colourWheel[7],lw=2.5)
ax.plot(year,np.nanmean(delta_vpd,axis=(1,2)),color=colourWheel[9],lw=2.5)
ax.plot(year,np.nanmean(delta_theta,axis=(1,2)),color=colourWheel[4],lw=2.5)

ci_co2 = 1.96 * np.std(np.nanmean(delta_co2,axis = (1,2) ))/np.sqrt(len(year))
ci_temp = 1.96 * np.std(np.nanmean(delta_temp,axis = (1,2) ))/np.sqrt(len(year))
ci_vpd = 1.96 * np.std(np.nanmean(delta_vpd,axis = (1,2) ))/np.sqrt(len(year))
ci_theta = 1.96 * np.std(np.nanmean(delta_theta,axis = (1,2) ))/np.sqrt(len(year))

ax.fill_between(year, (np.nanmean(delta_co2,axis = (1,2) )-ci_co2), (np.nanmean(delta_co2,axis = (1,2) )+ci_co2), color=colourWheel[0], alpha=.1)
ax.fill_between(year, (np.nanmean(delta_temp,axis = (1,2) )-ci_temp), (np.nanmean(delta_temp,axis = (1,2) )+ci_temp), color=colourWheel[7], alpha=.1)
ax.fill_between(year, (np.nanmean(delta_vpd,axis = (1,2) )-ci_vpd), (np.nanmean(delta_vpd,axis = (1,2) )+ci_vpd), color=colourWheel[9], alpha=.1)
ax.fill_between(year, (np.nanmean(delta_theta,axis = (1,2) )-ci_theta), (np.nanmean(delta_theta,axis = (1,2) )+ci_theta), color=colourWheel[4], alpha=.1)


ax.set_ylabel(u'$F_{4}$ change (-)',fontsize=14)
ax.set_ylim((-0.06,0.02))
ax.set_xlim((1981,2017))
ax.text(0.05, 0.99, '(a)',transform=ax.transAxes,va = 'top',fontsize=14)

ax.set_xticks([1985,1990,1995,2000,2005,2010,2015],fontsize=12)

ax.legend(['$c_{a}$','$T_{air}$','VPD',r'$\theta$'], fontsize=12, loc='center left', bbox_to_anchor= (-0.1, 1.1), ncol=5,
            borderaxespad=0, frameon=False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

ax.tick_params(labelsize=12)


# Figure b
columns=['co2','temp','vpd','theta']

deltas = [np.nanmean(np.nanmean(delta_co2[30:35],axis = (1,2) )),np.nanmean(np.nanmean(delta_temp[30:35],axis = (1,2) )),np.nanmean(np.nanmean(delta_vpd[30:35],axis = (1,2) )),np.nanmean(np.nanmean(delta_theta[30:35],axis = (1,2) ))]



column = 2
row = 0

ax = fig_figure5.add_subplot(gs[row, column])

ax.bar(columns, deltas,color=[colourWheel[0],colourWheel[7],colourWheel[9],colourWheel[4]])
ax.set_xticklabels(('$c_{a}$','$T_{air}$','VPD',r'$\theta$'))

ax.set_ylim((-0.06,0.02))

ax.text(0.05, 0.99, '(b)',transform=ax.transAxes,va = 'top',fontsize=14)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

ax.tick_params(labelsize=12)


## GPP

# Figure c
gs = gspec.GridSpec(5, 5, figure=fig_figure5, hspace=0.3,  width_ratios=[1.2,0.15,0.5,0,0.5],height_ratios=[0.15,0.15,0.15,0.15,0.15])


column = 0
row = 1

ax = fig_figure5.add_subplot(gs[row, column])

ax.plot(year,np.nanmean(delta_gpp_c3_co2,axis = (1,2) ),color='grey',lw=2.5)
ax.plot(year,np.nanmean(delta_gpp_c4_theta,axis = (1,2) ),color='grey',lw=2.5,ls="dashed")

ax.legend(['$C_{3}$','$C_{4}$'], fontsize=12, loc='center left', bbox_to_anchor= (0.2, 0.95), ncol=2,
            borderaxespad=0, frameon=False)


ax.plot(year,np.nanmean(delta_gpp_c3_co2,axis = (1,2) ),color=colourWheel[0],lw=2.5)
ax.plot(year,np.nanmean(delta_gpp_c3_temp,axis=(1,2)),color=colourWheel[7],lw=2.5)
ax.plot(year,np.nanmean(delta_gpp_c3_vpd,axis=(1,2)),color=colourWheel[9],lw=2.5)
ax.plot(year,np.nanmean(delta_gpp_c3_theta,axis=(1,2)),color=colourWheel[4],lw=2.5)

ci_co2 = 1.96 * np.std(np.nanmean(delta_gpp_c3_co2,axis = (1,2) ))/np.sqrt(len(year))
ci_temp = 1.96 * np.std(np.nanmean(delta_gpp_c3_temp,axis = (1,2) ))/np.sqrt(len(year))
ci_vpd = 1.96 * np.std(np.nanmean(delta_gpp_c3_vpd,axis = (1,2) ))/np.sqrt(len(year))
ci_theta = 1.96 * np.std(np.nanmean(delta_gpp_c3_theta,axis = (1,2) ))/np.sqrt(len(year))

ax.fill_between(year, (np.nanmean(delta_gpp_c3_co2,axis = (1,2) )-ci_co2), (np.nanmean(delta_gpp_c3_co2,axis = (1,2) )+ci_co2), color=colourWheel[0], alpha=.1)
ax.fill_between(year, (np.nanmean(delta_gpp_c3_temp,axis = (1,2) )-ci_temp), (np.nanmean(delta_gpp_c3_temp,axis = (1,2) )+ci_temp), color=colourWheel[7], alpha=.1)
ax.fill_between(year, (np.nanmean(delta_gpp_c3_vpd,axis = (1,2) )-ci_vpd), (np.nanmean(delta_gpp_c3_vpd,axis = (1,2) )+ci_vpd), color=colourWheel[9], alpha=.1)
ax.fill_between(year, (np.nanmean(delta_gpp_c3_theta,axis = (1,2) )-ci_theta), (np.nanmean(delta_gpp_c3_theta,axis = (1,2) )+ci_theta), color=colourWheel[4], alpha=.1)


ax.plot(year,np.nanmean(delta_gpp_c4_co2,axis = (1,2) ),color=colourWheel[0],lw=2.5,ls="dashed")
ax.plot(year,np.nanmean(delta_gpp_c4_temp,axis=(1,2)),color=colourWheel[7],lw=2.5,ls="dashed")
ax.plot(year,np.nanmean(delta_gpp_c4_vpd,axis=(1,2)),color=colourWheel[9],lw=2.5,ls="dashed")
ax.plot(year,np.nanmean(delta_gpp_c4_theta,axis=(1,2)),color=colourWheel[4],lw=2.5,ls="dashed")

ci_co2 = 1.96 * np.std(np.nanmean(delta_gpp_c4_co2,axis = (1,2) ))/np.sqrt(len(year))
ci_temp = 1.96 * np.std(np.nanmean(delta_gpp_c4_temp,axis = (1,2) ))/np.sqrt(len(year))
ci_vpd = 1.96 * np.std(np.nanmean(delta_gpp_c4_vpd,axis = (1,2) ))/np.sqrt(len(year))
ci_theta = 1.96 * np.std(np.nanmean(delta_gpp_c4_theta,axis = (1,2) ))/np.sqrt(len(year))

ax.fill_between(year, (np.nanmean(delta_gpp_c4_co2,axis = (1,2) )-ci_co2), (np.nanmean(delta_gpp_c4_co2,axis = (1,2) )+ci_co2), color=colourWheel[0], alpha=.1)
ax.fill_between(year, (np.nanmean(delta_gpp_c4_temp,axis = (1,2) )-ci_temp), (np.nanmean(delta_gpp_c4_temp,axis = (1,2) )+ci_temp), color=colourWheel[7], alpha=.1)
ax.fill_between(year, (np.nanmean(delta_gpp_c4_vpd,axis = (1,2) )-ci_vpd), (np.nanmean(delta_gpp_c4_vpd,axis = (1,2) )+ci_vpd), color=colourWheel[9], alpha=.1)
ax.fill_between(year, (np.nanmean(delta_gpp_c4_theta,axis = (1,2) )-ci_theta), (np.nanmean(delta_gpp_c4_theta,axis = (1,2) )+ci_theta), color=colourWheel[4], alpha=.1)


ax.set_ylabel(u'GPP change (PgC yr$^{-1}$)',fontsize=14)
ax.set_ylim((-30,200))
ax.set_xlim((1981,2017))
ax.text(0.05, 0.99, '(c)',transform=ax.transAxes,va = 'top',fontsize=14)

ax.set_xticks([1985,1990,1995,2000,2005,2010,2015],fontsize=12)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

ax.tick_params(labelsize=12)


# Figure d
columns=['co2','temp','vpd','theta']

deltas = [np.nanmean(np.nanmean(delta_gpp_c3_co2[30:35],axis = (1,2) )),np.nanmean(np.nanmean(delta_gpp_c3_temp[30:35],axis = (1,2) )),np.nanmean(np.nanmean(delta_gpp_c3_vpd[30:35],axis = (1,2) )),np.nanmean(np.nanmean(delta_gpp_c3_theta[30:35],axis = (1,2) ))]

column = 2
row = 1

ax = fig_figure5.add_subplot(gs[row, column])

ax.bar(columns, deltas,color=[colourWheel[0],colourWheel[7],colourWheel[9],colourWheel[4]])
ax.set_xticklabels(('$c_{a}$','$T_{air}$','VPD',r'$\theta$'))

ax.set_ylim((-30,200))

ax.text(0.05, 0.99, '(d) $C_{3}$',transform=ax.transAxes,va = 'top',fontsize=14)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

ax.tick_params(labelsize=12)


# Figure e
columns=['co2','temp','vpd','theta']

deltas = [np.nanmean(np.nanmean(delta_gpp_c4_co2[30:35],axis = (1,2) )),np.nanmean(np.nanmean(delta_gpp_c4_temp[30:35],axis = (1,2) )),np.nanmean(np.nanmean(delta_gpp_c4_vpd[30:35],axis = (1,2) )),np.nanmean(np.nanmean(delta_gpp_c4_theta[30:35],axis = (1,2) ))]

column = 4

ax = fig_figure5.add_subplot(gs[row, column])

ax.bar(columns, deltas,color=[colourWheel[0],colourWheel[7],colourWheel[9],colourWheel[4]])
ax.set_xticklabels(('$c_{a}$','$T_{air}$','VPD',r'$\theta$'))

ax.set_ylim((-10,60))

ax.text(0.05, 0.99, '(e) $C_{4}$',transform=ax.transAxes,va = 'top',fontsize=14)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

ax.tick_params(labelsize=12)


## D13C

# Figure f

column = 0
row = 2

ax = fig_figure5.add_subplot(gs[row, column])

ax.plot(year,np.nanmean(delta_D13C_c3_co2,axis = (1,2) ),color=colourWheel[0],lw=2.5)
ax.plot(year,np.nanmean(delta_D13C_c3_temp,axis=(1,2)),color=colourWheel[7],lw=2.5)
ax.plot(year,np.nanmean(delta_D13C_c3_vpd,axis=(1,2)),color=colourWheel[9],lw=2.5)
ax.plot(year,np.nanmean(delta_D13C_c3_theta,axis=(1,2)),color=colourWheel[4],lw=2.5)

ci_co2 = 1.96 * np.std(np.nanmean(delta_D13C_c3_co2,axis = (1,2) ))/np.sqrt(len(year))
ci_temp = 1.96 * np.std(np.nanmean(delta_D13C_c3_temp,axis = (1,2) ))/np.sqrt(len(year))
ci_vpd = 1.96 * np.std(np.nanmean(delta_D13C_c3_vpd,axis = (1,2) ))/np.sqrt(len(year))
ci_theta = 1.96 * np.std(np.nanmean(delta_D13C_c3_theta,axis = (1,2) ))/np.sqrt(len(year))

ax.fill_between(year, (np.nanmean(delta_D13C_c3_co2,axis = (1,2) )-ci_co2), (np.nanmean(delta_D13C_c3_co2,axis = (1,2) )+ci_co2), color=colourWheel[0], alpha=.1)
ax.fill_between(year, (np.nanmean(delta_D13C_c3_temp,axis = (1,2) )-ci_temp), (np.nanmean(delta_D13C_c3_temp,axis = (1,2) )+ci_temp), color=colourWheel[7], alpha=.1)
ax.fill_between(year, (np.nanmean(delta_D13C_c3_vpd,axis = (1,2) )-ci_vpd), (np.nanmean(delta_D13C_c3_vpd,axis = (1,2) )+ci_vpd), color=colourWheel[9], alpha=.1)
ax.fill_between(year, (np.nanmean(delta_D13C_c3_theta,axis = (1,2) )-ci_theta), (np.nanmean(delta_D13C_c3_theta,axis = (1,2) )+ci_theta), color=colourWheel[4], alpha=.1)

ax.plot(year,np.nanmean(delta_D13C_c4_co2,axis = (1,2) ),color=colourWheel[0],lw=2.5,ls="dashed")
ax.plot(year,np.nanmean(delta_D13C_c4_temp,axis=(1,2)),color=colourWheel[7],lw=2.5,ls="dashed")
ax.plot(year,np.nanmean(delta_D13C_c4_vpd,axis=(1,2)),color=colourWheel[9],lw=2.5,ls="dashed")
ax.plot(year,np.nanmean(delta_D13C_c4_theta,axis=(1,2)),color=colourWheel[4],lw=2.5,ls="dashed")

ci_co2 = 1.96 * np.std(np.nanmean(delta_D13C_c4_co2,axis = (1,2) ))/np.sqrt(len(year))
ci_temp = 1.96 * np.std(np.nanmean(delta_D13C_c4_temp,axis = (1,2) ))/np.sqrt(len(year))
ci_vpd = 1.96 * np.std(np.nanmean(delta_D13C_c4_vpd,axis = (1,2) ))/np.sqrt(len(year))
ci_theta = 1.96 * np.std(np.nanmean(delta_D13C_c4_theta,axis = (1,2) ))/np.sqrt(len(year))

ax.fill_between(year, (np.nanmean(delta_D13C_c4_co2,axis = (1,2) )-ci_co2), (np.nanmean(delta_D13C_c4_co2,axis = (1,2) )+ci_co2), color=colourWheel[0], alpha=.1)
ax.fill_between(year, (np.nanmean(delta_D13C_c4_temp,axis = (1,2) )-ci_temp), (np.nanmean(delta_D13C_c4_temp,axis = (1,2) )+ci_temp), color=colourWheel[7], alpha=.1)
ax.fill_between(year, (np.nanmean(delta_D13C_c4_vpd,axis = (1,2) )-ci_vpd), (np.nanmean(delta_D13C_c4_vpd,axis = (1,2) )+ci_vpd), color=colourWheel[9], alpha=.1)
ax.fill_between(year, (np.nanmean(delta_D13C_c4_theta,axis = (1,2) )-ci_theta), (np.nanmean(delta_D13C_c4_theta,axis = (1,2) )+ci_theta), color=colourWheel[4], alpha=.1)

ax.set_ylabel(u'$\mathregular{\u0394}^{13}$C change (‰)',fontsize=14)
ax.set_ylim((-0.3,0.4))
ax.set_xlim((1981,2017))
ax.text(0.05, 0.99, '(f)',transform=ax.transAxes,va = 'top',fontsize=14)

ax.set_xticks([1985,1990,1995,2000,2005,2010,2015],fontsize=12)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

ax.tick_params(labelsize=12)


# Figure g
columns=['co2','temp','vpd','theta']

deltas = [np.nanmean(np.nanmean(delta_D13C_c3_co2[30:35],axis = (1,2) )),np.nanmean(np.nanmean(delta_D13C_c3_temp[30:35],axis = (1,2) )),np.nanmean(np.nanmean(delta_D13C_c3_vpd[30:35],axis = (1,2) )),np.nanmean(np.nanmean(delta_D13C_c3_theta[30:35],axis = (1,2) ))]

column = 2

ax = fig_figure5.add_subplot(gs[row, column])

ax.bar(columns, deltas,color=[colourWheel[0],colourWheel[7],colourWheel[9],colourWheel[4]])
ax.set_xticklabels(('$c_{a}$','$T_{air}$','VPD',r'$\theta$'))

ax.set_ylim((-0.3,0.4))

ax.text(0.05, 0.99, '(g) $C_{3}$',transform=ax.transAxes,va = 'top',fontsize=14)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

ax.tick_params(labelsize=12)


# Figure h
columns=['co2','temp','vpd','theta']

deltas = [np.nanmean(np.nanmean(delta_D13C_c4_co2[30:35],axis = (1,2) )),np.nanmean(np.nanmean(delta_D13C_c4_temp[30:35],axis = (1,2) )),np.nanmean(np.nanmean(delta_D13C_c4_vpd[30:35],axis = (1,2) )),np.nanmean(np.nanmean(delta_D13C_c4_theta[30:35],axis = (1,2) ))]

column = 4

ax = fig_figure5.add_subplot(gs[row, column])

ax.bar(columns, deltas,color=[colourWheel[0],colourWheel[7],colourWheel[9],colourWheel[4]])
ax.set_xticklabels(('$c_{a}$','$T_{air}$','VPD',r'$\theta$'))

ax.set_ylim((-0.3,0.4))

ax.text(0.05, 0.99, '(h) $C_{4}$',transform=ax.transAxes,va = 'top',fontsize=14)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

ax.tick_params(labelsize=12)


fig_figure5.savefig('~/Figure5.pdf', bbox_inches='tight')

plt.close()


# In[ ]:




