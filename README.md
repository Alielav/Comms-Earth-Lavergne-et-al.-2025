# Comms-Earth-Lavergne-et-al.-2025
This repository contains the Python script to run the C3/C4 competition model and Graven et al. (2020) simple carbon cycle model, analyse simulations and create the figures presented in:

Lavergne et al. (accepted in principle) The recent decline in C4 vegetation abundance exerts little impact on atmospheric carbon isotopic composition, Communications Earth & Environment

# Code availability
The scripts are written in Python.
- Script to run the C3/C4 competition model and plot Figures 1-5: C3C4_model_simulations_analyses.py
- Script to run a modified version of Graven et al. (2020) model: Graven et al. 2020 model Python version with three vegetation boxes and C3 C4 plants - L2025.py
- Script to plot output from modified Graven et al. (2020) model and produce Figure 6: Plot outputs carbon cycle model L2025 - figure 6.py

# Data availability
- Processed climate data and model outputs to run .py scripts are available at https://zenodo.org/records/17726762.
- The annual percentage treecover from MEaSURES VCF5KYR v001 (VCF5KYR_1982_2016_05d.nc) is available at https://doi.org/10.5067/MEaSUREs/VCF/VCF5KYR.001.
- Urban areas and C3 and C4 crop distribution from LUHv2-2019 data are available at https://daac.ornl.gov/VEGETATION/guides/LUH2_GCB2019.html.
- MODIS Land Processes Distributed Active Archive Center (LP DAAC) landcover for snowandice (modis_snowandice_0.5d-2010.nc) and barren_sparsely_vegetated (modis_barren_sparsely_vegetated_0.5d-2010.nc) is derived from https://www.earthdata.nasa.gov/data/catalog/lpcloud-mcd12q1-061 but was regridded to 0.5x0.5 spatial resolution and is available at https://zenodo.org/records/17726762.
- The map of fraction of C4 plants from Still et al. (2009) is available at https://doi.org/10.3334/ORNLDAAC/932.
- The global C4 distribution map developed by Luo et al. (2024) is available at https://zenodo.org/records/10516423.
- The AVHRR GIMMS fAPAR data is derived from Zhu et al. (2013) and was updated until 2016 and made available by R. Myneni (data request contact: rmyneni@bu.edu).
- The concentrations and isotopic compositions of atmospheric CO2 are available in the Supplementary Material of Köhler et al. (2017), Graven et al. (2017) and Graven et al. (2020) papers, respectively but also available at https://zenodo.org/records/17726762. 
- The soil carbon isotopic data were extracted from Dong et al. (2022) and is available at https://doi.org/10.5281/zenodo.6556096. The subset of published data used to test the predictive skills of the model is also available here Glob_Soil_δ13C.csv.
- The leaf carbon isotopic data are derived from Cornwell et al. (2018) and available in the original paper. The subset of data used for the purpose of this paper is also available here leaf13C.csv.


# Author and contact

Aliénor Lavergne (alienor.lavergne@gmail.com)

# Acknowledgement
This research is a contribution to the LEMONTREE (Land Ecosystem Models based On New Theory, obseRvations and ExperimEnts) project, funded Schmidt Futures LLC (G-21-61881) (A.L., S.P.H., I.C.P.). A.L., S.P.H. and K.A. acknowledge support from the ERC-funded project GC2.0 (Global Change 2.0: Unlocking the past for a clearer future, grant number 694481). I.C.P. and N.D. acknowledge support from the ERC-funded project REALM (Re-inventing Ecosystem And Land-surface Models, grant number 787203). We thank R. Myneni and Z. Zhu for providing the AVHRR GIMMS fAPAR dataset, and the many researchers who have made their stable carbon isotope and plant C4 fraction data publicly available. We also thank David Orme for incorporating the new code into the official Python version of the P-model, Heather Graven for suggesting the use of the simple carbon cycle model and providing guidance for running it, and Joseph Ovwemuvwose for helpful discussions on crop inclusions in the analyses.


# References
Cornwell, W. K. et al. Climate and soils together regulate photosynthetic carbon isotope discrimination within C3 plants worldwide. Glob. Ecol. Biogeogr. 27, 1056–1067 (2018).

Dong, N. et al. Global Datasets of Carbon Isotope Composition (δ13C) for Ecological and Earth System Research. https://doi.org/10.5281/zenodo.6556096 (2022).

Graven, H. et al. Compiled records of carbon isotopes in atmospheric CO2 for historical simulations in CMIP6. Geosci. Model Dev. 10, 4405–4417 (2017).

Graven, H., Keeling, R. F. & Rogelj, J. Changes to Carbon Isotopes in Atmospheric CO2 Over the Industrial Era and Into the Future. Glob. Biogeochem. Cycles 34, (2020).

Köhler, P., Nehrbass-Ahles, C., Schmitt, J., Stocker, T. F. & Fischer, H. A 156 kyr smoothed history of the atmospheric greenhouse gases CO2, CH4, and N2O and their radiative forcing. Earth Syst. Sci. Data 9, 363–387 (2017).

Luo, X. et al. Mapping the global distribution of C4 vegetation using observations and optimality theory. Nat. Commun. 15, 1219 (2024).

Still, C. J., Berry, J., Collatz, G. J. & DeFries, R. ISLSCP II C4 Vegetation Percentage. (2009).

Zhu, Z. et al. Global Data Sets of Vegetation Leaf Area Index (LAI)3g and Fraction of Photosynthetically Active Radiation (FPAR)3g Derived from Global Inventory Modeling and Mapping Studies (GIMMS) Normalized Difference Vegetation Index (NDVI3g) for the Period 1981 to 2011. Remote Sens. 5, 927–948 (2013).
