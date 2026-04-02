#
#	Fig7.SST_Boxplot.py
#
##################################################

import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc
import wrf

from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import os


# ================================================
domain = "02"
tt   = 1718
nvar = "TSK" 

# DATE ----
year  = "2020"
month = "08"
days  = ["17","18", "19", "20"]

# DOMAIN ----
if domain == "01":
    lat = 48   ; slat = 27
    elon = 140.0; slon = 118

elif domain== "02":
   lat_max =45.0 ; elat = 38  
   lat_min =33.0 ; slat = 35  
   lon_max =138.0; elon = 133 
   lon_min =125.0; slon = 129 


pos_l = ["Donghae", "Ulleungdo[ASOS]", "Ulleungdo", "Ulleungdo_NE","Ulleungdo_NW","Dokdo", "Russia","East-South","Ulsan","Uljin","Imrang"]
pos_k = ["동해","울릉도(ASOS)", "울릉도","울릉도_북동","울릉도_북서","독도","러시아","동해남쪽","울산","울진","임랑"]
xlat  = [37.490 ,  37.481,  37.455,  38.007,  37.743,  37.24 , 40  , 32,  35.2, 36.912, 35.303]
xlon  = [129.942, 130.899, 131.114, 131.553, 130.601, 131.87 , 131 ,129, 129.5, 129.87, 129.293]
lon_idx=[168, 195, 202, 213, 186, 224 , 227, 170, 160, 167 , 155]
lat_idx=[135, 137, 136, 157, 145, 130 , 316, 88 ,  51, 114 , 55 ]

position = [2, 9, 10]


ull_coast_slat = 37.0
ull_coast_elat = 38.5
ull_coast_slon = 130.5
ull_coast_elon = 131.5

mid_coast_slat = 36.8 
mid_coast_elat = 38.0 
mid_coast_slon = 128.8
mid_coast_elon = 130.0

south_coast_slat = 35.5
south_coast_elat = 36.0
south_coast_slon = 129.2
south_coast_elon = 129.7



#=============================================================================
#-----------------------------------------------------------------------------
# 1. ** MODEL output load *** ------------------------------------------------
EXP_name1="SKIN-CNTL"
EXP_name2="CPLD-CNTL"
EXP_name3="CPLD-SKIN"

idr_w  = f"./"
ifn_ct = idr_w+f"sstx-wrfout_Fog_{domain}_2020-08-17_18:00:00.nc"  
ifn_sk = idr_w+f"skin-wrfout_Fog_{domain}_2020-08-17_18:00:00.nc"  

idr_c  = f"./"
ifn_cp = idr_c+f"cpld_wrfout_Fog_{domain}_2020-08-17_18:00:00.nc"

data0 = nc.Dataset(ifn_ct)
data1 = nc.Dataset(ifn_sk)
data2 = nc.Dataset(ifn_cp)
landmask=data1.variables['LANDMASK'][0,:,:]

lat2d = data1.variables['XLAT'][0,:,:]
lon2d = data1.variables['XLONG'][0,:,:]
ntime = 49

region_all_mask = (lat2d >= slat) & (lat2d <= elat) & (lon2d >= slon) & (lon2d <= elon) & (landmask == 0)

off_coast_mask = (lat2d >= ull_coast_slat) & (lat2d <= ull_coast_elat) & \
                (lon2d >= ull_coast_slon) & (lon2d <= ull_coast_elon) & \
                (landmask == 0)
mid_coast_mask = (lat2d >= mid_coast_slat) & (lat2d <= mid_coast_elat) & \
                (lon2d >= mid_coast_slon) & (lon2d <= mid_coast_elon) & \
                (landmask == 0)
south_coast_mask = (lat2d >= south_coast_slat) & (lat2d <= south_coast_elat) & \
                  (lon2d >= south_coast_slon) & (lon2d <= south_coast_elon) & \
                  (landmask == 0)


all_data = []
sst_data = {'all' : {'CNTL': [], 'SKIN': [], 'CPLD': []},
            'offshore' : {'CNTL': [], 'SKIN': [], 'CPLD': []},
            'mid_coast' : {'CNTL': [], 'SKIN': [], 'CPLD': []},
            'south_coast' : {'CNTL': [], 'SKIN': [], 'CPLD': []} }



# -----------------------------------------------
for k in range(ntime):
    sst_ct = wrf.getvar(data0,f"{nvar}",timeidx=k)-273.15  
    sst_sk = wrf.getvar(data1,f"{nvar}",timeidx=k)-273.15  
    sst_cp = wrf.getvar(data2,f"{nvar}",timeidx=k)-273.15  

    sst_ct_all = np.where(region_all_mask, sst_ct, np.nan)
    sst_sk_all = np.where(region_all_mask, sst_sk, np.nan)
    sst_cp_all = np.where(region_all_mask, sst_cp, np.nan)
    
    for val in sst_ct_all[~np.isnan(sst_ct_all)]:
        all_data.append({'Model': 'CNTL', 'Region': 'Entire Region', 'SST': val})
    for val in sst_sk_all[~np.isnan(sst_sk_all)]:
        all_data.append({'Model': 'SKIN', 'Region': 'Entire Region', 'SST': val})
    for val in sst_cp_all[~np.isnan(sst_cp_all)]:
        all_data.append({'Model': 'CPLD', 'Region': 'Entire Region', 'SST': val})
    
    sst_ct_off = np.where(off_coast_mask, sst_ct, np.nan)
    sst_sk_off = np.where(off_coast_mask, sst_sk, np.nan)
    sst_cp_off = np.where(off_coast_mask, sst_cp, np.nan)

    for val in sst_ct_off[~np.isnan(sst_ct_off)]:
        all_data.append({'Model': 'CNTL', 'Region': 'Offshore', 'SST': val})
    for val in sst_sk_off[~np.isnan(sst_sk_off)]:
        all_data.append({'Model': 'SKIN', 'Region': 'Offshore', 'SST': val})
    for val in sst_cp_off[~np.isnan(sst_cp_off)]:
        all_data.append({'Model': 'CPLD', 'Region': 'Offshore', 'SST': val})

    sst_ct_mid = np.where(mid_coast_mask, sst_ct, np.nan)
    sst_sk_mid = np.where(mid_coast_mask, sst_sk, np.nan)
    sst_cp_mid = np.where(mid_coast_mask, sst_cp, np.nan)
    
    for val in sst_ct_mid[~np.isnan(sst_ct_mid)]:
        all_data.append({'Model': 'CNTL', 'Region': 'Mid Coast', 'SST': val})
    for val in sst_sk_mid[~np.isnan(sst_sk_mid)]:
        all_data.append({'Model': 'SKIN', 'Region': 'Mid Coast', 'SST': val})
    for val in sst_cp_mid[~np.isnan(sst_cp_mid)]:
        all_data.append({'Model': 'CPLD', 'Region': 'Mid Coast', 'SST': val})
    
    sst_ct_south = np.where(south_coast_mask, sst_ct, np.nan)
    sst_sk_south = np.where(south_coast_mask, sst_sk, np.nan)
    sst_cp_south = np.where(south_coast_mask, sst_cp, np.nan)
    
    for val in sst_ct_south[~np.isnan(sst_ct_south)]:
        all_data.append({'Model': 'CNTL', 'Region': 'South Coast', 'SST': val})
    for val in sst_sk_south[~np.isnan(sst_sk_south)]:
        all_data.append({'Model': 'SKIN', 'Region': 'South Coast', 'SST': val})
    for val in sst_cp_south[~np.isnan(sst_cp_south)]:
        all_data.append({'Model': 'CPLD', 'Region': 'South Coast', 'SST': val})


df_combined = pd.DataFrame(all_data)
stats = df_combined.groupby(['Model', 'Region'])['SST'].describe()

plt.figure(figsize=(11, 6))
sns.set_style("whitegrid")


# Font ---------
plt.rcParams['font.family']='serif'
plt.rcParams['axes.unicode_minus']=False


plt.rcParams['boxplot.boxprops.linewidth'] = 1.0
plt.rcParams['boxplot.whiskerprops.linewidth'] = 1.0
plt.rcParams['boxplot.medianprops.linewidth'] = 1.0
plt.rcParams['boxplot.flierprops.linewidth'] = 1.0

plt.rcParams['boxplot.patchartist'] = True
plt.rcParams['boxplot.bootstrap'] = None
plt.rcParams['boxplot.whiskers'] = 1.5


box_data = []
positions = []
pos = 0
gap_between_models = 0.85 
region_gap = 0.15


for i, model in enumerate(['CNTL', 'SKIN', 'CPLD']):
    model_pos = i * gap_between_models 
    
    for j, region in enumerate(['Entire Region', 'Offshore', 'Mid Coast', 'South Coast']):
        data = df_combined[(df_combined['Model'] == model) & 
                          (df_combined['Region'] == region)]['SST'].values
        box_data.append(data)
        positions.append(model_pos + (j-1.5)*region_gap)

ax = plt.boxplot(box_data, positions=positions, patch_artist=True, medianprops={'color': 'black', 'linewidth': 1.5},
                showfliers=False, widths=0.12)

colors = ['#1B4965', '#3182BD', '#E6550D', '#8C564B'] * 3
for patch, color in zip(ax['boxes'], colors):
    patch.set_facecolor(color)

model_positions = [gap_between_models*i for i in range(3)]
plt.xticks(model_positions, ['CNTL', 'SKIN', 'CPLD'], fontsize=13, fontweight='bold')

plt.ylabel('SST [°C]', fontsize=16, fontweight='bold')
plt.xlabel('')  
plt.yticks(fontsize=13)
plt.ylim(18.5, 31)

custom_patches = [Patch(facecolor=color, edgecolor='black', linewidth=0.5) 
                  for color in colors]
plt.legend(custom_patches, ['Entire Region', 'Offshore', 'Central Coast', 'Southern Coast'], 
          loc='upper left', fontsize=12,
          frameon=True, facecolor="white", edgecolor='none',framealpha=0.8,
          handlelength=1.9, 
          handleheight=0.4) 

plt.grid(True, alpha=0.4, linestyle='--')
plt.tight_layout()

plt.show()


