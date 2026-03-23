#
#	Fig14.LWC-PBLH_IMR_Vert-Timeseries.py
#
###########################################################

import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc
import wrf
from wrf import (getvar, get_cartopy,ALL_TIMES, interplevel,latlon_coords,to_np)
from scipy.interpolate import interp1d

from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import os

# font ---------
plt.rcParams['font.family']='serif'
plt.rcParams['axes.unicode_minus']=False


#=======================================
# Define file paths and variables
domain = "02"
year = "2020"
month = "08"
days = ["17", "18", "19"]
lwc_min = 0.016 #0.016
tt=1718

stime=0
ntime = 24+9


ofn  = "LWC_Vert_timeseries_202008"
opath= f"./Fig/Vert_tmsr/"
os.makedirs(opath, exist_ok=True)


# domain -----
pos_l = ["Donghae", "Ulleungdo[ASOS]", "Ulleungdo", "Ulleungdo_NE","Ulleungdo_NW","Dokdo", "Russia","East-South","Ulsan","Uljin","Imrang","Ulsan_Zowi", "coast_imsi"]
pos_cl = ["DON","ULL[asos]", "ULL", "UNE","UNW","DOK", "RUS","ES","ULS","ULJ","IMR","ULS_Z","imsi"]
pos_k = ["동해","울릉도(ASOS)", "울릉도","울릉도_북동","울릉도_북서","독도","러시아","동해남쪽","울산","울진","임랑","울산_조위","연안_임시"]
xlat  = [37.490 ,  37.481,  37.455,  38.007,  37.743,  37.24 , 40  , 32,  35.2, 36.912, 35.303, 35.502, 35.8888 ]
xlon  = [129.942, 130.899, 131.114, 131.553, 130.601, 131.87 , 131 ,129, 129.5, 129.87,129.293,129.387, 129.999 ]
lon_idx=[168, 195, 202, 213, 186, 224 , 227, 170, 160, 167, 155, 156, 161]
lat_idx=[135, 137, 136, 157, 145, 130 , 316, 88 ,  51, 114, 55 , 62, 67 ]


heights = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                    120, 140, 150, 160, 180, 200, 220, 260, 280, 300, 350, 400,
                    450, 500, 550, 580, 600, 650, 700, 750, 800, 850, 900, 1000,
                    1084, 1196, 1308, 1422, 1802, 2239, 2739, 3275, 3877,
                    4530, 5226, 5990, 6752, 7510, 8277, 9029, 9800, 10565])


position_list = [10]  #Ulleungdo, Uljin,
#position_list = [2, 9, 10]  #Ulleungdo, Uljin,
#position_list = [1, 8, 9]   #Ulleungdo[ASOS], Ulsan, Uljin
print(f"{position_list}")



alpbet = [f"({chr(97+i)})" for i in range(6)]

#=============================================================================
#-----------------------------------------------------------------------------
#  ** MODEL output load *** ------------------------------------------------
EXP_name1="CNTL"
EXP_name2="SKIN"
EXP_name3="CPLD"

# WRF ----
idr_w = "/scratch/x3158a03/wrf_output/EAST-C/2008/AUTO/"
ifn_w = idr_w+f"sstx-ERA5-MetnoSST-2way_wrfout_Fog_{domain}_2020-08-17_18:00:00.nc"  #sst_fix
ifn_w2= idr_w+f"skin-ERA5-MetnoSST-2way_wrfout_Fog_{domain}_2020-08-17_18:00:00.nc"  # sst_skin

# 2. COAWST ----
idr_c = f"/scratch/x3158a03/coawst_output/2008/"
ifn_c = idr_c+f"WDM6-ERA5-SW-2way-YSU_wrfout_Fog_{domain}_2020-08-17_18:00:00.nc"  # solar_source HYCOM

ds_ct = nc.Dataset(ifn_w)
ds_sk = nc.Dataset(ifn_w2)
ds_cp = nc.Dataset(ifn_c)


z_ct_t  = getvar(ds_ct, 'z', timeidx=ALL_TIMES)
z_sk_t  = getvar(ds_sk, 'z', timeidx=ALL_TIMES)
z_cp_t  = getvar(ds_cp, 'z', timeidx=ALL_TIMES)

qcld_ct_t = getvar(ds_ct, 'QCLOUD', timeidx=ALL_TIMES) * (1.2*1e3)  # unit: kg/kg --> g/m^3
qcld_sk_t = getvar(ds_sk, 'QCLOUD', timeidx=ALL_TIMES) * (1.2*1e3)
qcld_cp_t = getvar(ds_cp, 'QCLOUD', timeidx=ALL_TIMES) * (1.2*1e3)

lsm_ct_t = getvar(ds_ct, 'LANDMASK', timeidx=0)
lsm_cp_t = getvar(ds_cp, 'LANDMASK', timeidx=0)

lwc_ct_t = qcld_ct_t.where(lsm_ct_t == 0, np.nan)
lwc_sk_t = qcld_sk_t.where(lsm_ct_t == 0, np.nan)
lwc_cp_t = qcld_cp_t.where(lsm_cp_t == 0, np.nan)

new_height = np.arange(0, 401, 5)
lwc_ct_t = interplevel(lwc_ct_t, z_ct_t, new_height)
lwc_sk_t = interplevel(lwc_sk_t, z_sk_t, new_height)
lwc_cp_t = interplevel(lwc_cp_t, z_cp_t, new_height)


fts = 11
lnw = 2

time = np.arange(ntime)
time_mt, height = np.meshgrid(time, new_height)

time_labels = ["06\nAug.18", "12", "18", "00\nAug.19", "06", "12" ]
tick_indices = [ 3,  3+6,  9+6,  15+6,  21+6, 27+6 ]
level_range = np.linspace(lwc_min, 2, 15)


fig, axes = plt.subplots(1, 3,figsize=(12, 4))



for idx, position in enumerate(position_list):
    print(f"position index = {position} ({pos_cl[position]})")

    row = idx

    # setting lat,lon ----
    alpha = 0
    lat_range = slice(lat_idx[position]-alpha, lat_idx[position]+alpha+1)
    lon_range = slice(lon_idx[position]-alpha, lon_idx[position]+alpha+1)


    # Read LWC -------
    lwc_ct = lwc_ct_t[stime:ntime, :, lat_range, lon_range].mean(axis=(2,3))
    lwc_sk = lwc_sk_t[stime:ntime, :, lat_range, lon_range].mean(axis=(2,3))
    lwc_cp = lwc_cp_t[stime:ntime, :, lat_range, lon_range].mean(axis=(2,3))

    lwc_ct = np.where(lwc_ct > lwc_min, lwc_ct, np.nan)
    lwc_sk = np.where(lwc_sk > lwc_min, lwc_sk, np.nan)
    lwc_cp = np.where(lwc_cp > lwc_min, lwc_cp, np.nan)
    print(f"lwc_ct shape: {lwc_ct.shape}")


    # read PBLH ------
    pblh_ct = ds_ct.variables['PBLH'][stime:ntime, lat_range, lon_range].mean(axis=(1,2))
    pblh_sk = ds_sk.variables['PBLH'][stime:ntime, lat_range, lon_range].mean(axis=(1,2))
    pblh_cp = ds_cp.variables['PBLH'][stime:ntime, lat_range, lon_range].mean(axis=(1,2))
    print("!!! pblh_ct.shape = ",pblh_ct.shape)

    pblh_ct = np.round(pblh_ct, 0)
    pblh_sk = np.round(pblh_sk, 0)
    pblh_cp = np.round(pblh_cp, 0)



    # =====================================================
    # Plotting      =======================================
    # =====================================================

    # 1. CNTL: LWC contour + pblh -------
    ax0 = axes[0]  # 첫 번째 행
    cf = ax0.contourf(time_mt, height, lwc_ct.T, levels=level_range, cmap='YlGnBu', extend="max")
    ax0.set_title(EXP_name1, fontsize=fts, fontweight='bold')
    ax0.plot(time, pblh_ct, 'k-', linewidth=2, label='PBL Height')

    ax0.tick_params(axis='both', which='major', labelsize=fts)
    ax0.grid(True, linestyle='--')
    ax0.set_xticks(tick_indices)
    ax0.set_xticklabels(time_labels)

    ax0.text(0.01, 0.97, f'{(alpbet[row*3 + 0])} {pos_cl[position]}', transform=ax0.transAxes,
          fontsize=fts, fontweight='bold', va='top', ha='left', color='black')
    ax0.set_ylabel('Altitude [m]', fontsize=fts-1)


    if row == len(position_list)-1:
        ax0.set_xlabel('Day/Hour[LST]', fontsize=fts-1, labelpad=10)
    else:
        ax0.set_xlabel('')

    if row == 0:
        ax0.legend(fontsize=fts-1)

    for spine in ax0.spines.values():
        spine.set_linewidth(1.2)


    # 2. SKIN :LWC contour + pblh -------
    ax1 = axes[1]
    cf = ax1.contourf(time_mt, height, lwc_sk.T[:,:], levels=level_range, cmap='YlGnBu', extend="max")
    ax1.set_title(EXP_name2, fontsize=fts, fontweight='bold')
    ax1.plot(time, pblh_sk, 'k-', linewidth=2, label='PBL Height')

    ax1.tick_params(axis='both', which='major', labelsize=fts)
    ax1.grid(True, linestyle='--')
    ax1.set_xticks(tick_indices)
    ax1.set_xticklabels(time_labels)

    if row == len(position_list)-1:
        ax1.set_xlabel('Day/Hour[LST]', fontsize=fts-1, labelpad=10)

    ax1.text(0.01, 0.97, f'{(alpbet[row*3 + 1])} {pos_cl[position]}', transform=ax1.transAxes,
          fontsize=fts, fontweight='bold', va='top', ha='left', color='black')

    for spine in ax1.spines.values():
        spine.set_linewidth(1.2)


    # 3. CPLD: LWC contour + pblh --------
    ax2 = axes[2]
    cf2 = ax2.contourf(time_mt, height, lwc_cp.T[:,:], levels=level_range, cmap='YlGnBu', extend="max")
    ax2.set_title(EXP_name3, fontsize=fts, fontweight='bold')
    ax2.plot(time, pblh_cp, 'k-', linewidth=2, label='PBL Height')

    ax2.tick_params(axis='both', which='major', labelsize=fts)
    ax2.grid(True, linestyle='--')
    ax2.set_xticks(tick_indices)
    ax2.set_xticklabels(time_labels)

    if row == len(position_list)-1:
        ax2.set_xlabel('Day/Hour[LST]', fontsize=fts-1, labelpad=10)

    ax2.text(0.01, 0.97, f'{(alpbet[row*3 + 2])} {pos_cl[position]}', transform=ax2.transAxes,
            fontsize=fts, fontweight='bold', va='top', ha='left', color='black')

    for spine in ax2.spines.values():
        spine.set_linewidth(1.2)


cbar_ax = fig.add_axes([0.15, 0.12, 0.7, 0.02])
cbar = plt.colorbar(cf, cax=cbar_ax, orientation='horizontal', extend="max")
cbar.set_label('Liquid Water Content [g m$^{-3}$]', fontsize=fts)
cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
cbar.ax.tick_params(labelsize=fts-1)


plt.tight_layout(rect=[0, 0.15, 1, 1])

plt.savefig(f"{opath}/{ofn}_{pos_cl[position]}_{alpha}", bbox_inches='tight', pad_inches=0.1, dpi=600)
print("Successfully saved")
plt.show()

plt.close()



ds_ct.close()
ds_sk.close()
ds_cp.close()


