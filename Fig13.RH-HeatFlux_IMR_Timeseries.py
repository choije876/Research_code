#
#	Fig13.RH-HeatFlux_IMR_Timeseries.py
#
############################################################

import numpy as np
import pandas as pd
import xarray as xr
import wrf
import netCDF4 as nc
from netCDF4 import Dataset

import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.interpolate import interp2d
from matplotlib.patches import Rectangle

import glob
import pytz
import os


# font ---------
plt.rcParams['font.family']='serif'
plt.rcParams['axes.unicode_minus']=False


#=======================================
# Define file paths and variables
PBL     = "YSU"
domain  = "02"
tt = 1718
alpbet = [f"({chr(97+i)})" for i in range(3)]

start_time = datetime(2020, 8, 18, 3, 0)  # 2020-08-18 03:00 KST = 2020-08-17 18:00 UTC
end_time = datetime(2020, 8, 19, 12, 0)    # 2020-08-19 18:00 KST = 2020-08-19 09:00 UTC


nvar_e1 = ["slhf","sshf", "ssrd", "strd"]
nvar_w1 = ["LH"  ,"HFX",  "SWDOWN", "GLW"]


# Region----
pos_l = ["Donghae", "Ulleungdo", "Ulleungdo_NE","Ulleungdo_NW","Dokdo", "Russia","East-South","Ulsan","Uljin","Imrang"]
pos_cl = ["DON", "ULL", "UNE","UNW","DOK", "RUS","ES","ULS","ULJ","IMR"]
pos_k = ["동해","울릉도_기상부이","울릉도_북동","울릉도_북서","독도","러시아","동해남쪽","울산_기상부이","울진_기상부이","임랑해수욕장"]
xlat  = [37.490 ,  37.455,  38.007,  37.743,  37.24 , 40  , 32,  35.2, 36.912, 35.303]
xlon  = [129.942, 131.114, 131.553, 130.601, 131.87 , 131 ,129, 129.5, 129.87, 129.293]
lon_idx=[168, 202, 213, 186, 224 , 227, 170, 160, 167 , 155]
lat_idx=[135, 136, 157, 145, 130 , 316, 88 ,  51, 114 , 55 ]

region_points = [9] # [1, 8, 9]



# date ---
year = "2020"
month= "08"
days = ["18", "19"]
start_time= datetime(2020, 8, 18, 3, 0)  # 2020-08-18 03:00 KST = 2020-08-17 18:00 UTC
end_time  = datetime(2020, 8, 19, 12, 0)    # 2020-08-19 18:00 KST = 2020-08-19 09:00 UTC
stime  = 0  ; etime= 24+9+1
stime_e= 18 ; etime_e=18+24+9+1
#--------------------------------------------

opath = f"./Fig/RH/"
os.makedirs(opath, exist_ok=True)





#=============================================================================
#-----------------------------------------------------------------------------
# 1. ** MODEL output load *** ------------------------------------------------
EXP_name1="CNTL"
EXP_name2="SKIN"
EXP_name3="CPLD"
MODEL = f"{EXP_name1}-{EXP_name2}-{EXP_name3}"

# WRF
idr_w = f"/scratch/x3158a03/wrf_output/EAST-C/2008/AUTO/"
ifn_w = idr_w+f"sstx-ERA5-MetnoSST-2way_wrfout_Fog_{domain}_2020-08-17_18:00:00.nc"  #sst_fix
ifn_w2= idr_w+f"skin-ERA5-MetnoSST-2way_wrfout_Fog_{domain}_2020-08-17_18:00:00.nc"  # sst_skin

# COAWST
idr_c = f"/scratch/x3158a03/coawst_output/2008/"
ifn_c = idr_c+f"WDM6-ERA5-SW-2way-YSU_wrfout_Fog_{domain}_2020-08-17_18:00:00.nc"  # solar_source

ds_ct = nc.Dataset(ifn_w)
ds_sk = nc.Dataset(ifn_w2)
ds_cp = nc.Dataset(ifn_c)

#data_w = xr.open_dataset(ifn_w)
#data_w2= xr.open_dataset(ifn_w2)
#data_c = xr.open_dataset(ifn_c)

lat2d_ct = ds_ct.variables['XLAT'][0,:,:]
lon2d_ct = ds_ct.variables['XLONG'][0,:,:]
lat2d_cp = ds_cp.variables['XLAT'][0,:,:]
lon2d_cp = ds_cp.variables['XLONG'][0,:,:]

ntime = len(ds_ct.variables['XLAT'][stime:etime,0,0])
print("ntime= ",ntime)
time = np.arange(ntime)



# ============================================
fts = 10
liw = 2.1

time_labels =["06\nAug.18", "12", "18", "00\nAug.19", "06", "12"]
tick_indices=[   3,  3+6,  9+6,  15+6,  21+6, 27+6]




for idx, position in enumerate(region_points):

    region = pos_k[position]
    print("region = ", pos_k[position])
    
    alpha = 0
    lat_range = slice(lat_idx[position]-alpha, lat_idx[position]+alpha+1)
    lon_range = slice(lon_idx[position]-alpha, lon_idx[position]+alpha+1)

    # Time -------
    start_time_utc = start_time - timedelta(hours=9)
    end_time_utc = end_time - timedelta(hours=9)

    times_w = ds_ct.variables['Times'][:]
    times = []
    for t in times_w:
         time_str = ''.join([c.decode('utf-8') for c in t]).replace('_', ' ')
         times.append(datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S'))

    time_mask = [(start_time_utc <= t <= end_time_utc) for t in times]
    time_indices = [i for i, mask in enumerate(time_mask) if mask]


    # LANDMASK 읽기 (0: 해양, 1: 육지)
    landmask1 = ds_ct.variables['LANDMASK'][0, lat_range, lon_range]  # WRF LANDMASK
    landmask2 = ds_cp.variables['LANDMASK'][0, lat_range, lon_range]  # COAWST LANDMASK

    rh_ct = wrf.getvar(ds_ct, "rh2", timeidx=wrf.ALL_TIMES)[time_indices, lat_range, lon_range]
    rh_sk = wrf.getvar(ds_sk, "rh2", timeidx=wrf.ALL_TIMES)[time_indices, lat_range, lon_range]
    rh_cp = wrf.getvar(ds_cp, "rh2", timeidx=wrf.ALL_TIMES)[time_indices, lat_range, lon_range]

    lhf_ct = ds_ct.variables['LH'][time_indices, lat_range, lon_range]
    lhf_sk = ds_sk.variables['LH'][time_indices, lat_range, lon_range]
    lhf_cp = ds_cp.variables['LH'][time_indices, lat_range, lon_range]

    hfx_ct = ds_ct.variables['HFX'][time_indices, lat_range, lon_range]
    hfx_sk = ds_sk.variables['HFX'][time_indices, lat_range, lon_range]
    hfx_cp = ds_cp.variables['HFX'][time_indices, lat_range, lon_range]


    lhf_ct = np.where(landmask1==0,lhf_ct, np.nan)
    lhf_sk = np.where(landmask1==0,lhf_sk, np.nan)
    lhf_cp = np.where(landmask2==0,lhf_cp, np.nan)

    hfx_ct = np.where(landmask1==0,hfx_ct, np.nan)
    hfx_sk = np.where(landmask1==0,hfx_sk, np.nan)
    hfx_cp = np.where(landmask2==0,hfx_cp, np.nan)

    rh_ct = np.where(landmask1==0,rh_ct, np.nan)
    rh_sk = np.where(landmask1==0,rh_sk, np.nan)
    rh_cp = np.where(landmask2==0,rh_cp, np.nan)



    lhf_ct_mean = np.nanmean(lhf_ct, axis=(1, 2))
    lhf_sk_mean = np.nanmean(lhf_sk, axis=(1, 2))
    lhf_cp_mean = np.nanmean(lhf_cp, axis=(1, 2))
    
    hfx_ct_mean = np.nanmean(hfx_ct, axis=(1, 2))
    hfx_sk_mean = np.nanmean(hfx_sk, axis=(1, 2))
    hfx_cp_mean = np.nanmean(hfx_cp, axis=(1, 2))

    rh_ct_mean = np.nanmean(rh_ct, axis=(1, 2))
    rh_sk_mean = np.nanmean(rh_sk, axis=(1, 2))
    rh_cp_mean = np.nanmean(rh_cp, axis=(1, 2))




    # =======================================================
    # Plotting            ===================================
    # =======================================================
  
    fig, axes = plt.subplots(1, 3, figsize=(12, 3), gridspec_kw={'hspace': 0.22, 'wspace': 0.4})  # 1row - 3col

    # Panel 1: RH ----- 
    ax0 = axes[0]

    ax0.plot(time, rh_ct_mean, label='CNTL', color='#6F8FAF', linewidth=liw, linestyle='-') # '#6F8FAF' 
    ax0.plot(time, rh_sk_mean, label='SKIN', color='#0066CC', linewidth=liw, linestyle='-')
    ax0.plot(time, rh_cp_mean, label='CPLD', color='#CA3433', linewidth=liw)

    ax0.tick_params(axis='x', labelsize=fts, width=2)
    ax0.set_xlabel('Day/Hour[LST]',fontsize=fts)
    ax0.set_xticks(tick_indices)
    ax0.set_xticklabels(time_labels)

    ax0.tick_params(axis='y', labelsize=fts, width=2)
    ax0.set_ylabel('RH [%]', fontsize=fts)
    ax0.set_ylim([85, 103])
    ax0.axhline(y=100, color='grey', linestyle='dotted', linewidth=1.4)  # Add horizontal line at y=0
    ax0.yaxis.set_major_locator(plt.MultipleLocator(5))


    ax0.text(0.02, 0.98, f'{(alpbet[0])} {pos_cl[position]}', transform=ax0.transAxes,
             fontsize=fts+1, verticalalignment='top', fontweight='bold')

    ax0.legend(loc='upper right', ncol=2, fontsize=fts-2, frameon=False,
               handlelength=1.2, handletextpad=0.4,columnspacing=0.7,labelspacing=0.25) #선길이, 간격, 열 간격, 행 간격


    # Panel 1: LHF -----
    ax1 = axes[1]
    ax1.plot(time, lhf_ct_mean, label='CNTL', color='#6F8FAF', linewidth=liw, linestyle='-')
    ax1.plot(time, lhf_sk_mean, label='SKIN', color='#0066CC', linewidth=liw, linestyle='-')
    ax1.plot(time, lhf_cp_mean, label='CPLD', color='#CA3433', linewidth=liw)

    ax1.tick_params(axis='x', labelsize=fts, width=2)
    ax1.set_xlabel('Day/Hour[LST]',fontsize=fts)
    ax1.set_xticks(tick_indices)
    ax1.set_xticklabels(time_labels)

    ax1.tick_params(axis='y', labelsize=fts, width=2)
    ax1.set_ylabel('Latent Heat flux [W/m$^2$]', fontsize=fts) #, fontweight='bold')
    ax1.set_ylim([-4,55])
    ax1.axhline(y=0, color='grey', linestyle='dotted', linewidth=1.6)  # Add horizontal line at y=0


    ax1.text(0.02, 0.98, f'{(alpbet[1])} {pos_cl[position]}', transform=ax1.transAxes,
             fontsize=fts+1, verticalalignment='top', fontweight='bold')



    # Panel 2: HFX -----
    ax2 = axes[2]
    ax2.plot(time, hfx_ct_mean, label='CNTL', color='#6F8FAF', linewidth=liw, linestyle='-')
    ax2.plot(time, hfx_sk_mean, label='SKIN', color='#0066CC', linewidth=liw, linestyle='-')
    ax2.plot(time, hfx_cp_mean, label='CPLD', color='#CA3433', linewidth=liw)

    ax2.tick_params(axis='x', labelsize=fts, width=2)
    ax2.set_xlabel('Day/Hour[LST]',fontsize=fts)
    ax2.set_xticks(tick_indices)
    ax2.set_xticklabels(time_labels)

    ax2.tick_params(axis='y', labelsize=fts, width=2)
    ax2.set_ylabel('Sensible Heat Flux [W/m$^2$]', fontsize=fts) #, fontweight='bold')
    ax2.set_ylim([-15,20])
    ax2.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax2.axhline(y=0, color='grey', linestyle='dotted', linewidth=1.)


    ax2.text(0.02, 0.98, f'{(alpbet[2])} {pos_cl[position]}', transform=ax2.transAxes,
             fontsize=fts+1, verticalalignment='top', fontweight='bold')


    ax0.grid(False)
    ax1.grid(False)
    ax2.grid(False)


    for spine in ax0.spines.values():
        spine.set_linewidth(1.2)  # Adjust the thickness as needed
    for spine in ax1.spines.values():
        spine.set_linewidth(1.2)  # Adjust the thickness as needed
    for spine in ax2.spines.values():
        spine.set_linewidth(1.2)  # Adjust the thickness as needed


    plt.tight_layout(rect=[0, 0.15, 1, 1])

    ofn = f"{opath}/RH-HeatFlux-d{domain}_{pos_cl[position]}_alpha{alpha}_1x3"
    plt.savefig(ofn + ".png", bbox_inches='tight', dpi=600, pad_inches=0.02)

    plt.show()
    plt.close()


ds_ct.close()
ds_sk.close()
ds_cp.close()


