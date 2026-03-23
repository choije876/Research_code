#
#	Fig5.Fog-Area-LWC_Timeseries.py
#
############################################################

import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import wrf
from wrf import (getvar, get_cartopy,ALL_TIMES, interplevel,latlon_coords,to_np)
import pandas as pd

import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
from wrf import (getvar, get_cartopy,ALL_TIMES, interplevel,latlon_coords,to_np)
from netCDF4 import Dataset
from pyproj import Proj
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

# font ---------
plt.rcParams['font.family']='serif'
plt.rcParams['axes.unicode_minus']=False


# Define Infomation ========================
domain = "02"
MODEL_ct = "CNTL"
MODEL_sk = "SKIN"
MODEL_cp = "CPLD"
tt  = 1718  # 1618
lev = 21 # 23: 500m
LEV = "400"
lwc_min = 0.016

# map range
elat = 42  #45 #44
slat = 35  #33 #32.5
elon = 135.5 #137.0
slon = 128.5 #123



# DATE -------------------------------------
year  = "2020"
month = "08"
days  = ["17","18", "19", "20"]
start_date = '2020-08-18 03:00'
end_date   = '2020-08-19 12:00'  # KST


opath = f"./Fig/FOG_Area/"
os.makedirs(opath, exist_ok=True)
#ofn = opath+"fogarea-lwc_percentage_timeseries_korea.png"
ofn = opath+"fogarea-lwc_percentage_timeseries_total_2x2.png"



stats_data_cos = []
stats_data_off = []


# Function  ---------------------
def create_simple_coastal_mask(lat, lon, distance_from_land_km=30):

    total_mask = np.zeros_like(lat, dtype=bool)
    coast_mask = np.zeros_like(lat, dtype=bool)

    ## total_slat=35 ;total_elat=41 ;total_slon=129 ;total_elon=135
    total_slat =35 ;total_elat =elat ;total_slon =slon ;total_elon =elon
    total_slat2=35 ;total_elat2=38.5 ;total_slon2=slon ;total_elon2=elon
    ##total_slat=ull_coast_slat ;total_elat=ull_coast_elat ;total_slon=ull_coast_slon ;total_elon=ull_coast_elon
    coast_slat=35 ;coast_elat=38 ;coast_slon=129 ;coast_elon=130.2
  
    east_total = (lon>=total_slon) & (lon<=total_elon) & (lat>=total_slat) & (lat<=total_elat)   
    east_total2= (lon>=total_slon2) & (lon<=total_elon2) & (lat>=total_slat2) & (lat<=total_elat2)   
    east_coast = (lon>=coast_slon) & (lon<=coast_elon) & (lat>=coast_slat) & (lat<=coast_elat)
    total_mask = east_total  # | south_coastal
    total_mask2= east_total2  # | south_coastal
    coast_mask = east_coast  # | south_coastal
#    east_offshore = (lon <= elon) & (lon >= 130.2) & (lat >= 36.0) & (lat <= 41)
    
    return total_mask, total_mask2, coast_mask


def calculate_regional_percentage(model_flag, total_mask, total_mask2, coast_mask, EXP_name):

    # Total area Fog : lev 내 어느 한층이라도 thresold(0.016) 이상
    # 1. 전체 그리드 개수
    total_area_points = np.sum(total_mask)
    total_area_points2= np.sum(total_mask2)
   #coast_area_points = np.sum(coast_mask)

    fog_tot_points = np.sum( (model_flag==2)& total_mask & ~np.isnan(model_flag) )
    fog_tot_points2= np.sum( (model_flag==2)& total_mask2& ~np.isnan(model_flag) )
    fog_cos_points = np.sum( (model_flag==2)& coast_mask & ~np.isnan(model_flag) )
    print("fog_tot_points = ",fog_tot_points) 
    print("fog_cos_points = ",fog_cos_points) 

    fog_prt_tot = (fog_tot_points / total_area_points) *100   
    fog_prt_tot2= (fog_tot_points2/ total_area_points2) *100   
    fog_prt_cos = (fog_cos_points / total_area_points ) *100   
    fog_prt_cos2= (fog_cos_points / total_area_points2) *100   


    return fog_prt_tot, fog_prt_tot2, fog_prt_cos, fog_prt_cos2




#=============================================================================
#-----------------------------------------------------------------------------
# 1. ** MODEL output load *** -----------------------------------------------
idr_w=f"/scratch/x3158a03/wrf_output/EAST-C/2008/AUTO/"
ifn_ct=idr_w+f"sstx-ERA5-MetnoSST-2way_wrfout_Fog_{domain}_2020-08-17_18:00:00.nc"  #sst_ct
ifn_sk=idr_w+f"skin-ERA5-MetnoSST-2way_wrfout_Fog_{domain}_2020-08-17_18:00:00.nc"  # sst_skin

# COAWST
idr_c=f"/scratch/x3158a03/coawst_output/2008/"
ifn_cp=idr_c+f"WDM6-2way-ERA5-GLORYS_wrfout_Fog_{domain}_2020-08-17_18:00:00.nc"


ct_data=xr.open_dataset(ifn_ct)
sk_data=xr.open_dataset(ifn_sk)
cp_data=xr.open_dataset(ifn_cp)

ds_ct = Dataset(ifn_ct)
ds_sk = Dataset(ifn_sk)
ds_cp = Dataset(ifn_cp)

lsm_ct = getvar(ds_ct, 'LANDMASK', timeidx=0)
lsm_cp = getvar(ds_cp, 'LANDMASK', timeidx=0)


fog_tot_ct_prct_list = []
fog_tot_sk_prct_list = []
fog_tot_cp_prct_list = []

fog_tot_ct_prct_list2 = []
fog_tot_sk_prct_list2 = []
fog_tot_cp_prct_list2 = []

fog_cos_ct_prct_list = []
fog_cos_sk_prct_list = []
fog_cos_cp_prct_list = []

fog_cos_ct_prct_list2 = []
fog_cos_sk_prct_list2 = []
fog_cos_cp_prct_list2 = []


lwc_tot_ct_list = []
lwc_tot_sk_list = []
lwc_tot_cp_list = []
lwc_tot_ct_list2 = []
lwc_tot_sk_list2 = []
lwc_tot_cp_list2 = []

lwc_cos_ct_list = []
lwc_cos_sk_list = []
lwc_cos_cp_list = []


stime=0 ; etime= 37
ntime = 24+9+3  #timeidx 0 ~ 9/15


for wt_time in range(stime, ntime):
    print("wt_time = ", wt_time )
    qc_ct= getvar(ds_ct, 'QCLOUD', timeidx=wt_time)[:lev,:,:].values * 1e3 * 1.2
    qc_sk= getvar(ds_sk, 'QCLOUD', timeidx=wt_time)[:lev,:,:].values * 1e3 * 1.2
    qc_cp= getvar(ds_cp, 'QCLOUD', timeidx=wt_time)[:lev,:,:].values * 1e3 * 1.2

    CBASE_c = getvar(ds_cp, "CBASEHT",timeidx=wt_time)    

    wrf_lat, wrf_lon = latlon_coords(CBASE_c)

    # 1. fog area -------------------------------------
    lwc_ct_flag = np.zeros_like(qc_ct[0,:,:], dtype=int)  # 2 차원 배열로 
    lwc_sk_flag = np.zeros_like(qc_sk[0,:,:], dtype=int)
    lwc_cp_flag = np.zeros_like(qc_cp[0,:,:], dtype=int)


    # (Fog) : lev 내 어느 한층이라도 lwc_min  이상
    lwc_fog_mask_ct = np.any(qc_ct>=lwc_min, axis=0)
    lwc_fog_mask_sk = np.any(qc_sk>=lwc_min, axis=0)
    lwc_fog_mask_cp = np.any(qc_cp>=lwc_min, axis=0)

    lwc_ct_flag[lwc_fog_mask_ct] = 2  # 5 : Fog -> 1
    lwc_sk_flag[lwc_fog_mask_sk] = 2  # 5 : Fog -> 1
    lwc_cp_flag[lwc_fog_mask_cp] = 2  # 5 : Fog -> 1

    lwc_ct_flag = np.where(lsm_ct == 0, lwc_ct_flag, np.nan)
    lwc_sk_flag = np.where(lsm_ct == 0, lwc_sk_flag, np.nan)
    lwc_cp_flag = np.where(lsm_cp == 0, lwc_cp_flag, np.nan)

    total_mask, total_mask2, coast_mask = create_simple_coastal_mask(wrf_lat, wrf_lon)

    print("**CNTL**")                                              
    fog_ct_total, fog_ct_total2, fog_ct_coast, fog_ct_coast2 = calculate_regional_percentage(lwc_ct_flag, total_mask, total_mask2, coast_mask,"CNTL")
    print("**SKIN**")
    fog_sk_total, fog_sk_total2, fog_sk_coast, fog_sk_coast2 = calculate_regional_percentage(lwc_sk_flag, total_mask, total_mask2, coast_mask,"SKIN")
    print("**CPLD**")
    fog_cp_total, fog_cp_total2, fog_cp_coast, fog_cp_coast2 = calculate_regional_percentage(lwc_cp_flag, total_mask, total_mask2, coast_mask,"CPLD")

    fog_tot_ct_prct_list.append(fog_ct_total) ; fog_cos_ct_prct_list.append(fog_ct_coast)
    fog_tot_sk_prct_list.append(fog_sk_total) ; fog_cos_sk_prct_list.append(fog_sk_coast)
    fog_tot_cp_prct_list.append(fog_cp_total) ; fog_cos_cp_prct_list.append(fog_cp_coast)
    fog_tot_ct_prct_list2.append(fog_ct_total2) ; fog_cos_ct_prct_list2.append(fog_ct_coast2)
    fog_tot_sk_prct_list2.append(fog_sk_total2) ; fog_cos_sk_prct_list2.append(fog_sk_coast2)
    fog_tot_cp_prct_list2.append(fog_cp_total2) ; fog_cos_cp_prct_list2.append(fog_cp_coast2)


    # 2. lwc amount mean ------------------------------------
    lwc_ct_total = np.where(total_mask, qc_ct, np.nan)    
    lwc_sk_total = np.where(total_mask, qc_sk, np.nan)    
    lwc_cp_total = np.where(total_mask, qc_cp, np.nan)
 
    lwc_ct_total2 = np.where(total_mask2, qc_ct, np.nan)    
    lwc_sk_total2 = np.where(total_mask2, qc_sk, np.nan)    
    lwc_cp_total2 = np.where(total_mask2, qc_cp, np.nan)

    lwc_ct_coast = np.where(coast_mask, qc_ct, np.nan)  
    lwc_sk_coast = np.where(coast_mask, qc_sk, np.nan)
    lwc_cp_coast = np.where(coast_mask, qc_cp, np.nan)


    # vertical sum
    lwc_ct_total = np.nanmean(np.nansum(lwc_ct_total, axis=0), axis=(0,1))*1e2
    lwc_sk_total = np.nanmean(np.nansum(lwc_sk_total, axis=0), axis=(0,1))*1e2
    lwc_cp_total = np.nanmean(np.nansum(lwc_cp_total, axis=0), axis=(0,1))*1e2

    lwc_ct_total2= np.nanmean(np.nansum(lwc_ct_total2, axis=0), axis=(0,1))*1e2
    lwc_sk_total2= np.nanmean(np.nansum(lwc_sk_total2, axis=0), axis=(0,1))*1e2
    lwc_cp_total2= np.nanmean(np.nansum(lwc_cp_total2, axis=0), axis=(0,1))*1e2

    lwc_ct_coast = np.nanmean(np.nansum(lwc_ct_coast, axis=0), axis=(0,1))*1e2
    lwc_sk_coast = np.nanmean(np.nansum(lwc_sk_coast, axis=0), axis=(0,1))*1e2
    lwc_cp_coast = np.nanmean(np.nansum(lwc_cp_coast, axis=0), axis=(0,1))*1e2


    lwc_tot_ct_list.append(lwc_ct_total)
    lwc_tot_sk_list.append(lwc_sk_total)
    lwc_tot_cp_list.append(lwc_cp_total)

    lwc_tot_ct_list2.append(lwc_ct_total2)
    lwc_tot_sk_list2.append(lwc_sk_total2)
    lwc_tot_cp_list2.append(lwc_cp_total2)

    lwc_cos_ct_list.append(lwc_ct_coast)
    lwc_cos_sk_list.append(lwc_sk_coast)
    lwc_cos_cp_list.append(lwc_cp_coast)



fog_tot_ct_mean_prct = np.mean(fog_tot_ct_prct_list)
fog_tot_sk_mean_prct = np.mean(fog_tot_sk_prct_list)
fog_tot_cp_mean_prct = np.mean(fog_tot_cp_prct_list)

fog_tot_ct_mean_prct2 = np.mean(fog_tot_ct_prct_list2)
fog_tot_sk_mean_prct2 = np.mean(fog_tot_sk_prct_list2)
fog_tot_cp_mean_prct2 = np.mean(fog_tot_cp_prct_list2)

fog_cos_ct_mean_prct = np.mean(fog_cos_ct_prct_list)
fog_cos_sk_mean_prct = np.mean(fog_cos_sk_prct_list)
fog_cos_cp_mean_prct = np.mean(fog_cos_cp_prct_list)

fog_cos_ct_mean_prct2 = np.mean(fog_cos_ct_prct_list2)
fog_cos_sk_mean_prct2 = np.mean(fog_cos_sk_prct_list2)
fog_cos_cp_mean_prct2 = np.mean(fog_cos_cp_prct_list2)


print("="*50)
print("fog_tot_ct_mean_prct =",fog_tot_ct_mean_prct)
print("fog_tot_sk_mean_prct=",fog_tot_sk_mean_prct)
print("fog_tot_cp_mean_prct=",fog_tot_cp_mean_prct)

print("fog_cos_ct_mean_prct=",fog_cos_ct_mean_prct)
print("fog_cos_sk_mean_prct=",fog_cos_sk_mean_prct)
print("fog_cos_cp_mean_prct=",fog_cos_cp_mean_prct)


# ===========================================================================
# Plotting
# ===========================================================================
fns= 12
line_w = 2.5

time = np.arange(ntime)
time_labels =["06\nAug.18", "12", "18", "00\nAug.19", "06", "12"]
tick_indices=[   3,  3+6,  9+6,  15+6,  21+6, 27+6  ]
alpbet = [f"{chr(97+i)}" for i in range(4)]



fig, axes = plt.subplots(2, 2, figsize=(9, 8))

fig.subplots_adjust(wspace=-0.5, hspace=0.1, left=0.1, right=0.98, top=0.92, bottom=0.12)


ax1 = axes[0,0]
ax1.plot(time, fog_tot_ct_prct_list, label='CNTL', color='#6F8FAF' ,linewidth=line_w) #linestyle='--' # '#6F8FAF'
ax1.plot(time, fog_tot_sk_prct_list, label='SKIN', color='#0066CC' ,linewidth=line_w, linestyle='-')
ax1.plot(time, fog_tot_cp_prct_list, label='CPLD', color='#CA3433' ,linewidth=line_w) #linestyle=':'

ax1.plot(time, fog_cos_ct_prct_list, label='CNTL', color='#6F8FAF' ,linewidth=line_w, linestyle='--') # '#6F8FAF'
ax1.plot(time, fog_cos_sk_prct_list, label='SKIN', color='#0066CC' ,linewidth=line_w, linestyle='--')
ax1.plot(time, fog_cos_cp_prct_list, label='CPLD', color='#CA3433' ,linewidth=line_w, linestyle='--')


ax1.text(0.01, 1.06, f'({alpbet[0]})', transform=ax1.transAxes,
          fontsize=fns, fontweight='bold', va='top', ha='left', color='black')

ax1.set_ylim([0, 50])
ax1.tick_params(axis='y' , labelsize=fns)
ax1.set_ylabel('Fog Area [%]', fontsize=fns )
ax1.yaxis.set_major_locator(plt.MultipleLocator(10))



ax2 = axes[0,1]
ax2.plot(time, lwc_tot_ct_list , label='CNTL', color='#6F8FAF' ,linewidth=line_w) 
ax2.plot(time, lwc_tot_sk_list , label='SKIN', color='#0066CC' ,linewidth=line_w) 
ax2.plot(time, lwc_tot_cp_list , label='CPLD', color='#CA3433' ,linewidth=line_w) 

ax2.plot(time, lwc_cos_ct_list , label='CNTL', color='#6F8FAF' ,linewidth=line_w, linestyle='--')
ax2.plot(time, lwc_cos_sk_list , label='SKIN', color='#0066CC' ,linewidth=line_w, linestyle='--')
ax2.plot(time, lwc_cos_cp_list , label='CPLD', color='#CA3433' ,linewidth=line_w, linestyle='--')

ax2.text(0.01, 1.06, f'({alpbet[1]})', transform=ax2.transAxes,
          fontsize=fns, fontweight='bold', va='top', ha='left', color='black')

ax2.set_ylim([0, 50])
ax2.tick_params(axis='y' , labelsize=fns)
ax2.set_ylabel('LWC [10$^{-2}$ g m$^{-3}$]', fontsize=fns )
ax2.yaxis.set_major_locator(plt.MultipleLocator(10))

ax1.tick_params(axis='x', labelbottom=False)  # ax1의 x축 라벨 숨김
ax2.tick_params(axis='x', labelbottom=False)  # ax2의 x축 라벨 숨김


# 2 row -------------------
ax3 = axes[1,0]
ax3.plot(time, fog_tot_ct_prct_list2, label='CNTL', color='#6F8FAF' ,linewidth=line_w) #linestyle='--' # '#6F8FAF'
ax3.plot(time, fog_tot_sk_prct_list2, label='SKIN', color='#0066CC' ,linewidth=line_w, linestyle='-')
ax3.plot(time, fog_tot_cp_prct_list2, label='CPLD', color='#CA3433' ,linewidth=line_w) #linestyle=':'

ax3.plot(time, fog_cos_ct_prct_list2, label='CNTL', color='#6F8FAF' ,linewidth=line_w, linestyle='--') # '#6F8FAF'
ax3.plot(time, fog_cos_sk_prct_list2, label='SKIN', color='#0066CC' ,linewidth=line_w, linestyle='--')
ax3.plot(time, fog_cos_cp_prct_list2, label='CPLD', color='#CA3433' ,linewidth=line_w, linestyle='--')


ax3.text(0.01, 1.06, f'({alpbet[2]})', transform=ax3.transAxes,
          fontsize=fns, fontweight='bold', va='top', ha='left', color='black')

ax3.set_ylim([0, 50])
ax3.tick_params(axis='y' , labelsize=fns)
ax3.set_ylabel('Fog Area [%]', fontsize=fns )
ax3.yaxis.set_major_locator(plt.MultipleLocator(10))

ax3.set_xticks(tick_indices)
ax3.set_xticklabels(time_labels)
ax3.set_xlabel('Day/Hour[LST]',fontsize=fns)


ax4 = axes[1,1]
ax4.plot(time, lwc_tot_ct_list2 , label='CNTL', color='#6F8FAF' ,linewidth=line_w)
ax4.plot(time, lwc_tot_sk_list2 , label='SKIN', color='#0066CC' ,linewidth=line_w)
ax4.plot(time, lwc_tot_cp_list2 , label='CPLD', color='#CA3433' ,linewidth=line_w)

ax4.plot(time, lwc_cos_ct_list , label='CNTL', color='#6F8FAF' ,linewidth=line_w, linestyle='--')
ax4.plot(time, lwc_cos_sk_list , label='SKIN', color='#0066CC' ,linewidth=line_w, linestyle='--')
ax4.plot(time, lwc_cos_cp_list , label='CPLD', color='#CA3433' ,linewidth=line_w, linestyle='--')

ax4.text(0.01, 1.06, f'({alpbet[3]})', transform=ax4.transAxes,
          fontsize=fns, fontweight='bold', va='top', ha='left', color='black')

ax4.set_ylim([0, 30])
ax4.tick_params(axis='y' , labelsize=fns)
ax4.set_ylabel('Liquid Water Content [10$^{-2}$ g m$^{-3}$]', fontsize=fns )
ax4.yaxis.set_major_locator(plt.MultipleLocator(5))

ax4.set_xticks(tick_indices)
ax4.set_xticklabels(time_labels)
ax4.set_xlabel('Day/Hour[LST]',fontsize=fns)





# 커스텀 범례 - 첫 번째 열: 실험, 두 번째 열: 선 종류
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#6F8FAF', linewidth=line_w, label='CNTL'),
    Line2D([0], [0], color='#0066CC', linewidth=line_w, label='SKIN'),
    Line2D([0], [0], color='#CA3433', linewidth=line_w, label='CPLD'),
    Line2D([0], [0], color='k', linewidth=line_w, linestyle='-', label='Total'),
    Line2D([0], [0], color='k', linewidth=line_w, linestyle='--', label='Coastal'),
    Line2D([0], [0], color='white', linewidth=0, label=''),  # 빈 공간
]

ax1.legend(handles=legend_elements, loc='upper left', ncol=2, fontsize=fns-2, 
           frameon=False, handlelength=1.7, handletextpad=0.5, 
           columnspacing=0.8, labelspacing=0.25)


#ax1.legend(loc='upper left', ncol=2, fontsize=fns-2, frameon=False,
#           handlelength=1.8, handletextpad=0.5,columnspacing=0.7,labelspacing=0.25) #선길이, 간격, 열 간



# Adjust border thickness
for spine in ax1.spines.values():
    spine.set_linewidth(1.2)
for spine in ax2.spines.values():
    spine.set_linewidth(1.2)
for spine in ax3.spines.values():
    spine.set_linewidth(1.2)
for spine in ax4.spines.values():
    spine.set_linewidth(1.2)


plt.tight_layout()
plt.savefig(ofn, bbox_inches='tight', dpi=600, pad_inches=0.01)
plt.show()
plt.close()



ct_data.close()
sk_data.close()
cp_data.close()
ds_ct.close()
ds_sk.close()
ds_cp.close()

