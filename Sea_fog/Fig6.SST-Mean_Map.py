#
#	Fig6.SST-Mean_Map.py
#     - CNTL vs SKIN vs CPLD
#
##################################################

import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc
import wrf

import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from pyproj import Proj
import os

# Font ---------
plt.rcParams['font.family']='serif'
plt.rcParams['axes.unicode_minus']=False

# ================================================
domain = "01"
tt   = 1718
nvar = "TSK" 


# DOMAIN-AREA ----
elat = 42  
slat = 34  
elon = 136
slon = 128

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


# DATE ----
year  = "2020"
month = "08"
days  = ["17","18", "19", "20"]

dates_kst = [
    datetime(2020, 8, 18),
]

time_lists_kst = []
time_lists_utc = []

for date_kst in dates_kst:

    daily_times_kst = []
    for hour in range(3, 24):
        daily_times_kst.append(date_kst + timedelta(hours=hour))
    time_lists_kst.append(daily_times_kst)

    daily_times_utc = []
    for hour in range(3, 24):
        utc_time = date_kst + timedelta(hours=hour) - timedelta(hours=9)
        daily_times_utc.append(utc_time)
    time_lists_utc.append(daily_times_utc)




# Function  ==========================================
def read_gk2a_hourly(time_list_utc, idr_s, proj_params):
    temp_s_list = []
    valid_mask_list = []

    lat_sat = None
    lon_sat = None

    for i, time in enumerate(time_list_utc):
        time_str = time.strftime("%Y%m%d%H00")
        gk2a_file = idr_s + f"gk2a_ami_le2_sst_ko020lc_{time_str}.nc"

        try:
            ds_s = xr.open_dataset(gk2a_file)

            if i == 0:
                x_sat_range = np.linspace(-899000, 899000, ds_s.dims['dim_x'])
                y_sat_range = np.linspace(899000, -899000, ds_s.dims['dim_y'])
                xv_sat, yv_sat = np.meshgrid(x_sat_range, y_sat_range)
                lon_sat, lat_sat = proj_params(xv_sat, yv_sat, inverse=True)


            temp_s = ds_s['SST'].values.astype(float)

            valid_mask = (temp_s != 65535) & (~np.isnan(temp_s))
            temp_s[~valid_mask] = np.nan
            temp_s[valid_mask] = temp_s[valid_mask] - 273.15 

            temp_s_list.append(temp_s)
            valid_mask_list.append(valid_mask)

            ds_s.close()

            kst_time = time + timedelta(hours=9)

        except Exception as e:
            print(f"File not found or error for {time_str}: {e}")
            if i == 0:
                print("Cannot proceed without first file for coordinates")
                return None, None, None
            temp_s_list.append(np.full_like(temp_s_list[0], np.nan))
            valid_mask_list.append(np.zeros_like(valid_mask_list[0], dtype=bool))

    return temp_s_list, valid_mask_list, (lat_sat, lon_sat)


def draw_box(ax, slat, elat, slon, elon, color='red', linestyle='-', linewidth=1.7, label=None):
    lons = [slon, elon, elon, slon, slon]
    lats = [slat, slat, elat, elat, slat]

    ax.plot(lons, lats, transform=ccrs.PlateCarree(),
            color=color, linewidth=linewidth, linestyle=linestyle)

    if label:
        ax.text(slon + (elon-slon)*0.5, elat + 0.1, label,
                transform=ccrs.PlateCarree(), color=color,
                fontsize=10, ha='center', va='bottom')


#=============================================================================
# 1. ** GK2A dataload *** ----------------------------------------------------
idr_s = f"./"

proj = Proj(proj='lcc',
        lat_1=30.0,
        lat_2=60.0,
        lat_0=38.0,  
        lon_0=126.0,
        x_0=0.0,
        y_0=0.0)

lat_sat = None
lon_sat = None


daily_data=[]
for day_idx in range(len(dates_kst)):
    date_kst = dates_kst[day_idx]
    time_list_kst = time_lists_kst[day_idx]
    time_list_utc = time_lists_utc[day_idx]


    temp_s_list, valid_mask_list, sat_coords = read_gk2a_hourly(time_list_utc, idr_s, proj)
    lat_sat, lon_sat = sat_coords

    temp_s_array = np.array(temp_s_list)
    sst_mean_s = np.nanmean(temp_s_array, axis=0)

    daily_data.append({
        'date': date_kst,
        'gk2a_mean': sst_mean_s,
        'lat_sat': lat_sat,
        'lon_sat': lon_sat
    })



#=============================================================================
#-----------------------------------------------------------------------------
# 1. ** MODEL output load *** ------------------------------------------------
EXP_name1="CNTL"
EXP_name2="SKIN"
EXP_name3="CPLD"
MODEL = f"{EXP_name1}-{EXP_name2}-{EXP_name3}"

idr_w = f"./"
ifn_we = idr_w+f"sstx-wrfout_Fog_{domain}_2020-08-17_18:00:00.nc" 
ifn_we2= idr_w+f"skin-wrfout_Fog_{domain}_2020-08-17_18:00:00.nc" 

idr_c = f"./"
ifn_ce= idr_c+f"cpld_wrfout_Fog_{domain}_2020-08-17_18:00:00.nc"  

data0= nc.Dataset(ifn_we)
data1= nc.Dataset(ifn_we2)
data2= nc.Dataset(ifn_ce)
landmask=data1.variables['LANDMASK'][0,:,:]

lat2d = data1.variables['XLAT'][0,:,:]
lon2d = data1.variables['XLONG'][0,:,:]



# -----------------------------------------------
sst_ct_list =[]
sst_sk_list =[]
sst_cp_list =[]


ntime = 25-3 

for k in range(ntime):
    sst1 = wrf.getvar(data0,f"{nvar}",timeidx=k)-273.15  
    sst2 = wrf.getvar(data1,f"{nvar}",timeidx=k)-273.15  
    sst3 = wrf.getvar(data2,f"{nvar}",timeidx=k)-273.15  

    sst1.coords["XLAT"] = (("south_north", "west_east"), lat2d)
    sst1.coords["XLONG"] = (("south_north", "west_east"), lon2d)
    sst2.coords["XLAT"] = (("south_north", "west_east"), lat2d)
    sst2.coords["XLONG"] = (("south_north", "west_east"), lon2d)
    sst3.coords["XLAT"] = (("south_north", "west_east"), lat2d)
    sst3.coords["XLONG"] = (("south_north", "west_east"), lon2d)

    sst1 = np.where(landmask ==1 , np.nan, sst1)
    sst2 = np.where(landmask ==1 , np.nan, sst2)
    sst3 = np.where(landmask ==1 , np.nan, sst3)


    south_north, west_east = sst1.shape

    sst_ct_list.append(sst1)
    sst_sk_list.append(sst2)
    sst_cp_list.append(sst3)

sst_ct_mean = np.nanmean(sst_ct_list, axis=0)
sst_sk_mean = np.nanmean(sst_sk_list, axis=0)
sst_cp_mean = np.nanmean(sst_cp_list, axis=0)




#=========================================
# Plotting ===============================
fns= 13
alpbet = [f"({chr(97+i)})" for i in range(4)]

levels = np.arange(22, 29, 0.5)
comap = 'RdBu_r'


region_by_area = {
    "동해": ['임랑해수욕장', '울릉도북서', '울릉도북동', '울릉도_기상부이',
    '동해_기상부이', '포항_기상부이', '울산_기상부이', '울진_기상부이']
}

pos_l = ["Imrang", "Ulleungdo_NW", "Ulleungdo_NE","Ulleungdo", "Donghae", "Pohang", "Ulsan", "Uljin"]
pos_k = ["임랑해수욕장", "울릉도_북서", "울릉도_북동", "울릉도_기상부이", "동해", "포항_기상부이", "울산_기상부이","울진_기상부이"]
xlat  = [35.303, 37.743,38.007,37.455,37.490,36.21,35.2,36.912]
xlon  = [129.293,130.601,131.553,131.114,129.942,129.47,129.5,129.87]
lon_idx=[155,186,213,202,168,157,160,167]
lat_idx=[55, 145,157,136,135, 88, 51,114]



fig, axes = plt.subplots(1, 4, figsize=(15, 5), subplot_kw={'projection': ccrs.PlateCarree()})

for ax in axes :
    ax.set_extent([slon, elon, slat, elat ], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='10m', linewidth=1.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    plt.setp(ax.spines.values(), lw=1.2, color='black')
    gl = ax.gridlines(draw_labels=True, linestyle="--", color="gray", alpha=0.5)
    gl.top_labels  = False
    gl.left_labels = False
    gl.right_labels = False


# 0. SST_Satellite ---
cf0 = axes[0].contourf(lon_sat, lat_sat, sst_mean_s,
                       levels=levels, cmap=comap, extend='both', transform=ccrs.PlateCarree())

axes[0].set_title(f'GK2A', fontsize=fns, fontweight='bold')
axes[0].text(0.02, 0.98, f'{(alpbet[0])}', transform=axes[0].transAxes,
              fontsize=fns+1, verticalalignment='top', fontweight='bold')


colors = ['#1B4965', '#3182BD', '#E6550D', '#8C564B'] 

draw_box(axes[0], ull_coast_slat, ull_coast_elat, ull_coast_slon, ull_coast_elon,
         color=colors[1]) 
draw_box(axes[0], mid_coast_slat, mid_coast_elat, mid_coast_slon, mid_coast_elon,
         color=colors[2])
draw_box(axes[0], south_coast_slat, south_coast_elat, south_coast_slon, south_coast_elon,
         color=colors[3]) 

for lat_pt, lon_pt in zip(xlat, xlon):
        axes[0].plot(
            lon_pt, lat_pt,
            marker='x', markersize=5, markeredgewidth=1.5,
            color='dimgrey', transform=ccrs.PlateCarree()
        )



# 1. SST_CNTL ---
cf1 = axes[1].contourf(wrf.to_np(lon2d), wrf.to_np(lat2d), wrf.to_np(sst_ct_mean), levels=levels,
                 cmap=comap, extend='both') 
axes[1].set_title(f'{EXP_name1}', fontsize=fns, fontweight='bold')
axes[1].text(0.01, 0.97, f'{(alpbet[1])}', transform=axes[1].transAxes,
         fontsize=fns, fontweight='bold', va='top', ha='left', color='black')

# 2. SST_SKIN ---
cf2 = axes[2].contourf(wrf.to_np(lon2d), wrf.to_np(lat2d), wrf.to_np(sst_sk_mean), levels=levels,
                 cmap=comap, extend='both') 
axes[2].set_title(f'{EXP_name2}', fontsize=fns, fontweight='bold')
axes[2].text(0.01, 0.97, f'{(alpbet[2])}', transform=axes[2].transAxes,
         fontsize=fns, fontweight='bold', va='top', ha='left', color='black')

# 3. SST_CPLD ---
cf3 = axes[3].contourf(wrf.to_np(lon2d), wrf.to_np(lat2d), wrf.to_np(sst_cp_mean), levels=levels,
                 cmap=comap, extend='both')  
axes[3].set_title(f'{EXP_name3}', fontsize=fns, fontweight='bold')
axes[3].text(0.01, 0.97, f'{(alpbet[3])}', transform=axes[3].transAxes,
         fontsize=fns, fontweight='bold', va='top', ha='left', color='black')
gl.right_labels = True


lef , bot,  right, top = 0, 0.5, 1, 0.
left, bottom, width, height = 0.15, 0.15, 0.7, 0.015

cbar_ax1 = fig.add_axes([left, bottom, width, height])
cbar1 = fig.colorbar(cf0, cax=cbar_ax1, orientation='horizontal')
if nvar == "TSK":
    cbar1.set_label(f'SST [°C]', fontsize=fns)
elif nvar == "T2":
    cbar1.set_label(f'T2m [°C]', fontsize=fns)
cbar1.ax.tick_params(labelsize=fns-1)


plt.tight_layout(rect=[lef, bot, right, top])
plt.show()
plt.close()


