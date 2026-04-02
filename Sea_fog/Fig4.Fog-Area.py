#
#	Fig4.Fog-Area.py
#    - Compare the Fog-Area 
#       between Satellite(GK2A) and Model(COAWST or WRF)
#
############################################################

import numpy as np
import xarray as xr
import wrf
import cartopy.crs as ccrs
from wrf import (getvar, get_cartopy,ALL_TIMES, interplevel,latlon_coords,to_np)
import pandas as pd

import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

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


# ===========================================================================
PBL   = "YSU"
domain= "02"
MODEL_1= "WRF"
MODEL_2= "COAWST"
lev = 21    
LEV = "400"
thrs= 0.016 
tt  = 1718  


# DATE -----
year  = "2020"
month = "08"
days  = ["17","18", "19", "20"]
wt_time_list=['2020-08-18 15:00','2020-08-18 19:00','2020-08-19 03:00']
wt_time_idxs=[12, 16, 24]

# Damain ----
elat = 42   
slat = 34.8 
elon = 135.5 
slon = 128.5 


#=============================================================================
#-----------------------------------------------------------------------------
# 1. ** MODEL output load *** ------------------------------------------------
idr_w = f"./"
ifn_we = idr_w+f"sstx-wrfout_Fog_{domain}_2020-08-17_18:00:00.nc"  
ifn_we2= idr_w+f"skin-wrfout_Fog_{domain}_2020-08-17_18:00:00.nc"  

idr_c = f"./"
ifn_ce= idr_c+f"cpld_wrfout_Fog_{domain}_2020-08-17_18:00:00.nc"  

we_data = xr.open_dataset(ifn_we)
we2_data= xr.open_dataset(ifn_we2)
ce_data = xr.open_dataset(ifn_ce)

wrf_e = Dataset(ifn_we)
wrf_e2= Dataset(ifn_we2)
coa_e = Dataset(ifn_ce)

lsm_ct = getvar(wrf_e, 'LANDMASK', timeidx=0)
lsm_cp = getvar(coa_e, 'LANDMASK', timeidx=0)


satellite_flags_list = []
wrf_flag_list = []
wrf2_flag_list= []
coa_flag_list = []
time_str_list = []


for wt_time in wt_time_idxs:
    qc_we = getvar(wrf_e, 'QCLOUD', timeidx=wt_time)[:lev,:,:].values * 1e3 * 1.2
    qc_we2= getvar(wrf_e2,'QCLOUD', timeidx=wt_time)[:lev,:,:].values * 1e3 * 1.2
    qc_ce = getvar(coa_e, 'QCLOUD', timeidx=wt_time)[:lev,:,:].values * 1e3 * 1.2
    qc_ce_high = getvar(coa_e, 'QCLOUD', timeidx=wt_time)[lev:,:,:].values * 1e3 * 1.2
    qc_ce_all  = getvar(coa_e, 'QCLOUD', timeidx=wt_time)[:,:,:].values * 1e3 * 1.2

    CBASE_we = getvar(wrf_e, "CBASEHT",timeidx=wt_time)
    CBASE_we2= getvar(wrf_e2,"CBASEHT",timeidx=wt_time)
    CBASE_ce = getvar(coa_e, "CBASEHT",timeidx=wt_time)


    wrf_e_flag = np.zeros_like(CBASE_we, dtype=int)
    wrf_e2_flag= np.zeros_like(CBASE_we2,dtype=int)
    coa_e_flag = np.zeros_like(CBASE_ce, dtype=int)


    # 1) Clear 
    wrf_e_flag[np.isnan(CBASE_we)] = 1
    wrf_e2_flag[np.isnan(CBASE_we2)] = 1
    coa_e_flag[np.isnan(CBASE_ce)] = 1

    # 2) Fog 
    lwc_fog_mask  = np.any(qc_we>=thrs, axis=0)
    lwc_fog_mask1 = np.any(qc_we2>=thrs, axis=0)
    lwc_fog_mask2 = np.any(qc_ce>=thrs, axis=0)
    wrf_e_flag[lwc_fog_mask]  = 2  
    wrf_e2_flag[lwc_fog_mask1]= 2  
    coa_e_flag[lwc_fog_mask2] = 2  

    wrf_e_flag = np.where(lsm_ct == 0, wrf_e_flag, np.nan)
    wrf_e2_flag = np.where(lsm_ct == 0, wrf_e2_flag, np.nan)
    coa_e_flag = np.where(lsm_cp == 0, coa_e_flag, np.nan)

    wrf_lat, wrf_lon = latlon_coords(CBASE_ce)


    base_time = ce_data['Times'].isel(Time=0).values
    selected_time = ce_data['Times'].isel(Time=wt_time).values

    if isinstance(selected_time, bytes):
        selected_time_str = selected_time.decode('utf-8')
    else:
        selected_time_str = selected_time.tobytes().decode('utf-8')

    selected_time_dt = datetime.strptime(selected_time_str, '%Y-%m-%d_%H:%M:%S')
    sat_time = selected_time_dt.strftime('%Y%m%d%H%M')


    # 2. ** Satellite Data(UTC) load ** -----------------------------------------------
    idr_s="./"
    ifn_s=f"{idr_s}gk2aamile2fogko020lc{sat_time}.nc"
    satellite_data = xr.open_dataset(ifn_s)

    proj_lcc = Proj(
        proj='lcc', lat_1=30, lat_2=60, lat_0=38, lon_0=126, x_0=0, y_0=0, datum='WGS84'
    )
    x_sat_range = np.linspace(-899000, 899000, satellite_data.dims['dim_x'])  # meters
    y_sat_range = np.linspace(899000, -899000, satellite_data.dims['dim_y'])  # meters

    xv_sat, yv_sat = np.meshgrid(x_sat_range, y_sat_range)
    lon_sat, lat_sat = proj_lcc(xv_sat, yv_sat, inverse=True)

    satellite_data = satellite_data.assign_coords(
        lat=(("dim_y", "dim_x"), lat_sat),
        lon=(("dim_y", "dim_x"), lon_sat) )

    satellite_data["FOG"] = satellite_data["FOG"].assign_coords(
        lat=(("dim_y", "dim_x"), lat_sat),
        lon=(("dim_y", "dim_x"), lon_sat)
    )
    FOG=satellite_data["FOG"]


    fog_flat = FOG.values.flatten()
    lat_sat_flat = lat_sat.flatten()
    lon_sat_flat = lon_sat.flatten()

    fog_interpolated_to_wrf = griddata(
        (lat_sat_flat, lon_sat_flat), fog_flat,
        (wrf_lat.values, wrf_lon.values), method='nearest'  # 보간 대신 nearest로 플래그 값 유지
    )

    fog_interpolated_to_wrf = np.where(
        np.isin(fog_interpolated_to_wrf, [2, 3, 6, 7]), np.nan, fog_interpolated_to_wrf)


    value_mapping = {
        4: 2,
        5: 2
    }

    for original_value, new_value in value_mapping.items():
        fog_interpolated_to_wrf[fog_interpolated_to_wrf == original_value] = new_value


    satellite_flags = fog_interpolated_to_wrf
    satellite_flags = np.where(lsm_ct == 0, satellite_flags, np.nan)



    #=============================================================================
    # 1. **fog_interpolated_to_wrf gird**
    FOG_on_WRF_grid = xr.DataArray(
        fog_interpolated_to_wrf,
        dims=CBASE_ce.dims, 
        coords={"XLON": (("south_north", "west_east"), wrf_lon.values),
                "XLAT": (("south_north", "west_east"), wrf_lat.values)}
    )

    # 2. **wrf_e_flag에 WRF grid **
    WRF_Flags_with_Coords = xr.DataArray(
        wrf_e_flag,
        dims=CBASE_ce.dims,  
        coords={"XLON": (("south_north", "west_east"), wrf_lon.values),
                "XLAT": (("south_north", "west_east"), wrf_lat.values)}
    )

    # 3. **wrf_e2_flag grid **
    WRF_Flags_with_Coords = xr.DataArray(
        wrf_e2_flag,
        dims=CBASE_ce.dims,  
        coords={"XLON": (("south_north", "west_east"), wrf_lon.values),
                "XLAT": (("south_north", "west_east"), wrf_lat.values)}
    )

    # 4. **coa_e_flag grid**
    WRF_Flags_with_Coords = xr.DataArray(
        coa_e_flag,
        dims=CBASE_ce.dims,  
        coords={"XLON": (("south_north", "west_east"), wrf_lon.values),
                "XLAT": (("south_north", "west_east"), wrf_lat.values)}
    )

    kst_time_dt = selected_time_dt + timedelta(hours=9)
    tit_time = kst_time_dt.strftime('%Y.%m.%d %H')

    satellite_flags_list.append(satellite_flags)
    wrf_flag_list.append(wrf_e_flag)
    wrf2_flag_list.append(wrf_e2_flag)
    coa_flag_list.append(coa_e_flag)
    time_str_list.append(kst_time_dt.strftime('%Y.%m.%d %Hh'))


we_data.close()
we2_data.close()
ce_data.close()



# ===========================================================================
# Plotting
# ===========================================================================
coastal_lons  = [129, 130.2, 130.2, 129, 129]  
coastal_lats  = [35.0, 35.0, 38.0, 38.0, 35.0]
offshore_lons = [130.2, elon, elon, 130.2, 130.2]  
offshore_lats = [36.0, 36.0, 41.0, 41.0, 36.0]
fns= 11

colors = ['white', 'royalblue'] 
cmap   = ListedColormap(colors)
flag_values = [1, 2]
boundaries = [0.5, 1.5, 2.5]
ticks = [1, 2]
ticks_label = ["Clear sky", "Sea fog"]


alpbet = [f"({chr(97+i)})" for i in range(12)]
ytext=["GK2A","CNTL","SKIN","CPLD"]


norm = mcolors.BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)

left, bottom, width, height = 0.3, 0.053, 0.45, 0.02

extent = [slon, elon, slat, elat]


fig, axes = plt.subplots(4, 3, figsize=(9, 11), subplot_kw={'projection': ccrs.PlateCarree()})

plt.tight_layout()
fig.subplots_adjust(wspace=-0.5, hspace=0.06, left=0.0, right=0.98, top=0.92, bottom=0.12)


for i in range(4):
    for j in range(3):
        idx = i *3 + j
        ax = axes[i,j]
        ax.set_extent([slon,elon,slat,elat], crs=ccrs.PlateCarree())
        ax.coastlines(resolution='10m', linewidth=1.5)
        gl = ax.gridlines(draw_labels=True, linestyle="--", color="gray", alpha=0.5)
        gl.xlocator = mticker.MultipleLocator(2)
        gl.ylocator = mticker.MultipleLocator(2)
        gl.top_labels  = False
        gl.left_labels = False

        LST_LABEL=["1500","1900","0300"]
        if i ==0 :
            data = satellite_flags_list[j]
            ax.text(0.01, 0.97, f"{alpbet[idx]} {LST_LABEL[idx]} LST", transform=ax.transAxes,
                     fontsize=fns, fontweight='bold', va='top', ha='left', color='black')

            if j == 0 :
               ax.plot(coastal_lons, coastal_lats, transform=ccrs.PlateCarree(), color='red', linewidth=1.3, linestyle='-')
            
        elif i ==1 :
            data = wrf_flag_list[j]
            ax.text(0.01, 0.97, alpbet[idx], transform=ax.transAxes,
                   fontsize=fns+1, fontweight='bold', va='top', ha='left', color='black')
        elif i==2 :
            data = wrf2_flag_list[j]
            ax.text(0.01, 0.97, alpbet[idx], transform=ax.transAxes,
                   fontsize=fns+1, fontweight='bold', va='top', ha='left', color='black')
        elif i==3 :
            data = coa_flag_list[j]
            ax.text(0.01, 0.97, alpbet[idx], transform=ax.transAxes,
                   fontsize=fns+1, fontweight='bold', va='top', ha='left', color='black')

        if j == 0:
            ax.text(-0.05, 0.5, ytext[i], transform=ax.transAxes,
                    fontsize=fns+1, fontweight='bold', va='center', ha='right', rotation=90)

        cf = ax.pcolormesh(wrf_lon, wrf_lat, data, cmap=cmap, norm=norm)
       
        ax.set_xticks([])
        ax.set_yticks([])

        gl.top_labels   =False
        gl.bottom_labels=False
        gl.left_labels  =False
        gl.right_labels =False

        if i == 3 :
           gl.bottom_labels = True
        if j == 2 :
           gl.right_labels  = True
         

cbar_ax = fig.add_axes([left, bottom, width, height])
cbar = fig.colorbar(cf, cax=cbar_ax, orientation='horizontal', ticks=[1, 2])
cbar.set_ticks(ticks)
cbar.set_ticklabels(ticks_label)
cbar.ax.tick_params(labelsize=fns-2)

for ax in axes.flatten():
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)
        spine.set_edgecolor('black')


plt.show()
satellite_data.close()
plt.close()

