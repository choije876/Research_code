#
#	AppendixA_SST_intial_field.py
#	- comparing ERA5, HYCOM
#	- 2020.08.18 03KST (=2020.08.17 18UTC)
#
############################################################

import numpy as np
import xarray as xr
import netCDF4 as nc
import pandas as pd
from datetime import datetime, timedelta
from scipy.interpolate import interp1d, griddata

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from pyproj import Proj
import os


# font ---------
plt.rcParams['font.family']='serif'
plt.rcParams['axes.unicode_minus']=False



# Information Setting =================================
preofn = f"Initial-SST-field_ERA-HYCOM-model"
opath = f"./Fig/SST/"
os.makedirs(opath, exist_ok=True)
ofn = f"{opath}/{preofn}"


# DOMAIN-AREA ----
domain = "01"

# DOMAIN-AREA
if domain == "01":
    elat = 45   ; slat = 30
    elon = 136.0; slon = 119

elif domain== "02":
   lat_max =45.0 ; elat = 42 #44
   lat_min =33.0 ; slat = 34 #32.5
   lon_max =138.0; elon = 136.0
   lon_min =125.0; slon = 128



# =========================================================
# Read & Load file                                        =
# =========================================================
# 2. WRF ----
#idr_w = "/scratch/x3158a03/DATA/Reanalysis/2008/"
#ifn_w =idr_w+f"ERA5_2020_08_Korea.nc"  
idr_w = "/scratch/x3158a03/wrf_output/EAST-C/2008/AUTO/"
ifn_w =idr_w+f"sstx-ERA5-MetnoSST-2way_wrfout_Fog_{domain}_2020-08-17_18:00:00.nc"

# 3. COAWST ----
idr_c =f"/scratch/x3158a03/make_input_new/COAWST/preprocess/make_input/scripts/hycom/"
ifn_c = idr_c+f"hycom_GLBy0.08_expt_93.0_2020-08-17T18Z_subset_TYPs_new.nc" 
#idr_c =f"/scratch/x3158a03/coawst_output/2008/"
#ifn_c = idr_c+f"WDM6-ERA5-SW-2way-YSU_wrfout_Fog_{domain}_2020-08-17_18:00:00.nc"

ds_ct = nc.Dataset(ifn_w)
ds_cp = nc.Dataset(ifn_c)


# SST - from WRF./ROMS. ***
#sst_ct = ds_ct['sst'][18, :, :]
sst_cp = ds_cp["water_temp"][0, 0, :, :]
sst_ct = ds_ct['SST'][0,:,:]
#sst_cp = ds_cp['SST'][0,:,:]

lml_ct = ds_ct.variables['LANDMASK'][0, :, :]
#lml_cp = ds_cp.variables['LANDMASK'][0, :, :]
sst_ct = np.where(lml_ct == 0, sst_ct, np.nan)- 273.15
#sst_cp = np.where(lml_cp == 0, sst_cp, np.nan)- 273.15


#lat_e = ds_ct["latitude"]
#lon_e = ds_ct["longitude"]
lat_h = ds_cp["lat"]
lon_h = ds_cp["lon"]
#lon2d_e, lat2d_e = np.meshgrid(lon_e,lat_e)
lon2d_h, lat2d_h = np.meshgrid(lon_h,lat_h)
lat2d_e = ds_ct.variables['XLAT'][0,:,:]
lon2d_e = ds_ct.variables['XLONG'][0,:,:]
#lat2d_h = ds_cp.variables['XLAT'][0,:,:]
#lon2d_h = ds_cp.variables['XLONG'][0,:,:]


# ==============================================================
# Plotting
# ==============================================================
fns= 10
lef , bot,  right, top = 0, 0.5, 1, 0.

cmaps  = 'RdBu_r' #'seismic' ,'twilight_shifted', 'coolwarm'
#levels = np.linspace(19, 32, 13)
levels = np.arange(18, 33, 0.5)
alpbet = [f"({chr(97+i)})" for i in range(9)]

fig, axes = plt.subplots(1, 2, figsize=(6, 4), subplot_kw={'projection': ccrs.PlateCarree()})

i=0
for ax in axes :
    ax.set_extent([slon, elon, slat, elat ], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='10m', linewidth=0.8)
    ax.add_feature(cfeature.LAND, color='lightgrey', alpha=0.3)
    plt.setp(ax.spines.values(), lw=1.2, color='black')
    gl = ax.gridlines(draw_labels=True, linestyle="--", color="gray", alpha=0.3)#dms=False, x_inline=False, y_inline=False)
    gl.top_labels   = False
    gl.left_labels  = False
    print(ax)
    if i == 0: 
       gl.right_labels = False
    gl.xlabel_style = {'size': fns-2}
    gl.ylabel_style = {'size': fns-2}
    i+=1


# 1.ERA5 --------
cf0 = axes[0].contourf(lon2d_e, lat2d_e, sst_ct, levels=levels, cmap=cmaps, extend='both') 
#axes[0].set_title(f'ERA5', fontsize=fns, fontweight='bold')
axes[0].text(0.01, 0.97, f'{(alpbet[0])}ERA5', transform=axes[0].transAxes,
         fontsize=fns, fontweight='bold', va='top', ha='left', color='black')

# 2.HYCOM -------
cf1 = axes[1].contourf(lon2d_h, lat2d_h, sst_cp, levels=levels, cmap=cmaps, extend='both')
#axes[1].set_title(f'HYCOM', fontsize=fns, fontweight='bold')
axes[1].text(0.01, 0.97, f'{(alpbet[1])}HYCOM', transform=axes[1].transAxes,
         fontsize=fns, fontweight='bold', va='top', ha='left', color='black')

plt.tight_layout(rect=[0, 0.2, 0, 0])

cbar_ax1 = fig.add_axes([0.15, 0.18, 0.7, 0.015]) # [left, bottom, width, height]
cbar1 = fig.colorbar(cf0, cax=cbar_ax1, orientation='horizontal')
cbar1.set_label(f'SST [°C]', fontsize=fns, labelpad=0.1)
cbar1.set_ticks(np.arange(18, 33, 2))
cbar1.ax.tick_params(labelsize=fns-1)



plt.savefig(ofn, dpi=600, bbox_inches='tight', pad_inches=0.01)
plt.show()
plt.close()















