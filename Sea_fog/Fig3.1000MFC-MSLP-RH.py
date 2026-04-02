#
#	Fig3.1000MFC-MSLP-RH.py
#       - Moisture Flux and Convergence with ncl calculation
#       - MSLP (pattern) with H/L pressure
#       - 'RH'
############################################################

import numpy as np
import netCDF4 as nc
import xarray as xr
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datetime
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from scipy.ndimage import gaussian_filter
from scipy.ndimage import maximum_filter, minimum_filter
import os

# Font ---------
plt.rcParams['font.family']='serif'
plt.rcParams['axes.unicode_minus']=False


# ===========================================
year = "2020"
month= "08"
day  = np.arange(17,20+1)

# Time index -------
day_date  = ["0000LST 18 AUG", "1200LST 18 AUG","0000LST 19 AUG"]
day_ofn  = ["00_18", "12_18", "00_19"]
date_idxp = [21+24-6,  21+6+24, 27+6+24+6]
date_idxs = [21+24+24-6,  21+6+24+24, 27+6+24+6+24]


# Domain -----------
ilon=120;elon=137;ilat=30;elat=45

pressure_level = [1000, 975, 950, 925, 900, 850, 800, 750, 650, 550, 450, 300, 200]
level   = 1000
level_s = "1000"


EXP_name = "Moisture-Flux-Convergence-MSLP-2mRH"
opath="./Fig/ADV-VMF/"
os.makedirs(opath, exist_ok=True)


# ============================================
# Read file         ==========================
idr ="./"
ifnp=idr+"ERA5_hour-pres_2020-08-1619.nc"
ifns=idr+"ERA5_hour_sfc_2020-08-1519.nc"

idrl="./"
ifnl=idrl+"era5_hour_landmask.nc"

ds = xr.open_dataset("./mfc_output.nc")
dsp= xr.open_dataset(ifnp)
dss= xr.open_dataset(ifns)
dsl= xr.open_dataset(ifnl)


mfc_adv = ds["mfc_adv"]
mfc     = ds["mfc"]
uq	= ds["uq"]
vq      = ds["vq"]

psfc=dss["msl"]  
psfc=psfc.reindex(latitude=list(reversed(psfc.latitude)))

tsfc=dss["t2m"]
dsfc=dss["d2m"]
tsfc=tsfc.reindex(latitude=list(reversed(tsfc.latitude)))
dsfc=dsfc.reindex(latitude=list(reversed(dsfc.latitude)))

land=dsl["lsm"].isel(valid_time=0)
land=land.reindex(latitude=list(reversed(land.latitude)))


# ===========================================
# Function ==================================
def calculate_RH(t2m, d2m) :

    es = 6.112 * np.exp(17.67 * t2m / (t2m+243.5)) 
    e  = 6.112 * np.exp(17.67 * d2m / (d2m+243.5)) 
    rh = (e/es) * 100

    return rh



# ==========================================================
# (1) Calculate Moisture Flux Convergence ==================
# (Horizontal) : -V*∇q = -(u*dq/dx + v*dq/dy) ==============
mfc_day = []
uq_day  = []
vq_day  = []
p_day   = []
r2m_day = []

lat_min = max(tsfc.latitude.min().values, land.latitude.min().values)
lat_max = min(tsfc.latitude.max().values, land.latitude.max().values)
lon_min = max(tsfc.longitude.min().values, land.longitude.min().values)
lon_max = min(tsfc.longitude.max().values, land.longitude.max().values)

tsfc = tsfc.sel(latitude=slice(lat_min, lat_max),
                longitude=slice(lon_min, lon_max)) 
dsfc = dsfc.sel(latitude=slice(lat_min, lat_max),
                longitude=slice(lon_min, lon_max))
land = land.sel(latitude=slice(lat_min, lat_max),
                longitude=slice(lon_min, lon_max))


for ii, i in enumerate(date_idxp):

    # (1) MFC =============================================
    mfc_data = mfc.isel(time=ii)
    uq_data  = uq.isel(time=ii)
    vq_data  = vq.isel(time=ii)

    mfc_day.append(mfc_data)
    uq_day.append(uq_data)
    vq_day.append(vq_data)
  

    # (2) MSLP ============================================
    p_data = psfc.isel(valid_time=date_idxs[ii]) / 1e2  # Pa->hPa 
    p_day.append(p_data)


    # (3) RH 2m ==========================================
    tsfc_data = tsfc.isel(valid_time=date_idxs[ii])
    dsfc_data = dsfc.isel(valid_time=date_idxs[ii])
    r2m_data  = calculate_RH(tsfc_data, dsfc_data)
    r2m_data  = r2m_data.where(land<=0.5, np.nan)
    r2m_day.append(r2m_data)



# =============================================================
#  Plotting
# =============================================================
lons_mfc, lats_mfc = mfc_data.longitude, mfc_data.latitude
lons_msl, lats_msl = p_data.longitude, p_data.latitude
lons_rh, lats_rh = r2m_data.longitude, r2m_data.latitude

mfc_vmin, mfc_vmax = -10, 10
mfc_levels = [-10, -8, -6, -4, -2, -1, 0, 1, 2, 4, 6, 8, 10]
msl_vmin, msl_vmax, msl_l = 1004, 1015, 6  
rh_vmin, rh_vmax, rh_l = 95, 100, 6

alpbet = [f"({chr(97+i)})" for i in range(12)]


fig = plt.figure(figsize=(15, 13))
proj = ccrs.PlateCarree()

height_ratios = [7, 0.1, 1.8, 7, 0.1]  
gs = gridspec.GridSpec(5, 3, figure=fig, 
                      height_ratios=height_ratios,
                      hspace=0.04,  
                      wspace=0.02,  
                      left=0.02, right=0.98, 
                      top=0.98, bottom=0.2)


for i in range(len(date_idxp)*2):

    if i < 3:
        ax = fig.add_subplot(gs[0, i], projection=proj) 
    else:
        ax = fig.add_subplot(gs[3, i-3], projection=proj)  


    # 격자선 추가
    gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False, 
                       linewidth=0.3, linestyle='--', color='gray', alpha=0.5)
    gl.top_labels   = False
    gl.bottom_labels= False
    gl.right_labels = False
    gl.left_labels  = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}


    ax.set_extent([ilon, elon, ilat, elat], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    ax.add_feature(cfeature.LAND, color='lightgrey', alpha=0.5)
    ax.add_feature(cfeature.OCEAN, color='white')


    if i < 3 :
       # 1-1. MSLP ---------
       p_smooth   = gaussian_filter(p_day[i].values, sigma=1.5) 
       msl_levels = np.linspace(msl_vmin, msl_vmax, msl_l) 
       msl_moisture = ax.contour(lons_msl, lats_msl, p_smooth, levels=msl_levels, colors="k", linewidths=1.8)
       ax.clabel(msl_moisture, levels=msl_moisture.levels, inline=True, fmt='%d', inline_spacing=2)

       hl_positions = [
            [{"label": "H", "lon": 125.5, "lat": 32.2, "color": "blue"}],
            [{"label": "H", "lon": 126.3, "lat": 33., "color": "blue"}],
            [{"label": "H", "lon": 126.0, "lat": 33.5, "color": "blue"}],
            ]
       
       for item in hl_positions[i]:
            ax.text(item["lon"], item["lat"], item["label"], color=item["color"]
                   ,fontsize=16, fontweight='bold', ha='center', va='center', transform=ccrs.PlateCarree() )
 

       # 1-3. RH 2m --------
       rh_levels = np.linspace(rh_vmin, rh_vmax, rh_l)  
       r2m_moisture = ax.contourf(lons_rh, lats_rh, r2m_day[i], cmap='Blues', levels=rh_levels,
                           alpha=0.8, transform=ccrs.PlateCarree()) 
       

       ax.text(0.02, 0.97, f"{alpbet[i]} {day_date[i]}", transform=ax.transAxes,
               fontsize=12, fontweight='bold', va='top', ha='left', color='black'
               ,backgroundcolor="white", alpha=0.7)

    else:
       idx = i-3
       # 2. Moisture Flux Convergence ---
       cf_mfc = ax.contourf(lons_mfc, lats_mfc, mfc_day[idx], cmap='BrBG', levels=mfc_levels,
                           extend='both',alpha=0.8, transform=ccrs.PlateCarree() )

       # 2. Moisture flux vector -------------------
       skip  = 5 
       ref_vector = 100 
       cv = ax.quiver(lons_mfc[::skip], lats_mfc[::skip],
                      uq_day[idx][::skip, ::skip],
                      vq_day[idx][::skip, ::skip],
                      scale=ref_vector ,  
                      scale_units='xy',
                      color='black',
                      alpha=1,
                      width=0.007,
                      transform=ccrs.PlateCarree()
                      )
       if idx == 0:
           qk = ax.quiverkey(cv, 0.1, -0.12, ref_vector, rf'10$^{2}$ (g kg$^{{-1}}$ m s$^{{-1}}$)',
                      labelpos='S', coordinates='axes', fontproperties={'size': 10})
           qk.text.set_bbox(dict(facecolor='white', edgecolor='white', 
                          boxstyle='round',pad=0.5, alpha=0.9))
 
       
       ax.text(0.5, 0.97, alpbet[i], transform=ax.transAxes,
               fontsize=12, fontweight='bold', va='top', ha='left', color='black'
              ,backgroundcolor="white", alpha=0.7)

    if i>2:
        gl.bottom_labels = True
    if (i==2 or i==5) :
        gl.right_labels = True

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_edgecolor('black')


cbar_ax1 = fig.add_axes([0.15, 0.61, 0.7, 0.012])  
cbar1 = plt.colorbar(r2m_moisture, cax=cbar_ax1, orientation='horizontal')
cbar1.set_label(f'2 m Relative Humidity [%]', fontsize=15) 
cbar1.ax.tick_params(labelsize=15)

cbar_ax2 = fig.add_axes([0.15, 0.15, 0.7, 0.012]) 
cbar2 = plt.colorbar(cf_mfc, cax=cbar_ax2, orientation='horizontal')
cbar2.set_label('Moisture Convergence [g kg$^{-1}$ s$^{-1}$ (10$^{-5}$)]', 
                fontsize=15) 
cbar2.ax.tick_params(labelsize=15)


plt.show()
plt.close("all")
