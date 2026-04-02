#
#	Fig2.SST-OBS_mean.py	
#       - 1day :Daily mean of SST (Buoy & GK2A)
#       - 2 panels: (1) Buoy observations, (2) Satellite-Buoy difference
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

# Font ---------
plt.rcParams['font.family']='serif'
plt.rcParams['axes.unicode_minus']=False



# Information Setting =================================
nvar_b1 = "수온(°C)"
depth_level = "skin"

preofn = f"SST-Buoy-vs-GK2A_dailymean-diff"

dates_kst = [
    datetime(2020, 8, 18),
    datetime(2020, 8, 19)
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




# DOMAIN-AREA ----
domain = "02"

lat_max =45.0 ; elat = 39  
lat_min =33.0 ; slat = 32.5 
lon_max =138.0; elon = 133 
lon_min =125.0; slon = 122 



# Function ==========================================
def read_buoy_coordinates(excel_file):
    try:
        locations_df = pd.read_excel(excel_file, sheet_name="위경도")
        coordinates = {}
        for _, row in locations_df.iterrows():
            if pd.notna(row['지점명']):
                coordinates[row['지점명']] = {
                    'lat': row['위도'],
                    'lon': row['경도']
                }
        return coordinates
    except Exception as e:
        print(f"Error : {e}")
        return {}


def read_buoy_data(excel_file, region, time_list_kst, nvar_b1, coordinates):

    try:
        buoy_df = pd.read_excel(excel_file, sheet_name=region)
    except:
        return None


    buoy_df["일시"] = pd.to_datetime(buoy_df["일시"])
    location_info = coordinates[region]

    hourly_data = []
    for time in time_list_kst:
        time_data = buoy_df[buoy_df["일시"] == time]
        if len(time_data) > 0 :
            value = time_data[nvar_b1].values[0]
            hourly_data.append(value)
        else:
            hourly_data.append(np.nan)


    data = {
        'region': region,
        'lat': location_info['lat'],
        'lon': location_info['lon'],
        'time': time_list_kst,
        'variable1': nvar_b1,
        'hourly_data': np.array(hourly_data)
        }

    return data



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


def find_nearest_grid_point(lat_grid, lon_grid, target_lat, target_lon):
    dist = np.sqrt((lat_grid - target_lat)**2 + (lon_grid - target_lon)**2)
    y_idx, x_idx = np.unravel_index(np.argmin(dist), dist.shape)
    return y_idx, x_idx




# =========================================================
# Read & Load file                                        =
# =========================================================
date_labels = ["2020-08-18"] 

ifn_b1 = "KMA_Buoy_2020081720_nurion.xlsx"
idr_s = f"./"

proj = Proj(proj='lcc',
        lat_1=30.0,
        lat_2=60.0,
        lat_0=38.0,
        lon_0=126.0,
        x_0=0.0,
        y_0=0.0)

daily_buoy_means = []
lat_sat = None
lon_sat = None

coordinates1 = read_buoy_coordinates(ifn_b1)


pos_k = ["울릉도","동해","포항","울산","울진","동해78", "동해57","고성","삼척"
         ,"덕적도","칠발도","외연도","신안","인천", "부안","서해170", "서해206", "홍도","서해190","풍도","가거도"
         ,"거문도", "거제도","마라도","추자도","서귀포", "통영","남해239","남해244","남해111"]

pos_l = ["Ulleungdo", "Donghae", "Pohang", "Ulsan", "Uljin","Donghae78","Donghae57","Gosung","Samchuk"
        ,"Deokjeokdo","Chilbaldo", "Oeyeondo", "Sinan", "Incheon", "Buan", "WestSea170","WestSea206", "Hongdo","Seahae190","Pungdo","Gageodo"
        , "Geomundo", "Geojedo", "Marado", "Chujado", "Seogwipo", "Tongyeong","Namhae239","Namhae244","Namhae111"]


all_daily_data=[]

for day_idx in range(len(dates_kst)):
    date_kst = dates_kst[day_idx]
    time_list_kst = time_lists_kst[day_idx]
    time_list_utc = time_lists_utc[day_idx]

    temp_s_list, valid_mask_list, sat_coords = read_gk2a_hourly(time_list_utc, idr_s, proj)
    lat_sat, lon_sat = sat_coords

    temp_s_array = np.array(temp_s_list)
    sst_mean_s = np.nanmean(temp_s_array, axis=0)

    buoy_data_list=[]
    for region in pos_k:
        buoy_data = read_buoy_data(ifn_b1, region, time_list_kst, nvar_b1, coordinates1)
        if buoy_data is not None:
            buoy_data_list.append(buoy_data)


    filtered_buoy_data = {
        'regions': [],
        'lats': [],
        'lons': [],
        'buoy_sst_mean': [],
        'satellite_sst_mean': [],
        'difference': []  # satellite - buoy
    }
    
    for buoy in buoy_data_list:
        region = buoy['region']
        lat = buoy['lat']
        lon = buoy['lon']

        grid_y, grid_x = find_nearest_grid_point(lat_sat, lon_sat, lat, lon)

        valid_times = []
        buoy_sst_values = []
        sat_sst_values = []
    
        for t in range(len(temp_s_list)):
            sat_value = temp_s_list[t][grid_y, grid_x]
            buoy_value = buoy['hourly_data'][t]
    
            if not np.isnan(sat_value) and not np.isnan(buoy_value):
                valid_times.append(t)
                buoy_sst_values.append(buoy_value)
                sat_sst_values.append(sat_value)
    
        if len(valid_times) > 0:
            buoy_mean = np.nanmean(buoy_sst_values)
            sat_mean = np.nanmean(sat_sst_values)
            diff = sat_mean - buoy_mean
    
            filtered_buoy_data['regions'].append(region)
            filtered_buoy_data['lats'].append(lat)
            filtered_buoy_data['lons'].append(lon)
            filtered_buoy_data['buoy_sst_mean'].append(buoy_mean)
            filtered_buoy_data['satellite_sst_mean'].append(sat_mean)
            filtered_buoy_data['difference'].append(diff)
    
    
    all_daily_data.append({
        'date': date_kst,
        'gk2a_mean': sst_mean_s,
        'buoy_data': filtered_buoy_data,
        'lat_sat': lat_sat,
        'lon_sat': lon_sat
    })



# ================================================
# Plotting 
# ================================================
fns=11
alpbet = [f"({chr(97+i)})" for i in range(4)]


fig, axes = plt.subplots(2, 2, figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()},gridspec_kw={'hspace': -0.51, 'wspace': 0.05})
                          

for ax_row in axes:
    for ax in ax_row:
        ax.set_extent([slon, elon, slat, elat], crs=ccrs.PlateCarree())
        ax.coastlines(resolution='10m', linewidth=1.2)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_edgecolor('black')
        
        ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5, linestyle='--')


vmin1, vmax1 = 23, 30
levels1 = np.arange(vmin1, vmax1+0.1, 0.5)

vmin2, vmax2 = -2, 2
levels2 = np.arange(vmin2, vmax2+0.1, 0.5)


import matplotlib as mpl
n_colors1 = len(levels1) - 1
n_colors2 = len(levels2) - 1
colors1 = plt.cm.rainbow(np.linspace(0, 1, n_colors1))
colors2 = plt.cm.RdBu_r(np.linspace(0, 1, n_colors2))
cmap1 = mpl.colors.ListedColormap(colors1)
cmap2 = mpl.colors.ListedColormap(colors2)
norm1 = mpl.colors.BoundaryNorm(levels1, n_colors1)
norm2 = mpl.colors.BoundaryNorm(levels2, n_colors2)


for day_idx in range(len(all_daily_data)):
    daily_data = all_daily_data[day_idx]
    date_str = daily_data['date'].strftime('%Y-%m-%d')

    ax_left = axes[day_idx, 0]

    # 1) GK2A plot(contour) --------
    cf = ax_left.contourf(daily_data['lon_sat'], daily_data['lat_sat'], daily_data['gk2a_mean']
                    , levels=levels1, cmap=cmap1, norm=norm1, extend='both', transform=ccrs.PlateCarree())

    ax_left.text(0.02, 0.98, f'{(alpbet[2*day_idx])}', transform=ax_left.transAxes,
             fontsize=fns+1, verticalalignment='top', fontweight='bold')

    buoy_data = daily_data['buoy_data']
    ax_left.scatter(buoy_data['lons'], buoy_data['lats'], c=buoy_data['buoy_sst_mean'],
                     cmap=cmap1, norm=norm1, s=80, edgecolors='black', linewidth=1.3, zorder=5, transform=ccrs.PlateCarree())

    # 2) GK2A-Buoy -------
    ax_right=axes[day_idx, 1]

    if len(buoy_data['regions']) > 0:  
        sc = ax_right.scatter(buoy_data['lons'], buoy_data['lats'],  
                      c=buoy_data['difference'], cmap=cmap2, norm=norm2,
                      s=80, edgecolors='black', linewidth=1.3, zorder=5, transform=ccrs.PlateCarree())

    ax_right.text(0.02, 0.98, f'{(alpbet[2*day_idx+1])}', transform=ax_right.transAxes,
             fontsize=fns+1, verticalalignment='top', fontweight='bold')
     
     
left1, bottom1, width1, height1 = 0.13, 0.23, 0.35, 0.01
left2, bottom2, width2, height2 = 0.53, 0.23, 0.35, 0.01

cbar_ax1 = fig.add_axes([left1, bottom1, width1, height1])
cbar1 = fig.colorbar(cf, cax=cbar_ax1, orientation='horizontal', extend='both', spacing='uniform')
cbar1.set_label(f'SST [°C]', fontsize=fns)
cbar1.ax.tick_params(labelsize=fns-1)
tick_positions1 = np.arange(int(vmin1), int(vmax1)+1, 1.0) 
cbar1.set_ticks(tick_positions1)

cbar_ax2 = fig.add_axes([left2, bottom2, width2, height2])
sm = plt.cm.ScalarMappable(cmap=cmap2, norm=norm2)
sm.set_array([])
cbar2 = fig.colorbar(sm, cax=cbar_ax2, orientation='horizontal', extend='both')
cbar2.set_label(f'SST Difference(GK2A-Buoy) [°C]', fontsize=fns)
cbar2.ax.tick_params(labelsize=fns-1)
tick_positions = np.arange(int(vmin2), int(vmax2)+1, 1.)
cbar2.set_ticks(tick_positions)


plt.show()
plt.close()
