#
#	Fig1.Model-Domain.py	
#
############################################################

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.patheffects as pe


# font ---------
plt.rcParams['font.family']='serif'
plt.rcParams['axes.unicode_minus']=False



# ================================================
idr = "/scratch/x3158a03/DATA/Topo/"
ifn = idr + "ETOPO1_Bed_g_gmt4.grd"


ACTUAL_DOMAINS = {
    'domain1': { 
        'lon_min': 118.0, 'lon_max': 140.0,
        'lat_min': 27.0, 'lat_max': 48.0,
        'color': 'black',
        'linewidth': 3,
        'label': 'D01'
    },
    'domain2': { 
        'lon_min': 124.0, 'lon_max': 138.0,
        'lat_min': 33.0, 'lat_max': 45.0,
        'color': 'red',
        'linewidth': 2,
        'label': 'D02'
    }
}

pos_l = ["ULL", "ULJ", "IMR", "A"]
lat_l = [37.456,  36.907, 35.303,  37.3]
lon_l = [131.114,129.874,129.293, 127.6]



# Funtion ========================================

def load_etopo_data():
    try:
        bathy = xr.open_dataset(ifn)

        if 'z' in bathy.variables:
            depth = bathy['z']
        elif 'elevation' in bathy.variables:
            depth = bathy['elevation']
        else:
            depth_var = list(bathy.data_vars)[0]
            depth = bathy[depth_var]

        return depth

    except Exception as e:
        print(f"Error : {e}")
        return None

def create_actual_wrf_map():
    depth = load_etopo_data()
    if depth is None:
        return None, None, None

    margin = 2.0
    lon_min = ACTUAL_DOMAINS['domain1']['lon_min'] - margin
    lon_max = ACTUAL_DOMAINS['domain1']['lon_max'] + margin
    lat_min = ACTUAL_DOMAINS['domain1']['lat_min'] - margin
    lat_max = ACTUAL_DOMAINS['domain1']['lat_max'] + margin


    map_bathy = depth.sel(
        x=slice(lon_min, lon_max),
        y=slice(lat_min, lat_max)
    )

    # ============================================
    # Plotting 
    # ============================================
    fig = plt.figure(figsize=(10, 8))

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(resolution='50m', linewidth=1.0, color='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, color='black')
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.7)


    # 1. Depth(ocean)
    ocean_levs = np.arange(-3000,0, 50)
    ocean_data = map_bathy.where(map_bathy <= 0)
    cs_ocean = ocean_data.plot.contourf(ax=ax, x='x',y='y', levels=ocean_levs, cmap='Blues_r', transform=ccrs. PlateCarree(),
                                        add_colorbar=False, extend='min')
    # 2. Height(land)
    land_levs = np.arange(0, 1000, 50)
    land_data = map_bathy.where(map_bathy > 0)
    cs_land = land_data.plot.contourf(ax=ax, x='x', y='y', levels=land_levs, cmap='summer_r', transform=ccrs.PlateCarree(),
                                 add_colorbar=False, extend='max')    

    cbar_ocean = plt.colorbar(cs_ocean, ax=ax, shrink=0.6, pad=0.0, aspect=30)
    cbar_ocean.set_label('Depth [m]', fontsize=12, rotation=270, labelpad=20) 
    cbar_ocean.ax.tick_params(labelsize=10)
    cbar_ocean.ax.yaxis.set_ticks_position('none')
 
    cbar_land = plt.colorbar(cs_land, ax=ax, shrink=0.6, pad=0.03, aspect=30, location='right')
    cbar_land.set_label('Elevation [m]', fontsize=12, rotation=270, labelpad=20)
    cbar_land.ax.tick_params(labelsize=10)
    cbar_land.ax.yaxis.set_ticks_position('none')


    # 2) Adding WRF domian box ******
    for domain_name, domain_info in ACTUAL_DOMAINS.items():
        lon_coords = [
            domain_info['lon_min'], domain_info['lon_max'],
            domain_info['lon_max'], domain_info['lon_min'],
            domain_info['lon_min']
        ]
        lat_coords = [
            domain_info['lat_min'], domain_info['lat_min'],
            domain_info['lat_max'], domain_info['lat_max'],
            domain_info['lat_min']
        ]

        ax.plot(lon_coords, lat_coords,
               color=domain_info['color'],
               linewidth=domain_info['linewidth'],
               transform=ccrs.PlateCarree())

        label_lon = domain_info['lon_max'] - 1.0
        label_lat = domain_info['lat_max'] - 1.0

        ax.text(label_lon, label_lat,
               domain_info['label'],
               ha='right', va='top',
               color=domain_info['color'],
               fontweight='bold', fontsize=15,
               transform=ccrs.PlateCarree())

    ax.text(133, 38.9, 'East Sea', ha='center', va='bottom',
       fontsize=15, fontweight='bold', color='black',
       path_effects=[pe.withStroke(linewidth=3, foreground='white')])

    ax.text(133, 38.9, '(Sea of Japan)', ha='center', va='top',  
           fontsize=12, color='black',
           path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

    ax.text(123.5, 35.5, 'Yellow\nSea', ha='center', va='center',
           fontsize=15, fontweight='bold', color='black',
           path_effects=[pe.withStroke(linewidth=3, foreground='white')])


    # 3) Adding Buoy Points ******
    for i, (name, lat, lon) in enumerate(zip(pos_l, lat_l, lon_l)): 
        if i < (len(pos_l)-1) :
            ax.plot(lon, lat, 'x', markersize=6, markerfacecolor="black", markeredgecolor='black', markeredgewidth=2,
                    transform=ccrs.PlateCarree())
        
            ax.text(lon-0.3, lat-0.9, name, ha='left', va='bottom', fontsize=9, fontweight='bold', color='black',
                    path_effects=[pe.withStroke(linewidth=2.5, foreground='white')],
                    transform=ccrs.PlateCarree())
        else:
            ax.text(lon-0.1,lat+0.1, name, ha='left', va='bottom', fontsize=9, fontweight='bold',
                    color='black',
                    path_effects=[pe.withStroke(linewidth=2.5, foreground='white')],
                    transform=ccrs.PlateCarree())
            ax.text(lon_l[0], lat_l[0]+0.1, 'B', ha='left', va='bottom', fontsize=9, fontweight='bold', color='black',
                    path_effects=[pe.withStroke(linewidth=2.5, foreground='white')],
                    transform=ccrs.PlateCarree())

    # 4) Adding Path (A->ULL:B) ******
    ax.plot([lon_l[3],lon_l[0]], [lat_l[3],lat_l[0]], color='black', linewidth=2, linestyle='-', transform=ccrs.PlateCarree(), zorder=5)    



    gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False,
                     linewidth=0, color='gray', alpha=0, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    plt.tight_layout()
    plt.show()

    return fig, ax, map_bathy



