#
#	Fig8.Temp_Timeseries.py
#
############################################################

import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc
from netCDF4 import Dataset
from pyproj import Proj

import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.interpolate import interp2d
from matplotlib.patches import Rectangle

import glob
import pytz
import os


# Font ---------
plt.rcParams['font.family']='serif'
plt.rcParams['axes.unicode_minus']=False



# =======================================
PBL = "YSU"
domain = "02"
nvar_b1 = "수온(°C)"
nvar_w1 = "TSK"  
nvar_r1 = "temp"
nvar_b2 = "기온(°C)"
nvar_w2 = "T2"
tt=1718
LEV = "SKIN"
alpbet = [f"({chr(97+i)})" for i in range(6)]


# Date ----
year = "2020"
month= "08"
days = ["18", "19"]

start_time = datetime(2020, 8, 18, 4, 0)  
end_time = datetime(2020, 8, 19, 12, 0)   

start_datetime = datetime(2020, 8, 17, 18, 0)
end_datetime = datetime(2020, 8, 19, 3, 0)

time_range = pd.date_range(start=start_datetime, end=end_datetime, freq='10min')
time_range2= pd.date_range(start=start_datetime, end=end_datetime, freq='1H')

stime_list = [dt.strftime('%Y%m%d%H%M') for dt in time_range]
stime_hourly_list = [dt.strftime('%Y%m%d%H%M') for dt in time_range2]
#---------------------



#  ======================================================================
idr_b= "./"
ifn_b= idr_b+"202008-1819_donghae-set.xlsx"

idr_w = "./"
ifn_w =idr_w+f"sstx-wrfout_Fog_{domain}_2020-08-17_18:00:00.nc"  
ifn_w2=idr_w+f"skin-wrfout_Fog_{domain}_2020-08-17_18:00:00.nc"  

idr_c =f"./"
ifn_c = idr_c+f"cpld_wrfout_Fog_{domain}_2020-08-17_18:00:00.nc" 


ds_ct = nc.Dataset(ifn_w)
ds_sk = nc.Dataset(ifn_w2)
ds_cp = nc.Dataset(ifn_c)



# =========================================================
# function ------------------------------------------------
def read_buoy_coordinates(excel_file):
    locations_df = pd.read_excel(excel_file, sheet_name="위경도")
    coordinates = {}

    for _, row in locations_df.iterrows():
        if pd.notna(row['관측소명']):
            coordinates[row['관측소명']] = {
                'lat': row['위도'],
                'lon': row['경도']
            }

    return coordinates


def read_buoy_data(excel_file, region, start_time, end_time, nvar_b1, nvar_b2):
    coordinates = read_buoy_coordinates(excel_file)


    buoy_df = pd.read_excel(excel_file, sheet_name=region)
    buoy_df["일시"] = pd.to_datetime(buoy_df["일시"])

    filtered_df = buoy_df[(buoy_df["일시"] >= start_time) & (buoy_df["일시"] <= end_time)]
    filtered_df = filtered_df[filtered_df["일시"].dt.minute == 0].copy()


    start_hour = start_time.replace(minute=0, second=0, microsecond=0)
    end_hour = end_time.replace(minute=0, second=0, microsecond=0)
    full_time_range = pd.date_range(start=start_hour, end=end_hour, freq='1H')

    full_df = pd.DataFrame({'일시': full_time_range})

    merged_df = full_df.merge(filtered_df, on='일시', how='left')

    location_info = coordinates[region]
    var_data1 = merged_df[nvar_b1]
    var_data2 = merged_df[nvar_b2]
    time_data = merged_df["일시"]
    ntime = len(time_data)

    nan_count = var_data2.isna().sum()

    data = {
        'region': region,
        'variable1': nvar_b1,
        'variable2': nvar_b2,
        'time': time_data,
        'data1': var_data1,
        'data2': var_data2,
        'lat': location_info['lat'],
        'lon': location_info['lon']
        }


    return data, var_data1, var_data2, ntime



def find_nearest_pixel(lat_sat, lon_sat, target_lat, target_lon):

    distances = np.sqrt((lat_sat - target_lat)**2 + (lon_sat - target_lon)**2)
    min_idx = np.unravel_index(np.argmin(distances), distances.shape)

    return min_idx


def create_datetime_from_string(date_string):

    year = int(date_string[:4])
    month = int(date_string[4:6])
    day = int(date_string[6:8])
    hour = int(date_string[8:10])
    minute = int(date_string[10:12])

    return datetime(year, month, day, hour, minute)


def satellite_single_point(stime_hourly_list, target_lat, target_lon, idr_s, create_datetime_from_string):
     

    gk2a_file = idr_s + f"gk2a_ami_le2_sst_ko020lc_202008181800.nc"
    ds_s = xr.open_dataset(gk2a_file)

    proj = Proj(proj='lcc',
               lat_1=30.0,  
               lat_2=60.0,  
               lat_0=38.0,  
               lon_0=126.0, 
               x_0=0.0,     
               y_0=0.0)     

    x_sat_range = np.linspace(-899000, 899000, ds_s.dims['dim_x'])  
    y_sat_range = np.linspace(899000, -899000, ds_s.dims['dim_y'])  
    xv_sat, yv_sat = np.meshgrid(x_sat_range, y_sat_range)
    lon_sat, lat_sat = proj(xv_sat, yv_sat, inverse=True)

    nearest_idx = find_nearest_pixel(lat_sat, lon_sat, target_lat, target_lon)

    proj_initialized = True     


    datetime_list = []
    temp_sh_list = []


    for i, htime in enumerate(stime_hourly_list):
        dt = create_datetime_from_string(htime)
        datetime_list.append(dt)

        gk2a_file = idr_s + f"gk2a_ami_le2_sst_ko020lc_{htime}.nc"

        if not os.path.exists(gk2a_file):
            print(f"[{i+1:3d}/{len(stime_hourly_list)}] {htime} - File not found, adding NaN")
            temp_sh_list.append(np.nan)
            continue

        try:
            ds_s = xr.open_dataset(gk2a_file)
            temp_s = ds_s['SST'][nearest_idx[0], nearest_idx[1]].values
            temp_s = temp_s.astype(float)
            temp_s[temp_s == 655.35] = np.nan 
            temp_s = temp_s - 273.15  
            temp_sh_list.append(temp_s)

            ds_s.close()


        except Exception as e:
            print(f"[{i+1:3d}/{len(stime_hourly_list)}] {htime} - Error: {e}, adding NaN")
            temp_sh_list.append(np.nan)


    return datetime_list, temp_sh_list





# region -------------
pos_k_dong = ['한수원_기장','임랑해수욕장', '해운대해수욕장', '후포', '경포대해수욕장', '한수원_덕천',
    '대한해협', '울릉도', '울릉도북서', '울릉도북동', '울릉도_기상부이',
    '동해_기상부이', '포항_기상부이', ' 울산_기상부이', '울진_기상부이','거제도_기상부이']
pos_l_dong = ['Gijang','Imrang', 'Haeundae', 'Hupo', 'Gyeongpodae', 'Deokcheon', 'Daehan', 'Ulleung',
              'UlleungNW', 'UlleungNE', 'Ulleung_KMA', 'Donghae', 'Pohang', 'Ulsan_KMA', 'Uljin_KMA', 'Geoje']


pos_l = ["Donghae", "Ulleungdo", "Ulleungdo_NE","Ulleungdo_NW","Dokdo", "Russia","East-South","Ulsan","Uljin","Imrang","imsi"]
pos_cl = ["DON", "ULL", "UNE","UNW","DOK", "RUS","ES","ULS","ULJ","IMR","imsi"]
pos_k = ["동해","울릉도_기상부이","울릉도_북동","울릉도_북서","독도","러시아","동해남쪽","울산_기상부이","울진_기상부이","임랑해수욕장","imsi"]
xlat  = [37.490 ,  37.455,  38.007,  37.743,  37.24 , 40  , 32,  35.2, 36.912, 35.303, 35.6531 ]
xlon  = [129.942, 131.114, 131.553, 130.601, 131.87 , 131 ,129, 129.5, 129.87, 129.293,129.59 ]
lon_idx=[168, 202, 213, 186, 224 , 227, 170, 160, 167 , 154, 161]
lat_idx=[135, 136, 157, 145, 130 , 316, 88 ,  51, 114 , 55 , 67 ]


line_w = 2.1
font_w = 10
t1_ymin, t1_ymax = 22.5, 29 
t2_ymin, t2_ymax = 20.5, 29 
t3_ymin, t3_ymax = 15.5, 29 


fig, axs = plt.subplots(3, 2, figsize=(9, 10),gridspec_kw={'hspace': 0.22, 'wspace': 0.3})



time_labels =["06\nAug.18", "12", "18", "00\nAug.19", "06", "12"]
tick_indices=[   2,  2+6,  8+6,  14+6,  20+6, 26+6  ]

region_points = [1, 8, 9] 

for idx, position in enumerate(region_points):

    region = pos_k[position]

    alpha=0
    
    
    # ============================================
        
    data, sst_b, t2m_b, ntime_b = read_buoy_data(ifn_b, region, start_time, end_time,nvar_b1, nvar_b2)
        
    target_lat = data['lat']
    target_lon = data['lon']
        
        
    # 2. Model ------
    start_time_utc = start_time - timedelta(hours=9)
    end_time_utc = end_time - timedelta(hours=9)
    
    times_w = ds_ct.variables['Times'][:]
    times = []
    for t in times_w:
         time_str = ''.join([c.decode('utf-8') for c in t]).replace('_', ' ')
         times.append(datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S'))
    
    time_mask = [(start_time_utc <= t <= end_time_utc) for t in times]
    time_indices = [i for i, mask in enumerate(time_mask) if mask]
    filtered_times = [times[i] for i in time_indices]
    
    ntime  = len(time_indices)
    lml_ct = ds_ct.variables['LANDMASK'][0, lat_idx[position], lon_idx[position]]
    lml_cp = ds_cp.variables['LANDMASK'][0, lat_idx[position], lon_idx[position]]
    
    # T2m - from WRF. ***
    t2m_ct = ds_ct.variables['T2'][time_indices, lat_idx[position], lon_idx[position]]
    t2m_sk = ds_sk.variables['T2'][time_indices, lat_idx[position], lon_idx[position]]
    t2m_cp = ds_cp.variables['T2'][time_indices, lat_idx[position], lon_idx[position]]
    
    # SST - from WRF./ROMS. ***
    sst_ct = ds_ct.variables['TSK'][time_indices, lat_idx[position], lon_idx[position]]
    sst_sk = ds_sk.variables['TSK'][time_indices, lat_idx[position], lon_idx[position]]
    sst_cp = ds_cp.variables['TSK'][time_indices, lat_idx[position], lon_idx[position]]
    
    
    t2m_ct = np.where(lml_ct == 0, t2m_ct, np.nan)- 273.15
    t2m_sk = np.where(lml_ct == 0, t2m_sk, np.nan)- 273.15
    t2m_cp = np.where(lml_cp == 0, t2m_cp, np.nan)- 273.15
    
    sst_ct = np.where(lml_ct == 0, sst_ct, np.nan)- 273.15
    sst_sk = np.where(lml_ct == 0, sst_sk, np.nan)- 273.15
    sst_cp = np.where(lml_cp == 0, sst_cp, np.nan)- 273.15    

    
    # 3.GK2A -------
    idr_s = f"./"
    stime_list2, sst_s_list = satellite_single_point(stime_hourly_list, target_lat, target_lon, idr_s,create_datetime_from_string)
    
    sst_s_array = np.array(sst_s_list)
    


    # ==============================================================
    # Plotting
    # ==============================================================
    gk2a_times = pd.to_datetime(stime_list2)
    ntime2 = len(gk2a_times)
    gtime = np.arange(ntime2)
    btime= np.arange(ntime_b)
    time = np.arange(ntime)
    
    
    # Panel 1: Temperature(t2m) ---
    ax1 = axs[idx, 0]
   

    ax1.plot(btime, t2m_b , label="BUOY", color='black',linewidth=line_w-0.5) 
    ax1.plot( time, t2m_ct, label='CNTL', color='#6F8FAF' ,linewidth=line_w-0.1) 
    ax1.plot( time, t2m_sk, label='SKIN', color='#0066CC' ,linewidth=line_w-0.1) 
    ax1.plot( time, t2m_cp, label='CPLD', color='#CA3433'  ,linewidth=line_w-0.1)
   
    if idx == 0 : 
        ax1.set_ylim([t1_ymin, t1_ymax])
    elif idx == 1 : 
        ax1.set_ylim([t2_ymin, t2_ymax])
    elif idx == 2 : 
        ax1.set_ylim([t3_ymin, t3_ymax])

    ax1.tick_params(axis='y' , labelsize=font_w)
    ax1.set_ylabel('T2m [°C]', fontsize=font_w )
    ax1.yaxis.set_major_locator(plt.MultipleLocator(2))
    
    ax1.set_xticks(tick_indices)
    ax1.set_xticklabels(time_labels)

    if idx == 2 : 
        ax1.set_xlabel('Time [LST]',fontsize=font_w)

    ax1.text(0.02, 0.98, f'{(alpbet[2*idx])} {pos_cl[position]}', transform=ax1.transAxes, 
             fontsize=font_w+1, verticalalignment='top', fontweight='bold')
    
    

    # Panel 2: Temperature(sst/tsk) ---
    ax2 = axs[idx, 1]
    
    ax2.plot(btime, sst_b , label="BUOY" , color='black',linewidth=line_w-0.5, linestyle='-')
    ax2.plot(gtime, sst_s_array, label=f'GK2A', color='black',linewidth=line_w-0.5, linestyle='--')
    ax2.plot( time, sst_ct, label=f'CNTL', color='#6F8FAF' ,linewidth=line_w-0.1)
    ax2.plot( time, sst_sk, label=f'SKIN', color='#0066CC' ,linewidth=line_w-0.1)
    ax2.plot( time, sst_cp, label=f'CPLD', color='#CA3433' ,linewidth=line_w-0.1)
    
    if idx == 0 :
        ax2.set_ylim([t1_ymin, t1_ymax])
    elif idx == 1 :
        ax2.set_ylim([t2_ymin, t2_ymax])
    elif idx == 2 :
        ax2.set_ylim([t3_ymin, t3_ymax])


    ax2.tick_params(axis='y', labelsize=font_w)
    ax2.set_ylabel('SST [°C]',fontsize=font_w )
    ax2.yaxis.set_major_locator(plt.MultipleLocator(2))
    
    ax2.set_xticks(tick_indices)
    ax2.set_xticklabels(time_labels)
    if idx == 2 :
        ax2.set_xlabel('Time [LST]',fontsize=font_w)

    ax2.text(0.02, 0.98, f'{(alpbet[2*idx+1])} {pos_cl[position]}', transform=ax2.transAxes,
             fontsize=font_w+1, verticalalignment='top', fontweight='bold')
    
    
    if idx == 0:
        ax1.legend(loc='upper right', ncol=2, fontsize=font_w-2, frameon=False,
                    handlelength=1.2, handletextpad=0.4,columnspacing=0.7,labelspacing=0.25)   
        ax2.legend(loc='upper right', ncol=3, fontsize=font_w-2, frameon=False,
                    handlelength=1.2, handletextpad=0.4,columnspacing=0.7,labelspacing=0.25)

    for spine in ax1.spines.values():
        spine.set_linewidth(1.2)
    for spine in ax2.spines.values():
        spine.set_linewidth(1.2)
    
    plt.tight_layout()

plt.show()
plt.close()


