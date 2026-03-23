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

# font ---------
plt.rcParams['font.family']='serif'
plt.rcParams['axes.unicode_minus']=False

# Define Infomation ========================
domain = "01"
tt   = 1718
nvar = "TSK" # "T2" # "TSK"


# DOMAIN-AREA
elat = 42  #43  #42
slat = 34  #33  #34
elon = 136 #133 #136.0
slon = 128 #123 #128

# ulleungdo 영역
ull_coast_slat = 37.0
ull_coast_elat = 38.5
ull_coast_slon = 130.5
ull_coast_elon = 131.5

# 중부 연안 영역
mid_coast_slat = 36.8
mid_coast_elat = 38.0 # 39
mid_coast_slon = 128.8
mid_coast_elon = 130.0 #131

# 남부 연안 영역
south_coast_slat = 35.5
south_coast_elat = 36.0
south_coast_slon = 129.2
south_coast_elon = 129.7




# DATE -----
year  = "2020"
month = "08"
days  = ["17","18", "19", "20"]

dates_kst = [
    datetime(2020, 8, 18),
 #   datetime(2020, 8, 19)
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



# save ----
opath = f"./Fig/SST/"
os.makedirs(opath, exist_ok=True)


# Function  ==========================================
def read_gk2a_hourly(time_list_utc, idr_s, proj_params):
    """
    시간별 GK2A 위성 데이터를 읽고 평균 계산
    """
    temp_s_list = []
    valid_mask_list = []

    lat_sat = None
    lon_sat = None

    for i, time in enumerate(time_list_utc):
        # 파일명 생성 (YYYYMMDDHHMM 형식)
        time_str = time.strftime("%Y%m%d%H00")
        gk2a_file = idr_s + f"gk2a_ami_le2_sst_ko020lc_{time_str}.nc"

        try:
            ds_s = xr.open_dataset(gk2a_file)

            # 첫 번째 파일에서만 좌표 계산
            if i == 0:
                x_sat_range = np.linspace(-899000, 899000, ds_s.dims['dim_x'])
                y_sat_range = np.linspace(899000, -899000, ds_s.dims['dim_y'])
                xv_sat, yv_sat = np.meshgrid(x_sat_range, y_sat_range)
                lon_sat, lat_sat = proj_params(xv_sat, yv_sat, inverse=True)


            temp_s = ds_s['SST'].values.astype(float)

            valid_mask = (temp_s != 65535) & (~np.isnan(temp_s))
            temp_s[~valid_mask] = np.nan
            temp_s[valid_mask] = temp_s[valid_mask] - 273.15  # K to Celsius

            temp_s_list.append(temp_s)
            valid_mask_list.append(valid_mask)

            ds_s.close()

            kst_time = time + timedelta(hours=9)
          #  print("kst_time(satellite) = ",kst_time)  =  2020-08-18 00:00:00

        except Exception as e:
            print(f"File not found or error for {time_str}: {e}")
            # 파일이 없으면 NaN 배열 추가
            if i == 0:
                print("Cannot proceed without first file for coordinates")
                return None, None, None
            temp_s_list.append(np.full_like(temp_s_list[0], np.nan))
            valid_mask_list.append(np.zeros_like(valid_mask_list[0], dtype=bool))

    return temp_s_list, valid_mask_list, (lat_sat, lon_sat)


def draw_box(ax, slat, elat, slon, elon, color='red', linestyle='-', linewidth=1.7, label=None):
    # 박스의 모서리 좌표 생성
    lons = [slon, elon, elon, slon, slon]  # 시작점으로 다시 돌아와 닫힌 사각형 만들기
    lats = [slat, slat, elat, elat, slat]

    # 박스 그리기
    ax.plot(lons, lats, transform=ccrs.PlateCarree(),
            color=color, linewidth=linewidth, linestyle=linestyle)

    # 라벨이 있으면 추가
    if label:
        ax.text(slon + (elon-slon)*0.5, elat + 0.1, label,
                transform=ccrs.PlateCarree(), color=color,
                fontsize=10, ha='center', va='bottom')


#=============================================================================
# 1. ** GK2A dataload *** ----------------------------------------------------
idr_s = f"/scratch/x3158a03/DATA/Satellite/GK2A/work/2008_sst/"

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

    print(f"\n{'='*60}")
    print(f"Processing {date_kst.strftime('%Y-%m-%d')}")
    print(f"{'='*60}")


    # 1. Read GK2A  ---
    temp_s_list, valid_mask_list, sat_coords = read_gk2a_hourly(time_list_utc, idr_s, proj)
    lat_sat, lon_sat = sat_coords

    # 위성 데이터 평균 계산
    temp_s_array = np.array(temp_s_list)
    sst_mean_s = np.nanmean(temp_s_array, axis=0)
    print("sst_mean_s= ",sst_mean_s)

    # 저장
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

# WRF
idr_w = f"/scratch/x3158a03/wrf_output/EAST-C/2008/AUTO/"
ifn_we = idr_w+f"sstx-ERA5-MetnoSST-2way_wrfout_Fog_{domain}_2020-08-17_18:00:00.nc"  #sst_fix
ifn_we2= idr_w+f"skin-ERA5-MetnoSST-2way_wrfout_Fog_{domain}_2020-08-17_18:00:00.nc"  # sst_skin

# COAWST
idr_c = f"/scratch/x3158a03/coawst_output/2008/"
ifn_ce= idr_c+f"WDM6-ERA5-SW-2way-YSU_wrfout_Fog_{domain}_2020-08-17_18:00:00.nc"  # solar_source

data0= nc.Dataset(ifn_we)
data1= nc.Dataset(ifn_we2)
data2= nc.Dataset(ifn_ce)
landmask=data1.variables['LANDMASK'][0,:,:]

lat2d = data1.variables['XLAT'][0,:,:]
lon2d = data1.variables['XLONG'][0,:,:]
#ntime = len(data1.variables['XLAT'][:,0,0])
#print("ntime: ", ntime)



# -----------------------------------------------
sst_ct_list =[]
sst_sk_list =[]
sst_cp_list =[]


ntime = 25-3 # 8/18 daily mean : 18.3KST ~ 19.3KST

for k in range(ntime):
#k=24
    sst1 = wrf.getvar(data0,f"{nvar}",timeidx=k)-273.15  #timeidx=wrf.ALL_TIMES)[k,:,:] -273.15
    sst2 = wrf.getvar(data1,f"{nvar}",timeidx=k)-273.15  #timeidx=wrf.ALL_TIMES)[k,:,:] -273.15
    sst3 = wrf.getvar(data2,f"{nvar}",timeidx=k)-273.15  #timeidx=wrf.ALL_TIMES)[k,:,:] -273.15

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
#    gl.xlocator = mticker.MultipleLocator(2)
#    gl.ylocator = mticker.MultipleLocator(2)
    gl.top_labels  = False
    gl.left_labels = False
    gl.right_labels = False


# 0. SST_Satellite ---
cf0 = axes[0].contourf(lon_sat, lat_sat, sst_mean_s,
                       levels=levels, cmap=comap, extend='both', transform=ccrs.PlateCarree())

axes[0].set_title(f'GK2A', fontsize=fns, fontweight='bold')
axes[0].text(0.02, 0.98, f'{(alpbet[0])}', transform=axes[0].transAxes,
              fontsize=fns+1, verticalalignment='top', fontweight='bold')


# 울릉도 영역 박스 추가***
colors = ['#1B4965', '#3182BD', '#E6550D', '#8C564B'] 

draw_box(axes[0], ull_coast_slat, ull_coast_elat, ull_coast_slon, ull_coast_elon,
         color=colors[1]) #label='Ulleungdo')

# 중부 연안 영역 박스 추가
draw_box(axes[0], mid_coast_slat, mid_coast_elat, mid_coast_slon, mid_coast_elon,
         color=colors[2]) #label='Mid Coast')

# 남부 연안 영역 박스 추가
draw_box(axes[0], south_coast_slat, south_coast_elat, south_coast_slon, south_coast_elon,
         color=colors[3]) # label='South Coast')

# Add Buoy sites  ***
for lat_pt, lon_pt in zip(xlat, xlon):
        axes[0].plot(
            lon_pt, lat_pt,
            marker='x', markersize=5, markeredgewidth=1.5,
            color='dimgrey', transform=ccrs.PlateCarree()
        )



# 1. SST_CNTL ---
cf1 = axes[1].contourf(wrf.to_np(lon2d), wrf.to_np(lat2d), wrf.to_np(sst_ct_mean), levels=levels,
                 cmap=comap, #'Spectral_r',  #'coolwarm'
                 extend='both')  #'both'
axes[1].set_title(f'{EXP_name1}', fontsize=fns, fontweight='bold')
axes[1].text(0.01, 0.97, f'{(alpbet[1])}', transform=axes[1].transAxes,
         fontsize=fns, fontweight='bold', va='top', ha='left', color='black')


# 2. SST_SKIN ---
cf2 = axes[2].contourf(wrf.to_np(lon2d), wrf.to_np(lat2d), wrf.to_np(sst_sk_mean), levels=levels,
                 cmap=comap,  #'coolwarm'
                 extend='both')  #'both'
axes[2].set_title(f'{EXP_name2}', fontsize=fns, fontweight='bold')
axes[2].text(0.01, 0.97, f'{(alpbet[2])}', transform=axes[2].transAxes,
         fontsize=fns, fontweight='bold', va='top', ha='left', color='black')


# 3. SST_CPLD ---
cf3 = axes[3].contourf(wrf.to_np(lon2d), wrf.to_np(lat2d), wrf.to_np(sst_cp_mean), levels=levels,
                 cmap=comap,  #'bwr',  #'coolwarm'
                 extend='both')  #'both'
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

ofn = opath+f"SST_{MODEL}-Mean_d{domain}.png"
plt.savefig(ofn, bbox_inches='tight', pad_inches=0.2, dpi=300)
plt.show()
plt.close()


