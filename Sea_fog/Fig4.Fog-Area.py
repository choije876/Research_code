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


#---------------calculate flag------------------------------------------------
# 1 : Clear, 2 : Middle or High Cloud, 3 : Unkown,
# 4 : Probably Fog, 5 : Fog, 6 : Snow, 7 : Desert or semi-Desert
# -> 해당 파일에서 변환) 1 : Clear,  2: Fog,  3: Cloud
#-----------------------------------------------------------------------------
#=============================================================================

# Define Infomation ========================
PBL   = "YSU"
domain= "02"
MODEL_1= "WRF"
MODEL_2= "COAWST"
lev = 21    # 23: 500m, 5: 50m
LEV = "400"
thrs= 0.016 # fog thersold

tt  = 1718  # 1618


# DATE -----
year  = "2020"
month = "08"
days  = ["17","18", "19", "20"]
wt_time_list=['2020-08-18 15:00','2020-08-18 19:00','2020-08-19 03:00'] #KST
wt_time_idxs=[12, 16, 24]



# map range ----
elat = 42  # 45 #44
slat = 34.8  # 33 #32.5
elon = 135.5 # 137.0
slon = 128.5 # 123


# save ----
opath = f"./Fig/FOG_Area/"
os.makedirs(opath, exist_ok=True)




#=============================================================================
#-----------------------------------------------------------------------------
# 1. ** MODEL output load *** ------------------------------------------------
# WRF
idr_w = f"/scratch/x3158a03/wrf_output/EAST-C/2008/AUTO/"
ifn_we = idr_w+f"sstx-ERA5-MetnoSST-2way_wrfout_Fog_{domain}_2020-08-17_18:00:00.nc"  #sst_fix
ifn_we2= idr_w+f"skin-ERA5-MetnoSST-2way_wrfout_Fog_{domain}_2020-08-17_18:00:00.nc"  # sst_skin

# COAWST
idr_c = f"/scratch/x3158a03/coawst_output/2008/"
ifn_ce= idr_c+f"WDM6-ERA5-SW-2way-YSU_wrfout_Fog_{domain}_2020-08-17_18:00:00.nc"  # solar_source

we_data = xr.open_dataset(ifn_we)
we2_data= xr.open_dataset(ifn_we2)
ce_data = xr.open_dataset(ifn_ce)

wrf_e = Dataset(ifn_we)
wrf_e2= Dataset(ifn_we2)
coa_e = Dataset(ifn_ce)

lsm_ct = getvar(wrf_e, 'LANDMASK', timeidx=0)
lsm_cp = getvar(coa_e, 'LANDMASK', timeidx=0)


# 저장 리스트 초기화
satellite_flags_list = []
wrf_flag_list = []
wrf2_flag_list= []
coa_flag_list = []
time_str_list = []


for wt_time in wt_time_idxs:
    # WRF&COAWST (ERA5 vs FNL)
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


    # 1) Clear : CBASEHT=-999(nan) /  LWC = 0
    wrf_e_flag[np.isnan(CBASE_we)] = 1
    wrf_e2_flag[np.isnan(CBASE_we2)] = 1
    coa_e_flag[np.isnan(CBASE_ce)] = 1

    # 2) Fog : lev 내 어느 한층이라도 thresold(0.016) 이상
    lwc_fog_mask  = np.any(qc_we>=thrs, axis=0)
    lwc_fog_mask1 = np.any(qc_we2>=thrs, axis=0)
    lwc_fog_mask2 = np.any(qc_ce>=thrs, axis=0)
    wrf_e_flag[lwc_fog_mask]  = 2  # 5 : Fog -> 2
    wrf_e2_flag[lwc_fog_mask1]= 2  # 5 : Fog -> 2
    coa_e_flag[lwc_fog_mask2] = 2  # 5 : Fog -> 2

    wrf_e_flag = np.where(lsm_ct == 0, wrf_e_flag, np.nan)
    wrf_e2_flag = np.where(lsm_ct == 0, wrf_e2_flag, np.nan)
    coa_e_flag = np.where(lsm_cp == 0, coa_e_flag, np.nan)

#    # 3) Mid/High cloud base height: 400 m 이상
#    wrf_e_flag[CBASE_we > 400] = 3
#    wrf_e2_flag[CBASE_we2 > 400] = 3
#    coa_e_flag[CBASE_ce > 400] = 3


    # WRF의 위경도 좌표 추출
    wrf_lat, wrf_lon = latlon_coords(CBASE_ce)


    #------------------- calcuate er_time for satellite -------------------------------
    base_time = ce_data['Times'].isel(Time=0).values
    selected_time = ce_data['Times'].isel(Time=wt_time).values
    print("base_time",base_time)
    print("selected_time",selected_time)

    # 바이트 배열을 문자열로 변환
    if isinstance(selected_time, bytes):
        selected_time_str = selected_time.decode('utf-8')
    else:
        selected_time_str = selected_time.tobytes().decode('utf-8')

    selected_time_dt = datetime.strptime(selected_time_str, '%Y-%m-%d_%H:%M:%S')
    #selected_time_dt = selected_time.astype('datetime64[ms]').astype(datetime)
    sat_time = selected_time_dt.strftime('%Y%m%d%H%M')
    print(sat_time)



    # 2. ** Satellite Data(UTC) load ** -----------------------------------------------
    idr_s="/scratch/x3158a03/DATA/Satellite/GK2A/work/2008/"
    ifn_s=f"{idr_s}gk2aamile2fogko020lc{sat_time}.nc"
    print(ifn_s)
    satellite_data = xr.open_dataset(ifn_s)

    # 1.Lambert Conformal Conic projection setup from satellite metadata
    proj_lcc = Proj(
        proj='lcc', lat_1=30, lat_2=60, lat_0=38, lon_0=126, x_0=0, y_0=0, datum='WGS84'
    )
    # 위성 X, Y 좌표 범위 설정 (900x900 그리드, 각 점 간 2000m)
    x_sat_range = np.linspace(-899000, 899000, satellite_data.dims['dim_x'])  # meters
    y_sat_range = np.linspace(899000, -899000, satellite_data.dims['dim_y'])  # meters

    # 2. **위성 위도/경도 좌표 계산**
    xv_sat, yv_sat = np.meshgrid(x_sat_range, y_sat_range)
    lon_sat, lat_sat = proj_lcc(xv_sat, yv_sat, inverse=True)

    # 3. **위성 변수에 위경도 좌표 추가**
    # lat, lon 좌표를 xarray의 DataArray로 추가 (기존 dim_y, dim_x 길이와 동일하게)
    satellite_data = satellite_data.assign_coords(
        lat=(("dim_y", "dim_x"), lat_sat),
        lon=(("dim_y", "dim_x"), lon_sat) )

    # `FOG` 변수에 위도와 경도를 좌표로 지정
    #satellite_data["FOG500m"] = satellite_data["FOG500m"].assign_coords(
    satellite_data["FOG"] = satellite_data["FOG"].assign_coords(
        lat=(("dim_y", "dim_x"), lat_sat),
        lon=(("dim_y", "dim_x"), lon_sat)
    )
    FOG=satellite_data["FOG"]



    # 4. 위성 FOG 데이터를 MODEL(WRF/COAWST) 격자에 맞게 보간
    fog_flat = FOG.values.flatten()
    lat_sat_flat = lat_sat.flatten()
    lon_sat_flat = lon_sat.flatten()

    # WRF 격자에 맞춰 위성 FOG 플래그 보간
    fog_interpolated_to_wrf = griddata(
        (lat_sat_flat, lon_sat_flat), fog_flat,
        (wrf_lat.values, wrf_lon.values), method='nearest'  # 보간 대신 nearest로 플래그 값 유지
    )

    # 3, 6, 7인 값을 NaN으로 설정
    fog_interpolated_to_wrf = np.where(
        np.isin(fog_interpolated_to_wrf, [2, 3, 6, 7]), np.nan, fog_interpolated_to_wrf)
    print("fog_interpolated_to_wrf : ", fog_interpolated_to_wrf)


    ## 매핑 딕셔너리 정의: {기존값: 새로운값}
    value_mapping = {
#        2: 3,
        4: 2,
        5: 2
    }

    ## numpy 배열에서 매핑 적용
    for original_value, new_value in value_mapping.items():
        fog_interpolated_to_wrf[fog_interpolated_to_wrf == original_value] = new_value


    satellite_flags = fog_interpolated_to_wrf
    satellite_flags = np.where(lsm_ct == 0, satellite_flags, np.nan)



    #=============================================================================
    # 1. **fog_interpolated_to_wrf에 WRF 격자 좌표 추가**
    FOG_on_WRF_grid = xr.DataArray(
        fog_interpolated_to_wrf,
        dims=CBASE_ce.dims,  # WRF QC_LOW의 격자 차원 사용
        coords={"XLON": (("south_north", "west_east"), wrf_lon.values),
                "XLAT": (("south_north", "west_east"), wrf_lat.values)}
    )

    # 2. **wrf_e_flag에 WRF 격자 좌표 추가**
    WRF_Flags_with_Coords = xr.DataArray(
        wrf_e_flag,
        dims=CBASE_ce.dims,  # WRF QC_LOW의 격자 차원 사용
        coords={"XLON": (("south_north", "west_east"), wrf_lon.values),
                "XLAT": (("south_north", "west_east"), wrf_lat.values)}
    )

    # 3. **wrf_e2_flag에 WRF 격자 좌표 추가**
    WRF_Flags_with_Coords = xr.DataArray(
        wrf_e2_flag,
        dims=CBASE_ce.dims,  # WRF QC_LOW의 격자 차원 사용
        coords={"XLON": (("south_north", "west_east"), wrf_lon.values),
                "XLAT": (("south_north", "west_east"), wrf_lat.values)}
    )

    # 4. **coa_e_flag에 WRF 격자 좌표 추가**
    WRF_Flags_with_Coords = xr.DataArray(
        coa_e_flag,
        dims=CBASE_ce.dims,  # WRF QC_LOW의 격자 차원 사용
        coords={"XLON": (("south_north", "west_east"), wrf_lon.values),
                "XLAT": (("south_north", "west_east"), wrf_lat.values)}
    )



    # UTC를 KST로 변환 (UTC + 9시간)
    kst_time_dt = selected_time_dt + timedelta(hours=9)
    tit_time = kst_time_dt.strftime('%Y.%m.%d %H')


    # 각 flag 결과를 리스트에 저장
    satellite_flags_list.append(satellite_flags)
    wrf_flag_list.append(wrf_e_flag)
    wrf2_flag_list.append(wrf_e2_flag)
    coa_flag_list.append(coa_e_flag)
    time_str_list.append(kst_time_dt.strftime('%Y.%m.%d %Hh'))




# result : GK2A: satellite_flags , WRF: wrf_e_flag, WRF: wrf_e2_flag, COAWST: coa_e_flag

we_data.close()
we2_data.close()
ce_data.close()




# ===========================================================================
# Plotting
# ===========================================================================
coastal_lons  = [129, 130.2, 130.2, 129, 129]  # 첫 점으로 다시 돌아와 닫힌 다각형 형성
coastal_lats  = [35.0, 35.0, 38.0, 38.0, 35.0]
offshore_lons = [130.2, elon, elon, 130.2, 130.2]  # 첫 점으로 다시 돌아와 닫힌 다각형 형성
offshore_lats = [36.0, 36.0, 41.0, 41.0, 36.0]
fns= 11

# User-defined ColorMap Setting
colors = ['white', 'royalblue'] #, 'aliceblue']
cmap   = ListedColormap(colors)
flag_values = [1, 2]
# 0.5~1.5, 1.5~2.5, 2.5~3.5, 3.5~4.5의 범위로 색상 경계 설정
boundaries = [0.5, 1.5, 2.5]#, 3.5]
ticks = [1, 2]#, 3]
ticks_label = ["Clear sky", "Fog"] #, "3 (Cloud)"]


alpbet = [f"({chr(97+i)})" for i in range(12)]
ytext=["GK2A","CNTL","SKIN","CPLD"]


norm = mcolors.BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)

left, bottom, width, height = 0.3, 0.053, 0.45, 0.02

extent = [slon, elon, slat, elat]



# 한 figure에 4행 3열로 플롯 ----------
fig, axes = plt.subplots(4, 3, figsize=(9, 11), subplot_kw={'projection': ccrs.PlateCarree()})

plt.tight_layout()
fig.subplots_adjust(wspace=-0.5, hspace=0.06, left=0.0, right=0.98, top=0.92, bottom=0.12)
#fig.subplots_adjust(wspace=0.03, hspace=0.02, left=0.08, right=0.98, top=0.92, bottom=0.12)


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
               # add box ---
               ax.plot(coastal_lons, coastal_lats, transform=ccrs.PlateCarree(), color='red', linewidth=1.3, linestyle='-')
   	       # ax.plot(offshore_lons, offshore_lats, transform=ccrs.PlateCarree(), color='red', linewidth=1.3, linestyle='-')
            
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


       # if i == 0:
       #     ax.text(0.5, 1.1, time_str_list[j], transform=ax.transAxes,
       #             fontsize=fns+1, fontweight='bold', va='center', ha='center')
        if j == 0:
            ax.text(-0.05, 0.5, ytext[i], transform=ax.transAxes,
                    fontsize=fns+1, fontweight='bold', va='center', ha='right', rotation=90)


        cf = ax.pcolormesh(wrf_lon, wrf_lat, data, cmap=cmap, norm=norm)
       

        # 축 라벨 제거
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
         

# 공통 컬러바 추가
cbar_ax = fig.add_axes([left, bottom, width, height])
cbar = fig.colorbar(cf, cax=cbar_ax, orientation='horizontal', ticks=[1, 2])
#cbar.set_label('Flag', fontsize=fns-1)
cbar.set_ticks(ticks)
cbar.set_ticklabels(ticks_label)
cbar.ax.tick_params(labelsize=fns-2)



for ax in axes.flatten():
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)
        spine.set_edgecolor('black')



#ofn_time = kst_time_dt.strftime('%Y-%m-%d_%H')
ofn = f"{opath}/Fig3_GK2A-WRF-WRF2-COAWST_fog-area-{LEV}m"
plt.savefig(ofn, bbox_inches='tight', dpi=300, pad_inches=0.2)
print(f"Save figure successfully! : {ofn})")

plt.show()
satellite_data.close()
plt.close()

