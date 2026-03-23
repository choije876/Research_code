#
#	Fig7.SST_Boxplot.py
#
##################################################

import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc
import wrf

from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import os


# Define Infomation ========================
domain = "02"
tt   = 1718
nvar = "TSK" # "T2" # "TSK"

# DATE -----
year  = "2020"
month = "08"
days  = ["17","18", "19", "20"]

# DOMAIN-AREA
if domain == "01":
    lat = 48   ; slat = 27
    elon = 140.0; slon = 118

elif domain== "02":
   lat_max =45.0 ; elat = 38 #42
   lat_min =33.0 ; slat = 35 #34
   lon_max =138.0; elon = 133 #136
   lon_min =125.0; slon = 129   #128


pos_l = ["Donghae", "Ulleungdo[ASOS]", "Ulleungdo", "Ulleungdo_NE","Ulleungdo_NW","Dokdo", "Russia","East-South","Ulsan","Uljin","Imrang"]
pos_k = ["동해","울릉도(ASOS)", "울릉도","울릉도_북동","울릉도_북서","독도","러시아","동해남쪽","울산","울진","임랑"]
xlat  = [37.490 ,  37.481,  37.455,  38.007,  37.743,  37.24 , 40  , 32,  35.2, 36.912, 35.303]
xlon  = [129.942, 130.899, 131.114, 131.553, 130.601, 131.87 , 131 ,129, 129.5, 129.87, 129.293]
lon_idx=[168, 195, 202, 213, 186, 224 , 227, 170, 160, 167 , 155]
lat_idx=[135, 137, 136, 157, 145, 130 , 316, 88 ,  51, 114 , 55 ]

position = [2, 9, 10]


# setting lat,lon ----
#position = 2
#alpha = 0
#lat_range = slice(lat_idx[position]-alpha, lat_idx[position]+alpha+1)  # e.g., 134 to 138
#lon_range = slice(lon_idx[position]-alpha, lon_idx[position]+alpha+1)  # e.g., 200 to 204

# ulleungdo 영역
ull_coast_slat = 37.0
ull_coast_elat = 38.5
ull_coast_slon = 130.5
ull_coast_elon = 131.5

# 중부 연안 영역
mid_coast_slat = 36.8 #37.0
mid_coast_elat = 38.0 # 39
mid_coast_slon = 128.8 #128.5
mid_coast_elon = 130.0 #131

# 남부 연안 영역
south_coast_slat = 35.5
south_coast_elat = 36.0
south_coast_slon = 129.2
south_coast_elon = 129.7



# save ----
opath = f"./Fig/SST/"
os.makedirs(opath, exist_ok=True)
ofn = opath+"sst-boxplot"


#=============================================================================
#-----------------------------------------------------------------------------
# 1. ** MODEL output load *** ------------------------------------------------
EXP_name1="SKIN-CNTL"
EXP_name2="CPLD-CNTL"
EXP_name3="CPLD-SKIN"


# WRF
idr_w  = f"/scratch/x3158a03/wrf_output/EAST-C/2008/AUTO/"
ifn_ct = idr_w+f"sstx-ERA5-MetnoSST-2way_wrfout_Fog_{domain}_2020-08-17_18:00:00.nc"  #sst_fix
ifn_sk = idr_w+f"skin-ERA5-MetnoSST-2way_wrfout_Fog_{domain}_2020-08-17_18:00:00.nc"  # sst_skin

# COAWST
idr_c  = f"/scratch/x3158a03/coawst_output/2008/"
ifn_cp = idr_c+f"WDM6-ERA5-SW-2way-YSU_wrfout_Fog_{domain}_2020-08-17_18:00:00.nc"  # solar_source

data0 = nc.Dataset(ifn_ct)
data1 = nc.Dataset(ifn_sk)
data2 = nc.Dataset(ifn_cp)
landmask=data1.variables['LANDMASK'][0,:,:]

lat2d = data1.variables['XLAT'][0,:,:]
lon2d = data1.variables['XLONG'][0,:,:]
#ntime = len(data1.variables['XLAT'][:,0,0])
#print("ntime: ", ntime)
ntime = 49

# 영역 내 데이터 추출을 위한 마스크 생성
# 1. 전체 영역
region_all_mask = (lat2d >= slat) & (lat2d <= elat) & (lon2d >= slon) & (lon2d <= elon) & (landmask == 0)

# 2. offshore area
off_coast_mask = (lat2d >= ull_coast_slat) & (lat2d <= ull_coast_elat) & \
                (lon2d >= ull_coast_slon) & (lon2d <= ull_coast_elon) & \
                (landmask == 0)

# 3. 중부 연안 영역
mid_coast_mask = (lat2d >= mid_coast_slat) & (lat2d <= mid_coast_elat) & \
                (lon2d >= mid_coast_slon) & (lon2d <= mid_coast_elon) & \
                (landmask == 0)

# 4. 남부 연안 영역
south_coast_mask = (lat2d >= south_coast_slat) & (lat2d <= south_coast_elat) & \
                  (lon2d >= south_coast_slon) & (lon2d <= south_coast_elon) & \
                  (landmask == 0)


all_data = []
sst_data = {'all' : {'CNTL': [], 'SKIN': [], 'CPLD': []},
            'offshore' : {'CNTL': [], 'SKIN': [], 'CPLD': []},
            'mid_coast' : {'CNTL': [], 'SKIN': [], 'CPLD': []},
            'south_coast' : {'CNTL': [], 'SKIN': [], 'CPLD': []} }



# -----------------------------------------------
for k in range(ntime):
#k=24
    sst_ct = wrf.getvar(data0,f"{nvar}",timeidx=k)-273.15  #timeidx=wrf.ALL_TIMES)[k,:,:] -273.15
    sst_sk = wrf.getvar(data1,f"{nvar}",timeidx=k)-273.15  #timeidx=wrf.ALL_TIMES)[k,:,:] -273.15
    sst_cp = wrf.getvar(data2,f"{nvar}",timeidx=k)-273.15  #timeidx=wrf.ALL_TIMES)[k,:,:] -273.15


    # 1. 전체 영역
    sst_ct_all = np.where(region_all_mask, sst_ct, np.nan)
    sst_sk_all = np.where(region_all_mask, sst_sk, np.nan)
    sst_cp_all = np.where(region_all_mask, sst_cp, np.nan)
    
    # 전체 영역 데이터 추가
    for val in sst_ct_all[~np.isnan(sst_ct_all)]:
        all_data.append({'Model': 'CNTL', 'Region': 'Entire Region', 'SST': val})
    for val in sst_sk_all[~np.isnan(sst_sk_all)]:
        all_data.append({'Model': 'SKIN', 'Region': 'Entire Region', 'SST': val})
    for val in sst_cp_all[~np.isnan(sst_cp_all)]:
        all_data.append({'Model': 'CPLD', 'Region': 'Entire Region', 'SST': val})
    
    # 2. offshore area
    sst_ct_off = np.where(off_coast_mask, sst_ct, np.nan)
    sst_sk_off = np.where(off_coast_mask, sst_sk, np.nan)
    sst_cp_off = np.where(off_coast_mask, sst_cp, np.nan)

    # 중부 연안 데이터 추가
    for val in sst_ct_off[~np.isnan(sst_ct_off)]:
        all_data.append({'Model': 'CNTL', 'Region': 'Offshore', 'SST': val})
    for val in sst_sk_off[~np.isnan(sst_sk_off)]:
        all_data.append({'Model': 'SKIN', 'Region': 'Offshore', 'SST': val})
    for val in sst_cp_off[~np.isnan(sst_cp_off)]:
        all_data.append({'Model': 'CPLD', 'Region': 'Offshore', 'SST': val})


    # 3. 중부 연안 영역
    sst_ct_mid = np.where(mid_coast_mask, sst_ct, np.nan)
    sst_sk_mid = np.where(mid_coast_mask, sst_sk, np.nan)
    sst_cp_mid = np.where(mid_coast_mask, sst_cp, np.nan)
    
    # 중부 연안 데이터 추가
    for val in sst_ct_mid[~np.isnan(sst_ct_mid)]:
        all_data.append({'Model': 'CNTL', 'Region': 'Mid Coast', 'SST': val})
    for val in sst_sk_mid[~np.isnan(sst_sk_mid)]:
        all_data.append({'Model': 'SKIN', 'Region': 'Mid Coast', 'SST': val})
    for val in sst_cp_mid[~np.isnan(sst_cp_mid)]:
        all_data.append({'Model': 'CPLD', 'Region': 'Mid Coast', 'SST': val})
    
    # 3. 남부 연안 영역
    sst_ct_south = np.where(south_coast_mask, sst_ct, np.nan)
    sst_sk_south = np.where(south_coast_mask, sst_sk, np.nan)
    sst_cp_south = np.where(south_coast_mask, sst_cp, np.nan)
    
    # 남부 연안 데이터 추가
    for val in sst_ct_south[~np.isnan(sst_ct_south)]:
        all_data.append({'Model': 'CNTL', 'Region': 'South Coast', 'SST': val})
    for val in sst_sk_south[~np.isnan(sst_sk_south)]:
        all_data.append({'Model': 'SKIN', 'Region': 'South Coast', 'SST': val})
    for val in sst_cp_south[~np.isnan(sst_cp_south)]:
        all_data.append({'Model': 'CPLD', 'Region': 'South Coast', 'SST': val})


# DataFrame 생성
df_combined = pd.DataFrame(all_data)
# 통계 출력
stats = df_combined.groupby(['Model', 'Region'])['SST'].describe()
print(stats)

# Seaborn으로 박스플롯 생성
plt.figure(figsize=(11, 6))
sns.set_style("whitegrid")


# font ---------
plt.rcParams['font.family']='serif'
plt.rcParams['axes.unicode_minus']=False



# boxplot 간격 설정 - 이 부분이 중요합니다
# 그룹 간 간격을 좁히기 위한 설정 추가
plt.rcParams['boxplot.boxprops.linewidth'] = 1.0
plt.rcParams['boxplot.whiskerprops.linewidth'] = 1.0
plt.rcParams['boxplot.medianprops.linewidth'] = 1.0
plt.rcParams['boxplot.flierprops.linewidth'] = 1.0

# 모델 사이 간격 줄이기
plt.rcParams['boxplot.patchartist'] = True
plt.rcParams['boxplot.bootstrap'] = None
# 그룹 간 간격을 0으로 설정 (기본값은 0.1)
plt.rcParams['boxplot.whiskers'] = 1.5


# 박스플롯 생성 시 직접 간격 설정
# 데이터를 numpy array로 준비
box_data = []
positions = []
pos = 0
gap_between_models = 0.85  # 모델 간 간격 (더 작은 값 = 더 좁은 간격)
region_gap = 0.15


# 각 모델별로 박스플롯 직접 그리기
for i, model in enumerate(['CNTL', 'SKIN', 'CPLD']):
    model_pos = i * gap_between_models  # 모델 위치 계산
    
    # 각 지역별 데이터 추출
    for j, region in enumerate(['Entire Region', 'Offshore', 'Mid Coast', 'South Coast']):
        data = df_combined[(df_combined['Model'] == model) & 
                          (df_combined['Region'] == region)]['SST'].values
        box_data.append(data)
        # 박스 위치 (좌우로 약간 퍼지게)
        positions.append(model_pos + (j-1.5)*region_gap)

# 직접 박스플롯 그리기
ax = plt.boxplot(box_data, positions=positions, patch_artist=True, medianprops={'color': 'black', 'linewidth': 1.5},
                showfliers=False, widths=0.12)

# 패치 색상 설정
#colors = ['skyblue', 'darkorange', 'lightgreen', 'salmon'] * 3
#colors = ['dimgrey', '#a90308', '#8d5eb7', '#ff9408'] * 3
#colors = ['dimgrey', '#703be7', '#2c6fbb', '#155084'] * 3
#colors = ['#1B4965', '#5FA8D3', '#CAE9FF', '#EE964B'] * 3
colors = ['#1B4965', '#3182BD', '#E6550D', '#8C564B'] * 3
for patch, color in zip(ax['boxes'], colors):
    patch.set_facecolor(color)

# x축 레이블 위치 설정
model_positions = [gap_between_models*i for i in range(3)]
plt.xticks(model_positions, ['CNTL', 'SKIN', 'CPLD'], fontsize=13, fontweight='bold')

# 그래프 스타일링
plt.ylabel('SST [°C]', fontsize=16, fontweight='bold')
plt.xlabel('')  # x축 제목 제거
plt.yticks(fontsize=13)
plt.ylim(18.5, 31)

# 커스텀 범례 생성
#custom_lines = [plt.Line2D([0], [0], color=color, lw=4) for color in ['skyblue', 'darkorange', 'lightgreen', 'salmon']]
#custom_lines = [plt.Line2D([0], [0], color=color, lw=4) for color in ['dimgrey', '#a90308', '#8d5eb7', '#ff9408']]
#custom_lines = [plt.Line2D([0], [0], color=color, lw=4) for color in ['dimgrey', '#703be7', '#2c6fbb', '#155084']]
#plt.legend(custom_lines, ['Entire Region', 'Offshore', 'Central Coast', 'Southern Coast'], 
#          loc='upper left', fontsize=12)
custom_patches = [Patch(facecolor=color, edgecolor='black', linewidth=0.5) 
                  for color in colors]
plt.legend(custom_patches, ['Entire Region', 'Offshore', 'Central Coast', 'Southern Coast'], 
          loc='upper left', fontsize=12,
          frameon=True, facecolor="white", edgecolor='none',framealpha=0.8,
          handlelength=1.9,  # 박스 가로 길이
          handleheight=0.4)  # 박스 세로 길이

## 지역 정보 텍스트 추가
#region_info = (f"Entire: {slon}E-{elon}E, {slat}N-{elat}N\n"
#               f"Offshore: {ull_coast_slon}E-{ull_coast_elon}E, {ull_coast_slat}N-{ull_coast_elat}N\n"
#               f"Mid Coast: {mid_coast_slon}E-{mid_coast_elon}E, {mid_coast_slat}N-{mid_coast_elat}N\n"
#               f"South Coast: {south_coast_slon}E-{south_coast_elon}E, {south_coast_slat}N-{south_coast_elat}N")
#plt.figtext(0.15, 0.15, region_info, fontsize=9,
#            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

# 그래프 레이아웃 조정
plt.grid(True, alpha=0.4, linestyle='--')
plt.tight_layout()

# 파일 저장
ofn = f"{opath}/SST_Regional_Comparison_Boxplot_d{domain}_{year}-{month}.png"
plt.savefig(ofn, bbox_inches='tight', dpi=600)
plt.show()



# IQR(사분위간 범위 = Q3-Q1) /  Q3 + 1.5 × IQR / Q1 - 1.5 × IQR
#               CNTL          SKIN          CPLD
#count  2.555889e+06  2.555889e+06  2.555889e+06
#mean   2.448643e+01  2.483555e+01  2.506633e+01
#std    1.406256e+00  1.715052e+00  2.217165e+00
#min    2.075296e+01  2.057532e+01  8.159210e+00
#25%    2.337503e+01  2.345209e+01  2.329160e+01
#50%    2.414478e+01  2.432263e+01  2.500241e+01
#75%    2.534540e+01  2.611670e+01  2.666794e+01
#max    2.843491e+01  3.159998e+01  4.066937e+01


