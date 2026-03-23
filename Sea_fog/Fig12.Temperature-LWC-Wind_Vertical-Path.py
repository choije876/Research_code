#
#	Fig12.Temperature-LWC-Wind_Vertical-Path.py
#	- Inversion later with Temp & qc
#
############################################################

import numpy as np
import netCDF4 as nc
import wrf
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib.cm import get_cmap
import matplotlib as mpl
import cartopy.crs as ccrs
import metpy.plots as mpplots
import os


# font setting ---------
plt.rcParams['font.family']='serif'
plt.rcParams['axes.unicode_minus']=False



# Infomation Setting --
tt = 1718
domain = "02"
year = "2020"
month = "08"
ini_time="18"
days = ["16","17","18", "19", "20"]
alpbet = [f"({chr(97+i)})" for i in range(6)]

#lev= "1000"; klev=23
#lev= "400"; klev=21
lev= "all"; klev=35


opath= f"./Fig/Temp_Vert/{lev}m/"
os.makedirs(opath, exist_ok=True)


pos_l = ["Donghae", "Ulleungdo[ASOS]", "Ulleungdo", "Ulleungdo_NE","Ulleungdo_NW","Dokdo", "Russia","East-South","land"]
xlat  = [37.490 ,  37.481,  37.455,  38.007,  37.743,  37.24 , 40  ,36.2,   37]
xlon  = [129.942, 130.899, 131.114, 131.553, 130.601, 131.87 , 131 ,129.5, 129]#128]
lon_idx=[168, 195, 202, 213, 186, 224 , 227, 170, 99 ]
lat_idx=[135, 137, 136, 157, 145, 130 , 316, 88 , 123 ]

pointA = 8 #8
pointB = 2 #2


preofn=f"Figure12_{pos_l[pointA]}-{pos_l[pointB]}_Foehn-V10"


# 경로를 따라 여러 지점 생성
num_points= 100
path_lats = np.linspace(lat_idx[pointA], lat_idx[pointB], num_points).astype(int)
path_lons = np.linspace(lon_idx[pointA], lon_idx[pointB], num_points).astype(int)


# ==============================================================================
# READ & LOAD FILE =============================================================
# (1) WRF/COAWST with ERA5 IC/BC --------------
# WRF ----
idr_w = "/scratch/x3158a03/wrf_output/EAST-C/2008/AUTO/"
ifn1 = idr_w+f"sstx-ERA5-MetnoSST-2way_wrfout_Fog_{domain}_2020-08-17_18:00:00.nc"  #sst_fix
ifn_w1= idr_w+f"skin-ERA5-MetnoSST-2way_wrfout_Fog_{domain}_2020-08-17_18:00:00.nc"  # sst_skin

#COAWST ----
idr_c =f"/scratch/x3158a03/coawst_output/2008/"
ifn2= idr_c+f"WDM6-ERA5-SW-2way-YSU_wrfout_Fog_{domain}_2020-08-17_18:00:00.nc"  # solar_source

data1  = nc.Dataset(ifn1)
data_w1= nc.Dataset(ifn_w1)
data2  = nc.Dataset(ifn2)

lat2d= data2.variables['XLAT'][0,:,:]
lon2d= data2.variables['XLONG'][0,:,:]
time = data2.variables['XLAT'][:,0,0]
ntime = len(time)


#  데이터 전처리============================
#k  = 24


# 3 행 2열 ======================
#fig, axes = plt.subplots(3, 2, figsize=(12, 11), sharex=True, gridspec_kw={'hspace': 0.06})
fig = plt.figure(figsize=(12, 14))
grid = plt.GridSpec(4, 2, height_ratios=[1, 1, 1, 0.15], hspace=0.2, wspace=0.23)
axes = [[None, None], [None, None], [None, None]]

# 각 서브플롯 생성
for i in range(3):
    for j in range(2):
        axes[i][j] = fig.add_subplot(grid[i, j])



wt_time_list = [12, 24]

for ii, k in enumerate(wt_time_list) :

    hgt  =  wrf.getvar(data2,'HGT',timeidx=k)  # Height of Terrain
    height= wrf.getvar(data2, "z", timeidx=k)  # 지오포텐셜 고도/해발 고도(m)

    qc1 = wrf.getvar(data1, "QCLOUD", timeidx=k)
    temp1 = wrf.getvar(data1, "tc", timeidx=k)
    uwind1= wrf.getvar(data1, "U",timeidx=k)
    vwind1= wrf.getvar(data1, "V",timeidx=k)
    wwind1= wrf.getvar(data1, "W",timeidx=k)

    qc_w1 = wrf.getvar(data_w1, "QCLOUD", timeidx=k)
    temp_w1 = wrf.getvar(data_w1, "tc", timeidx=k)
    uwind_w1= wrf.getvar(data_w1, "U",timeidx=k)
    vwind_w1= wrf.getvar(data_w1, "V",timeidx=k)
    wwind_w1= wrf.getvar(data_w1, "W",timeidx=k)

    qc2 = wrf.getvar(data2, "QCLOUD", timeidx=k)
    temp2 = wrf.getvar(data2, "tc", timeidx=k)
    uwind2= wrf.getvar(data2, "U",timeidx=k)
    vwind2= wrf.getvar(data2, "V",timeidx=k)
    wwind2= wrf.getvar(data2, "W",timeidx=k)


    height= height[:klev,:,:]
    qc1   = qc1[:klev,:,:] * 1e3
    qc_w1 = qc_w1[:klev,:,:] * 1e3
    qc2   = qc2[:klev,:,:] * 1e3
    temp1   = temp1[:klev,:,:]
    temp_w1 = temp_w1[:klev,:,:]
    temp2   = temp2[:klev,:,:]

    uwind1  = uwind1[:klev,:,:]
    uwind_w1= uwind_w1[:klev,:,:]
    uwind2  = uwind2[:klev,:,:]

    vwind1  = vwind1[:klev,:,:]
    vwind_w1= vwind_w1[:klev,:,:]
    vwind2  = vwind2[:klev,:,:]

    wwind1  = wwind1[:klev,:,:]
    wwind_w1= wwind_w1[:klev,:,:]
    wwind2  = wwind2[:klev,:,:]


    # 경로를 따라 데이터 추출
    qc1_cross    = np.zeros((klev, num_points))
    qc_w1_cross  = np.zeros((klev, num_points))
    qc2_cross    = np.zeros((klev, num_points))
    qc_diff_cross= np.zeros((klev, num_points))

    temp1_cross    = np.zeros((klev, num_points))
    temp_w1_cross  = np.zeros((klev, num_points))
    temp2_cross    = np.zeros((klev, num_points))
    temp_diff_cross= np.zeros((klev, num_points))

    uwind1_cross    = np.zeros((klev, num_points))
    uwind_w1_cross  = np.zeros((klev, num_points))
    uwind2_cross    = np.zeros((klev, num_points))
    uwind_diff_cross= np.zeros((klev, num_points))

    vwind1_cross    = np.zeros((klev, num_points))
    vwind_w1_cross  = np.zeros((klev, num_points))
    vwind2_cross    = np.zeros((klev, num_points))
    vwind_diff_cross= np.zeros((klev, num_points))

    wwind1_cross    = np.zeros((klev, num_points))
    wwind_w1_cross  = np.zeros((klev, num_points))
    wwind2_cross    = np.zeros((klev, num_points))
    wwind_diff_cross= np.zeros((klev, num_points))

    height_cross = np.zeros((klev, num_points))
    terr_cross   = np.zeros(num_points)


    for i in range(num_points):
        lat_i = path_lats[i]
        lon_i = path_lons[i]
        qc1_cross[:, i]     = qc1[:, lat_i, lon_i]
        qc_w1_cross[:, i]   = qc_w1[:, lat_i, lon_i]
        qc2_cross[:, i]     = qc2[:, lat_i, lon_i]
        temp1_cross[:, i]     = temp1[:, lat_i, lon_i]
        temp_w1_cross[:, i]   = temp_w1[:, lat_i, lon_i]
        temp2_cross[:, i]     = temp2[:, lat_i, lon_i]
        uwind1_cross[:, i]     = uwind1[:, lat_i, lon_i]
        uwind_w1_cross[:, i]   = uwind_w1[:, lat_i, lon_i]
        uwind2_cross[:, i]     = uwind2[:, lat_i, lon_i]
        vwind1_cross[:, i]     = vwind1[:, lat_i, lon_i]
        vwind_w1_cross[:, i]   = vwind_w1[:, lat_i, lon_i]
        vwind2_cross[:, i]     = vwind2[:, lat_i, lon_i]
        wwind1_cross[:, i]     = wwind1[:, lat_i, lon_i]
        wwind_w1_cross[:, i]   = wwind_w1[:, lat_i, lon_i]
        wwind2_cross[:, i]     = wwind2[:, lat_i, lon_i]

        height_cross[:, i] = height[:, lat_i, lon_i]
        terr_cross[i] =hgt[lat_i, lon_i]


    # 풍향 계산 -----------------------------
    path_direction = np.zeros(num_points)
    for i in range(num_points-1):
        lat1, lon1 = lat2d[path_lats[i], path_lons[i]], lon2d[path_lats[i], path_lons[i]]
        lat2, lon2 = lat2d[path_lats[i+1], path_lons[i+1]], lon2d[path_lats[i+1], path_lons[i+1]]
        # 경로 방향각 계산 (북쪽 기준 시계방향)
        dlon = np.radians(lon2 - lon1)
        dlat = np.radians(lat2 - lat1)
        path_direction[i] = np.degrees(np.arctan2(dlon * np.cos(np.radians(lat1)), dlat))
    path_direction[-1] = path_direction[-2]  # 마지막 지점은 이전 값 사용


    # 바람 성분 계산 (단면에 평행한 성분과 수직 성분)
    wind_parallel1  = np.zeros((klev, num_points))   # 경로에 평행한 바람 (WRF)
    wind_vertical1  = np.zeros((klev, num_points))   # 수직 바람 (WRF)
    wind_parallel_w1= np.zeros((klev, num_points))   # 경로에 평행한 바람 (WRF)
    wind_vertical_w1= np.zeros((klev, num_points))   # 수직 바람 (WRF)
    wind_parallel2  = np.zeros((klev, num_points))   # 경로에 평행한 바람 (COAWST)
    wind_vertical2  = np.zeros((klev, num_points))   # 수직 바람 (COAWST)

    for i in range(num_points):
        theta = np.radians(path_direction[i])
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # 수평 바람 벡터를 경로 좌표계로 회전 (경로에 평행한 성분만)
        wind_parallel1[:, i]  = uwind1_cross[:, i] # * sin_theta + vwind1_cross[:, i] * cos_theta
        wind_parallel_w1[:, i]= uwind_w1_cross[:, i] # * sin_theta + vwind1_cross[:, i] * cos_theta
        wind_parallel2[:, i]  = uwind2_cross[:, i] # * sin_theta + vwind2_cross[:, i] * cos_theta

        # 수직 바람 성분 (실제 W 성분 사용)
        wind_vertical1[:, i] = wwind1_cross[:, i]
        wind_vertical_w1[:, i] = wwind_w1_cross[:, i]
        wind_vertical2[:, i] = wwind2_cross[:, i]

    print("max wind speed: ", np.sqrt(np.max(uwind_w1_cross**2+vwind_w1_cross**2)))



    # --------------------------------------
    # 메모리 확보 위해 안쓰는 변수 삭제 ---
    del qc1, qc2, qc_w1, temp1, temp_w1,temp2, uwind1, uwind2, uwind_w1,vwind1, vwind_w1, vwind2, wwind1, wwind2, wwind_w1
    import gc
    gc.collect()



    # ======================================
    # Plotting =============================
    # ======================================
    line_w = 1.5
    font_w = 12

    temp_cmap = plt.cm.get_cmap('RdBu_r').copy() 
    temp_cmap.set_bad('lightgrey', alpha=1.0)
    
#    temp_cmap = plt.cm.get_cmap("Spectral_r").copy() ; diff_cmap="coolwarm"
#    temp_cmap.set_bad('lightgrey', alpha=1.0)

    y_min = 0  # 또는 필요시 terrain_heights의 최솟값
    y_max = 1000

    temp1_min, temp1_max =  20, 30
    temp1_levels= np.linspace(temp1_min, temp1_max, 11)

    # 거리 축 계산 (km)
    distances = np.zeros(num_points)
    for i in range(1, num_points):
        # 대략적인 거리 계산 (위경도를 km로 변환)
        lat1, lon1 = lat2d[path_lats[i-1], path_lons[i-1]], lon2d[path_lats[i-1], path_lons[i-1]]
        lat2, lon2 = lat2d[path_lats[i], path_lons[i]], lon2d[path_lats[i], path_lons[i]]
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distances[i] = distances[i-1] + 6371 * c  # 지구 반지름 6371km

    # 방법 1: 각 지점별로 개별 contour 그리기 (정확한 방법)
    X_full = np.zeros((klev, num_points))
    Y_full = np.zeros((klev, num_points))
    terrain_mask = np.zeros_like(Y_full)
    terrain_heights = terr_cross


    for i in range(num_points):
        X_full[:, i] = distances[i]
        Y_full[:, i] = height_cross[:,i]


    for i in range(num_points):
        for k in range(klev):
           if Y_full[k, i] < terrain_heights[i]:
                # 지형보다 아래에 있는 셀은 NaN으로 설정
                temp1_cross[k, i] = np.nan
                temp_w1_cross[k, i] = np.nan
                temp2_cross[k, i] = np.nan
                qc1_cross[k, i] = np.nan
                qc_w1_cross[k, i] = np.nan
                qc2_cross[k, i] = np.nan
                wind_parallel1[k, i] = np.nan
                wind_parallel_w1[k, i] = np.nan
                wind_parallel2[k, i] = np.nan
                wind_vertical1[k, i] = np.nan
                wind_vertical_w1[k, i] = np.nan
                wind_vertical2[k, i] = np.nan

    qc1_min, qc1_max = np.min(qc1_cross), np.max(qc1_cross)



    ax1 = axes[0][ii]  # CNTL
    ax2 = axes[1][ii]  # SKIN
    ax3 = axes[2][ii]  # CPLD


    # Temperature shading ------------------------
    cf1 = ax1.contourf(X_full, Y_full, temp1_cross,
                   levels=temp1_levels, cmap=temp_cmap, extend='both')
    cf2 = ax2.contourf(X_full, Y_full, temp_w1_cross,
                    levels=temp1_levels, cmap=temp_cmap, extend='both')
    cf3 = ax3.contourf(X_full, Y_full, temp2_cross,
                    levels=temp1_levels, cmap=temp_cmap, extend='both')

    ct1 = ax1.contour(X_full, Y_full, temp1_cross,
                    levels=temp1_levels[::2], colors='grey', linewidths=0.5)
    ct2 = ax2.contour(X_full, Y_full, temp_w1_cross,
                    levels=temp1_levels[::2], colors='grey', linewidths=0.5)
    ct3 = ax3.contour(X_full, Y_full, temp2_cross,
                    levels=temp1_levels[::2], colors='grey', linewidths=0.5)
    ax1.clabel(ct1, inline=True, fontsize=font_w-2, fmt='%.1f')
    ax2.clabel(ct2, inline=True, fontsize=font_w-2, fmt='%.1f')
    ax3.clabel(ct3, inline=True, fontsize=font_w-2, fmt='%.1f')

    ax1.text(0.01, 0.97, f'{(alpbet[0+ii*3])}', transform=ax1.transAxes, fontsize=font_w, fontweight='bold', va='top', ha='left', color='black')
    ax2.text(0.01, 0.97, f'{(alpbet[1+ii*3])}', transform=ax2.transAxes, fontsize=font_w, fontweight='bold', va='top', ha='left', color='black')
    ax3.text(0.01, 0.97, f'{(alpbet[2+ii*3])}', transform=ax3.transAxes, fontsize=font_w, fontweight='bold', va='top', ha='left', color='black')

    ax1.set_ylim([y_min, y_max])
    ax2.set_ylim([y_min, y_max])
    ax3.set_ylim([y_min, y_max])


    ax1.text(0.01,1.06, 'CNTL', transform=ax1.transAxes, fontsize=font_w, fontweight='bold', va='top', ha='left', color='black')
    ax2.text(0.01,1.06, 'SKIN', transform=ax2.transAxes, fontsize=font_w, fontweight='bold', va='top', ha='left', color='black')
    ax3.text(0.01,1.06, 'CPLD', transform=ax3.transAxes, fontsize=font_w, fontweight='bold', va='top', ha='left', color='black')


    # Qc line ------------------------------------
    qc_contour_levels = [0.018, 0.1, 0.4, 0.8, 1.0]  # Adjust these levels as needed
    qc_diff_contour_levels = [-1.2, -0.8, -0.4, -0.05, 0.05, 0.4, 0.8, 1.2]

    ct11 = ax1.contour(X_full, Y_full, qc1_cross,
                    levels=qc_contour_levels, colors='black', linewidths=2)
    ct22 = ax2.contour(X_full, Y_full, qc_w1_cross,
                    levels=qc_contour_levels, colors='black', linewidths=2)
    ct33 = ax3.contour(X_full, Y_full, qc2_cross,
                    levels=qc_contour_levels, colors='black', linewidths=2)
    ax1.clabel(ct11, inline=True, fontsize=font_w, fmt='%.2f')
    ax2.clabel(ct22, inline=True, fontsize=font_w, fmt='%.2f')
    ax3.clabel(ct33, inline=True, fontsize=font_w, fmt='%.2f')


    # wind vector -------------------------------- 
    wind_skip_levels = 3  # 수직 레벨 간격
    wind_skip_points = 5  # 수평 지점 간격

    # 수직 바람 스케일
    w_scale_factor = 10
    wind_scale = 100

    wind_x = X_full[::wind_skip_levels, ::wind_skip_points]
    wind_y = Y_full[::wind_skip_levels, ::wind_skip_points]

    wind_u1  = wind_parallel1[::wind_skip_levels, ::wind_skip_points]
    wind_u_w1= wind_parallel_w1[::wind_skip_levels, ::wind_skip_points]
    wind_u2  = wind_parallel2[::wind_skip_levels, ::wind_skip_points]
    wind_v1  = wind_vertical1[::wind_skip_levels, ::wind_skip_points]  * w_scale_factor
    wind_v_w1= wind_vertical_w1[::wind_skip_levels, ::wind_skip_points]* w_scale_factor
    wind_v2  = wind_vertical2[::wind_skip_levels, ::wind_skip_points]  * w_scale_factor


    wv1 = ax1.quiver(wind_x, wind_y, wind_u1, wind_v1,
                scale=wind_scale, color='gold', alpha=0.9, width=0.005
                )
    wv_w1 = ax2.quiver(wind_x, wind_y, wind_u_w1, wind_v_w1,
                scale=wind_scale, color='gold', alpha=0.9, width=0.005
                )
    wv2 = ax3.quiver(wind_x, wind_y,  wind_u2, wind_v2,
                scale=wind_scale, color='gold', alpha=0.9, width=0.005
                )

    # 참조 화살표 (스케일 표시)
    ref_wind_speed_h = 15 # 15 m/s 수평 바람
    x_max = np.max(distances) 
    ref_x = x_max * 0.1  # 오른쪽 80% 위치
    ref_y = y_max * 0.1


    if ii ==1 :
        ax3.quiver(ref_x, ref_y, ref_wind_speed_h, 0,
                   scale=wind_scale/100, scale_units='xy', angles='xy',
                   color='dimgrey', width=0.004, headwidth=2)
        ax3.text(ref_x, ref_y-y_max*0.06, f'{ref_wind_speed_h} m/s',
                 fontsize=font_w-5, color='dimgrey', fontweight='bold',
                 verticalalignment='center')
   
 
    # 지형은 y=0에 표시
    ax1.fill_between(distances, 0, 0, color='saddlebrown', alpha=0.7)  # 바닥선만
    ax2.fill_between(distances, 0, 0, color='saddlebrown', alpha=0.7)
    ax3.fill_between(distances, 0, 0, color='saddlebrown', alpha=0.7)
    

#    ax1.set_title(f'CNTL', fontsize=font_w)
#    ax2.set_title(f'SKIN', fontsize=font_w)
#    ax3.set_title(f'CPLD', fontsize=font_w)

    if ii==0:
        ax1.set_ylabel('Height (m)', fontsize=font_w)
        ax2.set_ylabel('Height (m)', fontsize=font_w)
        ax3.set_ylabel('Height (m)', fontsize=font_w)

    
    # x축 위경도 표시 설정 ------------
    axes[0][ii].set_xticklabels([])  # 첫 번째 행 (ax1)
    axes[1][ii].set_xticklabels([])  # 두 번째 행 (ax2)

#    # 인덱스에 해당하는 실제 위경도 좌표 가져오기
#    path_lats_actual = np.array([lat2d[path_lats[i], path_lons[i]] for i in range(num_points)])
#    path_lons_actual = np.array([lon2d[path_lats[i], path_lons[i]] for i in range(num_points)])
#
#    num_xticks = 6
#    tick_indices = np.linspace(0, num_points-1, num_xticks, dtype=int)
#    tick_distances= distances[tick_indices]
#
#    xlabels = []
#    for i in tick_indices:
#        lat = path_lats_actual[i]
#        lon = path_lons_actual[i]
#        dist= distances[i]
#        xlabels.append(f"{lat:.1f}°N\n{lon:.1f}°E")

    tick_distances =  [distances[0], distances[-1]]
    xlabels = ["A", "B" ]
    ax3.set_xticks(tick_distances)
    ax3.set_xticklabels(xlabels, rotation=0, ha='right', fontsize=font_w-1)
   # ax3.set_xlabel('Lat, Lon', fontsize=font_w)
    
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis='x', length=0)

    if k >= 0 and k <= 20:  # 18일
        day_index = 2
    elif k >=21 and k<=44 :  #19일
        day_index = 3
    else:  # 20일
        day_index = 4


    kst  = k + 3 if k <= 20 else k - 21 if k <= 44 else k - 45
    ksti = f"{kst:02d}"
    dayi = days[day_index]

#    plt.suptitle(f'[{year}-{month}-{dayi}_{ksti} LST]', fontsize=font_w+3, y=0.95, fontweight='bold')



#cbar_ax = fig.add_subplot(grid[3, :])  # 하단 전체 영역 사용
cbar_ax = fig.add_axes([0.15, 0.12, 0.7, 0.012])
cbar = plt.colorbar(cf1, cax=cbar_ax, orientation='horizontal', extend="both")
cbar.set_label('Temperature [$^{o}$C]', fontsize=font_w)
#cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
cbar.ax.tick_params(labelsize=font_w)


plt.tight_layout(rect=[0, 0.15, 1, 0.96])

ofn  = f"{opath}/{preofn}_d{domain}_{year}-{month}-{dayi}_{ksti}KST"
plt.savefig(ofn, pad_inches=0.1, dpi=300, bbox_inches='tight')
plt.show()
plt.close()


data1.close()
data_w1.close()
data2.close()


