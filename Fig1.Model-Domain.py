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



# Setting ========================================
# 데이터 경로 ----
idr = "/scratch/x3158a03/DATA/Topo/"
ifn = idr + "ETOPO1_Bed_g_gmt4.grd"


# 실제 WRF 도메인 좌표-----
ACTUAL_DOMAINS = {
    'domain1': {  # D01 - 회색 박스
        'lon_min': 118.0, 'lon_max': 140.0,
        'lat_min': 27.0, 'lat_max': 48.0,
        'color': 'black',
        'linewidth': 3,
        'label': 'D01'
    },
    'domain2': {  # D02 - 빨간색 박스
        'lon_min': 124.0, 'lon_max': 138.0,
        'lat_min': 33.0, 'lat_max': 45.0,
        'color': 'red',
        'linewidth': 2,
        'label': 'D02'
    }
}


# Points (buoy) ----
#pos_l = ["ULL", "DON", "UNE", "UNW", "ULS", "ULJ", "POH","IMR"]
#lat_l = [37.456 , 37.481, 38.007, 37.743, 35.345, 36.907, 36.350, 35.303]
#lon_l = [131.114,129.950,131.553,130.601,129.841,129.874,129.783,129.293]
pos_l = ["ULL", "ULJ", "IMR", "A"]
lat_l = [37.456,  36.907, 35.303,  37.3]
lon_l = [131.114,129.874,129.293, 127.6]



# Funtion ========================================

def load_etopo_data():
    """ETOPO1 데이터 로드"""
    try:
        bathy = xr.open_dataset(ifn)
        print("ETOPO1 데이터 로드 성공!")

        if 'z' in bathy.variables:
            depth = bathy['z']
        elif 'elevation' in bathy.variables:
            depth = bathy['elevation']
        else:
            depth_var = list(bathy.data_vars)[0]
            depth = bathy[depth_var]

        print(f"전체 경도 범위: {float(depth.x.min()):.1f} ~ {float(depth.x.max()):.1f}")
        print(f"전체 위도 범위: {float(depth.y.min()):.1f} ~ {float(depth.y.max()):.1f}")

        return depth

    except Exception as e:
        print(f"데이터 로드 오류: {e}")
        return None

def create_actual_wrf_map():
    """실제 WRF 도메인 수심 맵 생성"""

    # 데이터 로드
    depth = load_etopo_data()
    if depth is None:
        return None, None, None

    margin = 2.0
    lon_min = ACTUAL_DOMAINS['domain1']['lon_min'] - margin
    lon_max = ACTUAL_DOMAINS['domain1']['lon_max'] + margin
    lat_min = ACTUAL_DOMAINS['domain1']['lat_min'] - margin
    lat_max = ACTUAL_DOMAINS['domain1']['lat_max'] + margin

    print(f"\n지도 영역:")
    print(f"경도: {lon_min:.1f} ~ {lon_max:.1f}")
    print(f"위도: {lat_min:.1f} ~ {lat_max:.1f}")

    # 영역 추출
    map_bathy = depth.sel(
        x=slice(lon_min, lon_max),
        y=slice(lat_min, lat_max)
    )

    # Plotting ===================================
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
    cbar_ocean.set_label('Depth [m]', fontsize=12, rotation=270, labelpad=20) #fontweight='bold'
    cbar_ocean.ax.tick_params(labelsize=10)
    cbar_ocean.ax.yaxis.set_ticks_position('none')
 
    cbar_land = plt.colorbar(cs_land, ax=ax, shrink=0.6, pad=0.03, aspect=30, location='right')
    cbar_land.set_label('Elevation [m]', fontsize=12, rotation=270, labelpad=20)
    cbar_land.ax.tick_params(labelsize=10)
    cbar_land.ax.yaxis.set_ticks_position('none')

#    # 틱 마크 완전 제거
#    for line in cbar.ax.yaxis.get_ticklines():
#        line.set_markersize(0)
#        line.set_markeredgewidth(0)


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

        # 도메인 라벨 (오른쪽 위)
        label_lon = domain_info['lon_max'] - 1.0
        label_lat = domain_info['lat_max'] - 1.0

        ax.text(label_lon, label_lat,
               domain_info['label'],
               ha='right', va='top',
               color=domain_info['color'],
               fontweight='bold', fontsize=15,
     #          path_effects=[pe.withStroke(linewidth=3, foreground='white')],
               transform=ccrs.PlateCarree())
     #          bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8))

    ax.text(131.5, 38.9, 'East Sea', ha='center', va='bottom',
       fontsize=15, fontweight='bold', color='black',
       path_effects=[pe.withStroke(linewidth=3, foreground='white')])

    # 부제목 (작은 폰트)
    ax.text(131.5, 38.9, '(Sea of Japan)', ha='center', va='top',  # 38.3
           fontsize=12, color='black',
           path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])


#    ax.text(135, 38.7, 'East Sea\n  (Sea of Japan)', ha='center', va='center',
#           fontsize=15, fontweight='bold', color='black',
#           path_effects=[pe.withStroke(linewidth=3, foreground='white')])
##           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.text(123.5, 35.5, 'Yellow\nSea', ha='center', va='center',
           fontsize=15, fontweight='bold', color='black',
           path_effects=[pe.withStroke(linewidth=3, foreground='white')])
      #    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))


    # 3) Adding Buoy Points ******
    for i, (name, lat, lon) in enumerate(zip(pos_l, lat_l, lon_l)): 
        if i < (len(pos_l)-1) :
            ax.plot(lon, lat, 'x', markersize=6, markerfacecolor="black", markeredgecolor='black', markeredgewidth=2,
                    transform=ccrs.PlateCarree())
        
            ax.text(lon-0.3, lat-0.9, name, ha='left', va='bottom', fontsize=9, fontweight='bold', color='black',
                    path_effects=[pe.withStroke(linewidth=2.5, foreground='white')],
                    transform=ccrs.PlateCarree())
        else:
            #ax.text(lon, lat+0.21, name, ha='left', va='bottom', fontsize=9, fontweight='bold', color='red',
            ax.text(lon-0.3,lat-0.8, name, ha='left', va='bottom', fontsize=9, fontweight='bold', color='black',
                    path_effects=[pe.withStroke(linewidth=2.5, foreground='white')],
                    transform=ccrs.PlateCarree())
            ax.text(lon_l[0]+1.0, lat_l[0]-0.9, '(B)', ha='left', va='bottom', fontsize=9, fontweight='bold', color='black',
                    path_effects=[pe.withStroke(linewidth=2.5, foreground='white')],
                    transform=ccrs.PlateCarree())

    # 4) Adding Path (A->ULL) ******
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


    # 지도 범위 설정
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    plt.tight_layout()

    # 그림 저장
    plt.savefig('actual_wrf_domains_bathymetry.png', dpi=400, bbox_inches='tight')
    print("지도가 'actual_wrf_domains_bathymetry.png'로 저장되었습니다.")

    plt.show()

    return fig, ax, map_bathy


def update_domain_coordinates(domain_name, lon_min, lon_max, lat_min, lat_max):
    """도메인 좌표 업데이트 함수"""
    ACTUAL_DOMAINS[domain_name].update({
        'lon_min': lon_min, 'lon_max': lon_max,
        'lat_min': lat_min, 'lat_max': lat_max
    })
    print(f"{domain_name} 좌표가 업데이트되었습니다:")
    print(f"  경도: {lon_min} ~ {lon_max}")
    print(f"  위도: {lat_min} ~ {lat_max}")

# 메인 실행
if __name__ == "__main__":
    print("실제 WRF 도메인 수심 맵을 생성합니다...")
    print("=" * 50)

    print("현재 도메인 설정 (이미지 기반으로 추정):")
    for domain, config in ACTUAL_DOMAINS.items():
        print(f"{config['label']}: 경도 {config['lon_min']}~{config['lon_max']}, "
              f"위도 {config['lat_min']}~{config['lat_max']}")

    print("\n*** 정확한 좌표로 수정이 필요하면 아래 함수를 사용하세요 ***")
    print("update_domain_coordinates('domain1', 110.0, 140.0, 27.0, 48.0)")
    print("update_domain_coordinates('domain2', 117.0, 135.0, 33.0, 45.0)")
    print("=" * 50)

    # 수심 맵 생성
    fig, ax, bathy_data = create_actual_wrf_map()

    print("완료!")


