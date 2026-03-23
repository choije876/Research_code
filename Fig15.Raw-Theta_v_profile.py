#
#	Fig15.Raw-Theta_v_profile.py
#      - theta_v + inversion layer height with max(dthv/dz)
#
############################################################

import numpy as np
import netCDF4 as nc
import wrf
from wrf import (getvar, get_cartopy,ALL_TIMES, interplevel,latlon_coords,to_np)
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os


# font ---------
plt.rcParams['font.family']='serif'
plt.rcParams['axes.unicode_minus']=False


#=======================================
# Define file paths and variables
domain = "02"
year   = "2020"
month  = "08"
days   = ["17", "18", "19"]
time_kst_list = ["09", "15","21","03"]
time_idx_list = [6, 12, 18, 24]


p0      = 1000    # hPa
kappa   = 0.2854  # Rd/cp
z_space = 10

pos_l = ["Donghae", "Ulleungdo[ASOS]", "Ulleungdo", "Ulleungdo_NE","Ulleungdo_NW","Dokdo", "Russia","East-South","Ulsan","Uljin","Imrang"]
pos_k = ["동해","울릉도(ASOS)", "울릉도","울릉도_북동","울릉도_북서","독도","러시아","동해남쪽","울산","울진","임랑"]
pos_cl= ["DON", "ULL_ASOS","ULL", "UNE","UNW", "DOK", "RUS","ES","ULS","ULJ","IMR"]
xlat  = [37.490 ,  37.481,  37.455,  38.007,  37.743,  37.24 , 40  , 32,  35.2, 36.912, 35.303]
xlon  = [129.942, 130.899, 131.114, 131.553, 130.601, 131.87 , 131 ,129, 129.5, 129.87, 129.293]
lon_idx=[168, 195, 202, 213, 186, 224 , 227, 170, 160, 167 , 155]
lat_idx=[135, 137, 136, 157, 145, 130 , 316, 88 ,  51, 114 , 55 ]

heights  = np.arange(0, 411, 5)
nlevel   = 23 
position_list  = [2, 9, 10]
position_names = ["ULL","ULJ","IMR"]


# save ------
opath = f"./Fig/Temp_Vert/"
ofn   = opath+"Vertical-Theta-v_SSmax-level_profile_3x4_6h"
os.makedirs(opath, exist_ok=True)



#=============================================================================
#-----------------------------------------------------------------------------
#  ** MODEL output load *** --------------------------------------------------
EXP_name1="CNTL"
EXP_name2="SKIN"
EXP_name3="CPLD"

# 1. WRF ----
idr_w = "/scratch/x3158a03/wrf_output/EAST-C/2008/AUTO/"
ifn_w = idr_w+f"sstx-ERA5-MetnoSST-2way_wrfout_Fog_{domain}_2020-08-17_18:00:00.nc"  #sst_fix
ifn_w2= idr_w+f"skin-ERA5-MetnoSST-2way_wrfout_Fog_{domain}_2020-08-17_18:00:00.nc"  # sst_skin

# 2. COAWST ----
idr_c = f"/scratch/x3158a03/coawst_output/2008/"
ifn_c = idr_c+f"WDM6-ERA5-SW-2way-YSU_wrfout_Fog_{domain}_2020-08-17_18:00:00.nc"  # solar_source HYCOM

ds_ct = nc.Dataset(ifn_w)
ds_sk = nc.Dataset(ifn_w2)
ds_cp = nc.Dataset(ifn_c)


# ------------------------------------------------
# Read output 
# (1) CNTL(ct) , SKIN(sk),  CPLD(cp)
stab_data = {
    'CNTL': {pos: [[] for _ in range(len(time_idx_list))] for pos in position_names},
    'SKIN': {pos: [[] for _ in range(len(time_idx_list))] for pos in position_names},
    'CPLD': {pos: [[] for _ in range(len(time_idx_list))] for pos in position_names}
}

stab_list = {
    'CNTL': {pos: [[] for _ in range(len(time_idx_list))] for pos in position_names},
    'SKIN': {pos: [[] for _ in range(len(time_idx_list))] for pos in position_names},
    'CPLD': {pos: [[] for _ in range(len(time_idx_list))] for pos in position_names}
}

lev_list = {
    'CNTL': {pos: [[] for _ in range(len(time_idx_list))] for pos in position_names},
    'SKIN': {pos: [[] for _ in range(len(time_idx_list))] for pos in position_names},
    'CPLD': {pos: [[] for _ in range(len(time_idx_list))] for pos in position_names}
}


i = 0
for position in position_list :
    pos_name = pos_cl[position]
    latt = lat_idx[position] 
    lonn = lon_idx[position]


    for idx, k in enumerate(time_idx_list) :

        z_ct   = wrf.getvar(ds_ct, 'z', timeidx=k)[:nlevel+1, latt, lonn]     # m (height)
        tv_ct  = wrf.getvar(ds_ct, 'tv', timeidx=k)[:nlevel+1,latt, lonn]    # K
        p_ct   = wrf.getvar(ds_ct, 'pressure', timeidx=k)[:nlevel+1, latt, lonn]  # hPa
        lsm_ct = wrf.getvar(ds_ct, 'LANDMASK', timeidx=0)[latt, lonn]
        
        z_sk   = wrf.getvar(ds_sk, 'z', timeidx=k)[:nlevel+1, latt, lonn]     # m (height)
        tv_sk  = wrf.getvar(ds_sk, 'tv', timeidx=k)[:nlevel+1, latt, lonn]    # K
        p_sk   = wrf.getvar(ds_sk, 'pressure', timeidx=k)[:nlevel+1, latt, lonn]  # hPa
        lsm_sk = wrf.getvar(ds_sk, 'LANDMASK', timeidx=0)[latt, lonn]
        
        z_cp   = wrf.getvar(ds_cp, 'z', timeidx=k)[:nlevel+1, latt, lonn]     # m (height)
        tv_cp  = wrf.getvar(ds_cp, 'tv', timeidx=k)[:nlevel+1, latt, lonn]    # K
        p_cp   = wrf.getvar(ds_cp, 'pressure', timeidx=k)[:nlevel+1, latt, lonn]  # hPa
        lsm_cp = wrf.getvar(ds_cp, 'LANDMASK', timeidx=0)[latt, lonn]
        
        
        z_ct  = z_ct.where(lsm_ct==0)  
        tv_ct = tv_ct.where(lsm_ct==0) 
        p_ct  = p_ct.where(lsm_ct==0)
        
        z_sk  = z_sk.where(lsm_sk==0)  
        tv_sk = tv_sk.where(lsm_sk==0) 
        p_sk  = p_sk.where(lsm_sk==0)
        
        z_cp  = z_cp.where(lsm_cp==0)  
        tv_cp = tv_cp.where(lsm_cp==0) 
        p_cp  = p_cp.where(lsm_cp==0)
        
    
        z_ct_np  = z_ct.values.squeeze()  
        tv_ct_np = tv_ct.values.squeeze()
        p_ct_np  = p_ct.values.squeeze() 
    
        z_sk_np  = z_sk.values.squeeze()  
        tv_sk_np = tv_sk.values.squeeze()
        p_sk_np  = p_sk.values.squeeze() 
    
        z_cp_np  = z_cp.values.squeeze()  
        tv_cp_np = tv_cp.values.squeeze()
        p_cp_np  = p_cp.values.squeeze() 
    
    
        f_ct_tv = interp1d(z_ct_np, tv_ct_np, bounds_error=False, fill_value=np.nan)
        f_sk_tv = interp1d(z_sk_np, tv_sk_np, bounds_error=False, fill_value=np.nan)
        f_cp_tv = interp1d(z_cp_np, tv_cp_np, bounds_error=False, fill_value=np.nan)
        f_ct_p  = interp1d(z_ct_np, p_ct_np, bounds_error=False, fill_value=np.nan)
        f_sk_p  = interp1d(z_sk_np, p_sk_np, bounds_error=False, fill_value=np.nan)
        f_cp_p  = interp1d(z_cp_np, p_cp_np, bounds_error=False, fill_value=np.nan)
    
        tv_ct = f_ct_tv(heights)  
        tv_sk = f_sk_tv(heights)
        tv_cp = f_cp_tv(heights)
        p_ct  = f_ct_p(heights)
        p_sk  = f_sk_p(heights)
        p_cp  = f_cp_p(heights)

        
        # Virutal temperature theta ----------- 
        the_v_ct = tv_ct * (p0 / p_ct) ** kappa
        the_v_sk = tv_sk * (p0 / p_sk) ** kappa
        the_v_cp = tv_cp * (p0 / p_cp) ** kappa

        stab_data['CNTL'][pos_name][idx] = the_v_ct #stab_ct
        stab_data['SKIN'][pos_name][idx] = the_v_sk #stab_sk
        stab_data['CPLD'][pos_name][idx] = the_v_cp #stab_cp
     

        #=============================================================================
        # Calculate d theta / d z
        #=============================================================================
         
        n_levels = the_v_ct.shape[0]
        
        
        stab_ct = np.zeros_like(the_v_ct)
        stab_sk = np.zeros_like(the_v_sk)
        stab_cp = np.zeros_like(the_v_cp)
        
        
        for ll in range(0, n_levels - 1):

            dth_ct = the_v_ct[ll+1] - the_v_ct[ll]
            dz_ct  = z_space
        
            dth_sk = the_v_sk[ll+1] - the_v_sk[ll]
            dz_sk  = z_space
        
            dth_cp = the_v_cp[ll+1] - the_v_cp[ll]
            dz_cp  = z_space
        
            stab_ct[ll] = dth_ct / dz_ct  # K/m
            stab_sk[ll] = dth_sk / dz_sk  # K/m
            stab_cp[ll] = dth_cp / dz_cp  # K/m
       
 
        stab_ct_max = np.nanmax(stab_ct, axis=0)                 
        stab_sk_max = np.nanmax(stab_sk, axis=0)                 
        stab_cp_max = np.nanmax(stab_cp, axis=0)

        idx_ct = np.nanargmax(stab_ct, axis=0)
        idx_sk = np.nanargmax(stab_sk, axis=0)
        idx_cp = np.nanargmax(stab_cp, axis=0)
        z_at_ct = heights[idx_ct]  
        z_at_sk = heights[idx_sk]  
        z_at_cp = heights[idx_cp]  

        stab_list['CNTL'][pos_name][idx].append(stab_ct_max)
        stab_list['SKIN'][pos_name][idx].append(stab_sk_max)
        stab_list['CPLD'][pos_name][idx].append(stab_cp_max)
        lev_list['CNTL'][pos_name][idx].append(z_at_ct)       
        lev_list['SKIN'][pos_name][idx].append(z_at_sk)       
        lev_list['CPLD'][pos_name][idx].append(z_at_cp)       
        
 

# ================================================
#  Plotting ======================================
# ================================================
liw = 2.2
fts = 10
tis = 15
alpbet = [f"({chr(97+i)})" for i in range(12)]

xmin, xmax = 297, 308
offset = 0.2

fig, axes = plt.subplots(3, 4, figsize=(12, 10))


for row, pos_name in enumerate(position_names):
    for col in range(4) :
        ax = axes[row, col]

        ax.plot(stab_data['CNTL'][pos_name][col], heights, '#6F8FAF', linewidth=liw, label=EXP_name1)
        ax.plot(stab_data['SKIN'][pos_name][col], heights, 'b-', linewidth=liw, label=EXP_name2)
        ax.plot(stab_data['CPLD'][pos_name][col], heights, 'r-', linewidth=liw, label=EXP_name3)

        ax.set_xlim([xmin, xmax])
        ax.xaxis.set_major_locator(plt.MultipleLocator(2))
        ax.set_ylim([0, 402])
        ax.yaxis.set_major_locator(plt.MultipleLocator(50))
     

        ax.scatter(xmax-offset, lev_list['CNTL'][pos_name][col], color='#6F8FAF', s=40, zorder=5, clip_on= False )
        ax.scatter(xmax, lev_list['SKIN'][pos_name][col], color='b'      , s=40, zorder=5, clip_on= False )
        ax.scatter(xmax+offset, lev_list['CPLD'][pos_name][col], color='r'      , s=40, zorder=5, clip_on= False )


        if col ==0:
            ax.set_ylabel('Height [m]', fontsize=fts)
        if row == 2:
            ax.set_xlabel('θ$_{v}$', fontsize=fts)  
 
        if row == 0:
            LST_TIME=time_kst_list[col] 
            ax.text(0.02, 0.98, f'{(alpbet[col+row*4])} {pos_name}\n {LST_TIME}LST', transform=ax.transAxes,
                    fontsize=fts+1, verticalalignment='top', fontweight='bold')
        else:
            ax.text(0.02, 0.98, f'{(alpbet[col+row*4])} {pos_name}', transform=ax.transAxes,
                    fontsize=fts+1, verticalalignment='top', fontweight='bold')

        
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)

        if row == 0 and col == 0 :
            ax.legend(fontsize=10, loc='upper left', ncol=1,  frameon=False,
                      bbox_to_anchor=(0.02, 0.8),
                      handlelength=1.2, handletextpad=0.4, columnspacing=0.7, labelspacing=0.25)

    
plt.tight_layout()
plt.savefig(f"{ofn}.png", dpi=400, bbox_inches='tight')
plt.show()

plt.close()



