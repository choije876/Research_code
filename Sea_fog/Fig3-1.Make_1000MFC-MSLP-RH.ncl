;============================================
;  Fig3_1.Middle-map_1000MFC-MSLP-RH.ncl
;   - figure 3. base field and convergence
;   - 2row 3col
;   - MSLP (pattern) with H/L pressure + 'RH'
;   - Moisture flux convergence
;============================================

begin

; --- time ----
day_date= (/"18_06LST", "18_12LST","18_18LST"/)
date_idxp = (/21+24,  21+6+24, 27+6+24/)
date_idxs = (/21+24+24,  21+6+24+24, 27+6+24+24/)

ntime = dimsizes(date_idxp)

; --- domain ---
ilon=120
elon=137
ilat=30
elat=45

pressure = 1000

; --- Read File ---
idr = "./"
idr_lsm = "./"

ifnp	= addfile(idr + "ERA5_hour-pres_2020-08-1619.nc", "r")
ifns	= addfile(idr + "ERA5_hour_sfc_2020-08-1519.nc", "r")
ifn_lsm = addfile(idr_lsm + "era5_hour_landmask.nc", "r")

u100 = ifnp->u(date_idxp,{pressure},::-1,{ilon:elon})
v100 = ifnp->v(date_idxp,{pressure},::-1,{ilon:elon})
t100 = ifnp->t(date_idxp,{pressure},::-1,{ilon:elon})
r100 = ifnp->r(date_idxp,{pressure},::-1,{ilon:elon})

psfc = ifns->msl(date_idxs, ::-1,{ilon:elon})
t2   = ifns->t2m(date_idxs, ::-1,{ilon:elon})
d2   = ifns->d2m(date_idxs, ::-1,{ilon:elon})

land = ifn_lsm->lsm(0,::-1,{ilon:elon})

u1000 = u100(:,{ilat:elat},:)
v1000 = v100(:,{ilat:elat},:)
t1000 = t100(:,{ilat:elat},:)
r1000 = r100(:,{ilat:elat},:)
mslp = psfc(:,{ilat:elat},:) 
t2m  = t2(:,{ilat:elat},:)
d2m  = d2(:,{ilat:elat},:)
landmask = land({ilat:elat},:)  ; 0: land, 1: ocean

nlat = dimsizes(u1000(0,:,0))
nlon = dimsizes(u1000(0,0,:))

p = new((/ntime,nlat,nlon/),float)


;-----------------------------------------
;--- make MSLP
mslp = mslp / 1.0e2 
copy_VarCoords(psfc(:,{ilat:elat},:),mslp)

;-----------------------------------------
;--- make RH
es = 6.112 * exp(17.67 * (t2m-273.15) / ((t2m-273.15)+243.5) ) ;hPa
e  = 6.112 * exp(17.67 * (d2m-273.15) / ((d2m-273.15)+243.5) ) ;hPa
rh2m = (e/es) * 100
copy_VarCoords(t2m, rh2m)
rh2m@units     = "%"
rh2m@long_name = "2m Relative Humidity"


;-----------------------------------------
;--- make moisture flux
p = 1000
copy_VarCoords(v1000, p)
q = mixhum_ptrh(p, t1000, r1000, -1) ; 

copy_VarCoords(r1000, q)
vq = v1000*q
uq = u1000*q
copy_VarCoords(v1000, vq)
copy_VarCoords(u1000, uq)

;---make MFC_convergence
;---advection
; MFC = -div(qV) = -(u*dq/dx + v*dq/dy) - q*div(V)
; advect_variable : (u*dq/dx + v*dq/dy)

long_name = "MFC_advection"
units     = "g/(kg-s)"       ; (m/s)*(g/kg)*(1/m) => (m/s)*(g/kg-m) => g/(kg-s)
gridType  = 1
opt_adv   = 0
mfc_adv   = advect_variable(u1000,v1000,q, gridType, long_name, units, opt_adv)
mfc_adv   = -mfc_adv*1e5
;mfc_adv   = -mfc_adv*10000


;------convergecne
duv=uv2dvF_Wrap(u1000, v1000) 	; duv = du/dx + dv/dy


;;mfc_con = (-q*duv)
mfc_con = (-q*duv)*1e5 ; 100000
copy_VarCoords(mfc_adv,mfc_con)

;---calcu total MFC
mfc = (mfc_adv + mfc_con)
mfc@long_name = "Moisture Flux Convergence"

copy_VarCoords(mfc_adv,mfc)

; landsea_mask ---
lsmk = conform(mfc, landmask, (/1,2/))
mfc_sea = where(lsmk.le.0.5, mfc, 0) ; mfc@_FillValue)
copy_VarCoords(mfc, mfc_sea)


;===============================
; Save MFC, MF data 
;===============================

system("rm -f mfc_output.nc")

; 변수 속성 설정 후 저장
mfc_sea@long_name = "Moisture Flux Convergence"
mfc_sea@units     = "g kg-1 s-1 (x1e-5)"

mfc_adv@long_name = "MFC Advection Term"
mfc_adv@units     = "g kg-1 s-1 (x1e-5)"

mfc_con@long_name = "MFC Convergence Term"
mfc_con@units     = "g kg-1 s-1 (x1e-5)"

uq@long_name      = "Zonal Moisture Flux"
uq@units          = "m s-1 g kg-1"

vq@long_name      = "Meridional Moisture Flux"
vq@units          = "m s-1 g kg-1"

rh2m@long_name    = "2m Relative Humidity"
rh2m@units        = "%"

; valid_time 좌표를 float으로 변환하여 저장
mfc_sea!0 = "time"
mfc_adv!0 = "time"
mfc_con!0 = "time"
uq!0      = "time"
vq!0      = "time"
rh2m!0    = "time"

delete(mfc_sea&time)
delete(mfc_adv&time)
delete(mfc_con&time)
delete(uq&time)
delete(vq&time)
delete(rh2m&time)

fout = addfile("mfc_output.nc", "c")

fout->mfc     = mfc_sea
fout->mfc_adv = mfc_adv
fout->vq      = vq
fout->rh2m    = rh2m


