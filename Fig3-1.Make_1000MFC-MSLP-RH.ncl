;============================================
;  Fig3_1.Middle-map_1000MFC-MSLP-RH.ncl
;   - figure 3. base field and convergence
;   - 2row 3col
;   - MSLP (pattern) with H/L pressure + 'RH'
;   - Moisture flux convergence
;============================================

begin

; --- time ----
;day_date = (/"18_00LST", "18_06LST","19_06LST"/)
;date_idxp = (/21+24-6,  21+24, 27+6+24+12/)
;date_idxs = (/21+24+24-6,  21+24+24, 27+6+24+12+24/)

;day_date = (/"18_12LST", "18_18LST","19_00LST"/)
;date_idxp = (/27+24,  27+6+24, 27+6+24+6/)
;date_idxs = (/27+24+24, 27+6+24+24, 27+6+24+6+24/)

day_date= (/"18_06LST", "18_12LST","18_18LST"/)
date_idxp = (/21+24,  21+6+24, 27+6+24/)
date_idxs = (/21+24+24,  21+6+24+24, 27+6+24+24/)

ntime = dimsizes(date_idxp)
print("ntime="+ntime)

; --- domain ---
ilon=120
elon=137
ilat=30
elat=45

pressure = 1000

; --- Read File ---
idr = "/scratch/x3158a03/DATA/Reanalysis/2008/"
idr_lsm = "/scratch/x3158a03/DATA/Reanalysis/"

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
; print("nlat"+nlat) ; 61

p = new((/ntime,nlat,nlon/),float)
;u1000 = new((/ntime,nlat,nlon/),float)
;v1000 = new((/ntime,nlat,nlon/),float)
;r1000 = new((/ntime,nlat,nlon/),float)
;t1000 = new((/ntime,nlat,nlon/),float)


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
printVarSummary(rh2m)


;-----------------------------------------
;--- make moisture flux
p = 1000
copy_VarCoords(v1000, p)
q = mixhum_ptrh(p, t1000, r1000, -1) ; 
;print(q)
;printVarSummary("mixhum_ptrh(p, t1000, r1000, -1): q = "+q)
print("=====================================================")

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

  printVarSummary(mfc_adv)
  printMinMax(mfc_adv, 0)
  print("mfc_adv =" + mfc_adv)
  print("--------")

;------convergecne
duv=uv2dvF_Wrap(u1000, v1000) 	; duv = du/dx + dv/dy
print("duv =" + duv(0,50,10))


;;mfc_con = (-q*duv)
mfc_con = (-q*duv)*1e5 ; 100000
copy_VarCoords(mfc_adv,mfc_con)

printVarSummary(mfc_con)
print(mfc_con)
;---calcu total MFC
mfc = (mfc_adv + mfc_con)
mfc@long_name = "Moisture Flux Convergence"
;mfc@units     = "g/(kg-s)"  ; (g/kg)(1/s) => g/(kg-s)

copy_VarCoords(mfc_adv,mfc)
printVarSummary(mfc)
print("mfc = " + mfc)

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
fout->mfc_con = mfc_con
fout->uq      = uq
fout->vq      = vq
fout->rh2m    = rh2m



exit
      ;============================================
      ; Plotting ==================================
      ;============================================
      plt_rh = new(ntime,graphic)  ; row1 : RH+MSLP
      plt_lp = new(ntime,graphic)
      plt_mf = new(ntime,graphic)  ; row2 : MFC+vector
      plt_mc = new(ntime,graphic)

      wks = gsn_open_wks("png","./"+tostring(pressure)+"-MSLP-RH-MoistureFlux_")
      

        ; (Row 1-1) RH ----------------------------
        resr			  = True
	resr@gsnFrame             = False
        resr@gsnDraw              = False
        resr@gsnAddCyclic         = False            ; regional data - not global
        resr@gsnLeftString 	  = ""
        resr@gsnRightString 	  = ""
        

        resr@mpOutlineOn          = True
        resr@mpFillOn             = False
        resr@mpLimitMode          = "LatLon"
        resr@mpMinLonF            = ilon
        resr@mpMaxLonF            = elon
        resr@mpMinLatF            = ilat
        resr@mpMaxLatF            = elat

        resr@cnFillOn             = True
        resr@cnLinesOn            = False
        resr@cnLineLabelsOn       = False
        resr@lbLabelBarOn         = False
        resr@cnLevelSelectionMode = "ExplicitLevels"
        resr@cnLevels             = (/95, 96, 97, 98 , 99 , 100/)  ; 값에 맞게 조정
        resr@cnFillColors         = (/49, 81, 113, 145, 177, 209/)

        resr@tiMainFont           = "times-roman"
        resr@tmXBLabelFont        = "times-roman"
        resr@tmYRLabelFont        = "times-roman"

        resr@tmBorderThicknessF   = 2.5 
        resr@tmYROn               = False
        resr@tmYLOn               = False   ; 왼쪽도 off
        resr@tmYLLabelsOn         = False
        resr@tmYRLabelsOn         = False

        gsn_define_colormap(wks, "WhiteBlue")
        plt_rh(0) = gsn_csm_contour_map(wks, rh2m(0,:,:), resr)
        plt_rh(1) = gsn_csm_contour_map(wks, rh2m(1,:,:), resr)

        resr@tmYROn               = True
        resr@tmYRLabelsOn         = True
        resr@tmYLOn               = False
        resr@tmYLLabelsOn         = False
        plt_rh(2) = gsn_csm_contour_map(wks, rh2m(2,:,:), resr)


        ; (Row 1-2) MSLP ------------------------------
        resm			= True                 
        resm@gsnFrame           = False
        resm@gsnDraw            = False
        resm@gsnAddCyclic       = False            ; regional data - not global
        resm@gsnLeftString 	  = ""
        resm@gsnRightString 	  = ""

        resm@cnFillOn             = False
        resm@cnInfoLabelOn 	  = False      ; 등고선 정보 off
        resm@cnLinesOn            = True
        resm@cnLineLabelsOn       = True
        resm@cnLineLabelFont      = "times-roman"
        resm@cnLineLabelFontHeightF = 0.012
        resm@cnLevelSelectionMode = "ManualLevels"
        resm@cnMinLevelValF       = 1000.0
        resm@cnMaxLevelValF       = 1030.0
        resm@cnLevelSpacingF      = 2.0
        resm@cnLineColor          = "Black"
        resm@cnLineThicknessF     = 1.8

        resm@gsnHighLowLabelsOn   = True   ; H/L 표시
        resm@gsnHighLabelString   = "H"
        resm@gsnLowLabelString    = "L"
        resm@gsnHighLabelColor    = "blue"
        resm@gsnLowLabelColor     = "red"
        resm@gsnHighLabelFontHeightF = 0.02
        resm@gsnLowLabelFontHeightF  = 0.02

        plt_lp(0) = gsn_csm_contour(wks, mslp(0,:,:), resm)
        plt_lp(1) = gsn_csm_contour(wks, mslp(1,:,:), resm)
        plt_lp(2) = gsn_csm_contour(wks, mslp(2,:,:), resm)
       

        overlay(plt_rh(0), plt_lp(0))
        overlay(plt_rh(1), plt_lp(1))
        overlay(plt_rh(2), plt_lp(2))
        

        ; (Row 2-1) 1000-hPa moisture flux ---------
        res                       = True
        res@gsnFrame		  = False
        res@gsnDraw		  = False
        res@gsnAddCyclic          = False            ; regional data - not global
 
        res@vcRefMagnitudeF       = 15               ; vec magnitude
        res@vcRefLengthF          = 0.01             ; size of ref vector
        res@vcLineArrowThicknessF = 4
        res@vcLineArrowColor = "Black"
;        res@vcMonoLineArrowColor  = True        
;        res@vcMinDistanceF        = 0.015            ; thins arrows near pole
        res@vcGlyphStyle          = "LineArrow"      ; turn on curly vectors
     
 
        gsn_define_colormap(wks, "BrownBlue12")
        plt_mf(0) = gsn_csm_vector(wks,uq(0,::5,::5),vq(0,::5,::5),res)
        plt_mf(1) = gsn_csm_vector(wks,uq(1,::5,::5),vq(1,::5,::5),res)
        plt_mf(2) = gsn_csm_vector(wks,uq(2,::5,::5),vq(2,::5,::5),res)

 
        ; (3-2) 1000-hPa moisture flux convergence ---------
        res2                   	= True             ; plot mods desired
        res2@gsnDraw           	= False            ; don't draw yet
        res2@gsnFrame          	= False            ; don't advance frame yet
        res2@gsnAddCyclic      	= False     
        res2@mpShapeMode 	= "FreeAspect" ; 비율을 자유롭게 설정할 수 있게 함
        res2@vpWidthF    	= 0.4          ; 뷰포트 가로 길이 (0.0 ~ 1.0 사이)
        res2@vpHeightF   	= 0.4          ; 뷰포트 세로 길이 (가로와 비슷하게 맞춤) 
        res2@mpOutlineOn       	= True

        res2@cnFillOn          	= True             ; turn on color
        res2@cnLinesOn         	= False            ; turn off contour lines
        res2@cnLineLabelsOn    	= False            ; turn off contour lines
        res2@mpFillOn          	= False            ; turn off map fill
        res2@lbLabelBarOn      	= False            ; turn off individual cb's
        res2@lbBoxEndCapStyle  	= "TriangleBothEnds"

        res2@cnLevelSelectionMode = "ExplicitLevels"
        ;res2@cnLevels 		= (/-6,-4,-2,-1,-0.6,-0.2,0.2,0.6,1,2,4,6/)
        ;res2@cnLevels 		= (/-6,-5,-4,-3,-2,-1,1,2,3,4,5,6/)
        res2@cnLevels 		= (/-10,-8,-6,-4,-2,-1,1,2,4,6,8,10/)

        res2@cnFillColors     	= (/1,2,3,4,5,6,0,8,9,11,12,13,14/) ; set White-in-Middle color map

        res2@tmXBMinorOn 	= False           ; no lon minor tickmarks
        res2@tmYLMinorOn 	= False           ; no lat minor tickmarks
        res2@tmXBMajorLengthF 	= 0.0
        res2@tmYLMajorLengthF 	= 0.0
        res2@mpLimitMode 	= "LatLon"
        res2@mpMinLonF   	= ilon
        res2@mpMaxLonF   	= elon
        res2@mpMinLatF   	= ilat
        res2@mpMaxLatF   	= elat

        res2@tiMainFont       = "times-roman"  ; 제목
        res2@tmXBLabelFont    = "times-roman"  ; x축 라벨
        res2@tmYLLabelFont    = "times-roman"  ; y축 라벨
        res2@lbLabelFont      = "times-roman"  ; colorbar 라벨
      
        res2@tmBorderThicknessF   = 1.5 
        res2@tmYROn               = False
        res2@tmYLOn               = False
        res2@tmYLLabelsOn         = False
        res2@tmYRLabelsOn         = False

        plt_mc(0) = gsn_csm_contour_map(wks,mfc_sea(0,:,:),res2)
        plt_mc(1) = gsn_csm_contour_map(wks,mfc_sea(1,:,:),res2)

        res2@tmYROn               = True
        res2@tmYRLabelsOn         = True
        res2@tmYLOn               = False
        res2@tmYLLabelsOn         = False
        plt_mc(2) = gsn_csm_contour_map(wks,mfc_sea(2,:,:),res2)
      
        overlay(plt_mc(0),plt_mf(0))
        overlay(plt_mc(1),plt_mf(1))
        overlay(plt_mc(2),plt_mf(2))

        
        ;--- Panel 2row 3col ---
        plots_all = new(ntime*2, graphic)
        plots_all(0) = plt_rh(0)
        plots_all(1) = plt_rh(1)
        plots_all(2) = plt_rh(2)
        plots_all(3) = plt_mc(0)
        plots_all(4) = plt_mc(1)
        plots_all(5) = plt_mc(2)

        ;--- Row 1 패널: RH + MSLP ---
        pnlres1                             = True
        pnlres1@gsnPanelFigureStrings  	    = (/"(a)","(b)","(c)"/)
        pnlres1@gsnPanelBottom              = 0.5   ; 상단 절반에 배치
        pnlres1@gsnPanelYWhiteSpacePercent  = 5
        pnlres1@gsnPanelXWhiteSpacePercent  = 3
        pnlres1@gsnPanelLabelBar            = True
        pnlres1@lbLabelFont                 = "times-roman"
        pnlres1@lbLabelFontHeightF          = 0.015
        pnlres1@pmLabelBarWidthF            = 0.6
        pnlres1@pmLabelBarHeightF           = 0.05
        pnlres1@lbTitleString               = "2m Relative Humidity [%]"
        pnlres1@lbTitleFontHeightF          = 0.015
        pnlres1@lbTitleDirection            = "Across"
        pnlres1@gsnFrame                    = False   ; frame 넘기지 않음

        gsn_panel(wks, plt_rh, (/1,3/), pnlres1)
        
        ;--- Row 2 패널: MFC + vector ---
        pnlres2                             = True
        pnlres1@gsnFrame                    = True   ; frame 넘기지 않음
        pnlres2@gsnPanelTop                 = 0.5   ; 하단 절반에 배치
        pnlres2@gsnPanelYWhiteSpacePercent  = 5
        pnlres2@gsnPanelXWhiteSpacePercent  = 3
        pnlres2@gsnPanelLabelBar            = True
        pnlres2@lbLabelFont                 = "times-roman"
        pnlres2@lbLabelFontHeightF          = 0.015
        pnlres2@pmLabelBarWidthF            = 0.6
        pnlres2@pmLabelBarHeightF           = 0.05
        pnlres2@lbTitleString               = "MFC [g kg~S~-1~N~ s~S~-1~N~]"
        pnlres2@lbTitleFontHeightF          = 0.015
        pnlres2@lbTitleDirection            = "Across"
                
        gsn_panel(wks, plt_mc, (/1,3/), pnlres2)




end

exit

        pnlres_rh = True
        ;pnlres_rh@gsnPanelMainString 	= "2016_JAN_500 hPa geopotential height_Composite"
        pnlres_rh@gsnPanelYWhiteSpacePercent = 4
        pnlres_rh@gsnPanelXWhiteSpacePercent = 4
      ;  pnlres_rh@lbTitleString    	= "gpm"                ; title string
      ;  pnlres_rh@lbTitlePosition  	= "Right"              ; title position
        pnlres_rh@lbTitleFontHeightF	= .15               ; make title smaller
        pnlres_rh@lbTitleDirection 	= "Across"             ; title direction
        pnlres_rh@gsnPanelLabelBar 	= True
        pnlres_rh@lbLabelFont              = "times-roman"  ; colorbar 폰트
        pnlres_rh@lbLabelFontHeightF       = 0.015          ; colorbar 라벨 폰트 크기
        pnlres_rh@pmLabelBarWidthF         = 0.6            ; colorbar 가로 폭 (0.0~1.0)
        pnlres_rh@pmLabelBarHeightF        = 0.05           ; colorbar 두께 (0.0~1.0)
     
       
        pnlres_mc = True
        pnlres_rh@gsnPanelYWhiteSpacePercent = 4
        pnlres_rh@gsnPanelXWhiteSpacePercent = 4
        


        gsn_panel(wks, plots_all, (/2,3/), pnlres)

end

