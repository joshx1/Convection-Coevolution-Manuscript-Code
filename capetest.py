import netCDF4 as nc4
import xarray as xr
import sys
from netCDF4 import Dataset
from pathlib import Path
import numpy as np
import metpy.calc as mpcalc
from metpy.plots import SkewT
from metpy.units import units
from scipy import stats
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn import linear_model, svm
from global_land_mask import globe
import gc
from tqdm import tqdm
import xarray_parcel.modules.parcel_functions as parcel

np.seterr(divide='ignore', invalid='ignore')

def satur(pressure,temperature):
    func = mpcalc.saturation_mixing_ratio(pressure,temperature)
    return xr.apply_ufunc(func,pressure,temperature)

mode = 3
## Daily Control
if mode == 1:
    #l_prect=[]
    #l_prect.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/timselsum-daily.output.PRECT.nc')

    #l_preccdzm=[]
    #l_preccdzm.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/timselsum-daily.output.PRECCDZM.nc')

    #l_precc=[]
    #l_precc.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/timselsum-daily.output.PRECC.nc')

    #l_precl=[]
    #l_precl.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/timselsum-daily.output.PRECL.nc')

    #l_precsc=[]
    #l_precsc.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/timselsum-daily.output.PRECSC.nc')


    #l_ta=[]
    #l_ta.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/pressure-levels/timselsum-daily.output.T.nc')

    #l_hus=[]
    #l_hus.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/pressure-levels/timselsum-daily.output.Q.nc')

    #l_u=[]
    #l_u.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/pressure-levels/timselsum-daily.output.U.nc')

    #l_v=[]
    #l_v.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/pressure-levels/timselsum-daily.output.V.nc')

    #l_omega=[]
    #l_omega.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/pressure-levels/timselsum-daily.output.OMEGA.nc')

    l_prect=[]
    l_prect.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/output.PRECT.nc')

    l_preccdzm=[]
    l_preccdzm.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/output.PRECCDZM.nc')

    l_precc=[]
    l_precc.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/output.PRECC.nc')

    l_precl=[]
    l_precl.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/output.PRECL.nc')

    l_precsc=[]
    l_precsc.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/output.PRECSC.nc')


    l_ta=[]
    l_ta.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/pressure-levels/output.T.nc')

    l_hus=[]
    l_hus.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/pressure-levels/output.Q.nc')

    l_omega=[]
    l_omega.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/pressure-levels/output.OMEGA.nc')


## Shallow Off
if mode == 2:
    #l_prect=[]
    #l_prect.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/timselsum-daily.output.PRECT.nc')

    #l_preccdzm=[]
    #l_preccdzm.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/timselsum-daily.output.PRECCDZM.nc')

    #l_precc=[]
    #l_precc.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/timselsum-daily.output.PRECC.nc')
    #l_precl=[]
    #l_precl.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/timselsum-daily.output.PRECL.nc')

    #l_precsc=[]
    #l_precsc.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/timselsum-daily.output.PRECSC.nc')

    #l_ta=[]
    #l_ta.append('/g/data1b/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/pressure-levels/timselsum-daily.output.T.nc')

    #l_hus=[]
    #l_hus.append('/g/data1b/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/pressure-levels/timselsum-daily.output.Q.nc')

    #l_u=[]
    #l_u.append('/g/data1b/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/pressure-levels/timselsum-daily.output.U.nc')

    #l_v=[]
    #l_v.append('/g/data1b/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/pressure-levels/timselsum-daily.output.V.nc')

    #l_omega=[]
    #l_omega.append('/g/data1b/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/pressure-levels/timselsum-daily.output.OMEGA.nc')

    l_prect=[]
    l_prect.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/output.PRECT.nc')

    l_preccdzm=[]
    l_preccdzm.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/output.PRECCDZM.nc')

    l_precc=[]
    l_precc.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/output.PRECC.nc')

    l_precl=[]
    l_precl.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/output.PRECL.nc')

    l_precsc=[]
    l_precsc.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/output.PRECSC.nc')


    l_ta=[]
    l_ta.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/pressure-levels/output.T.nc')

    l_hus=[]
    l_hus.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/pressure-levels/output.Q.nc')

    l_omega=[]
    l_omega.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/pressure-levels/output.OMEGA.nc')



# Deep conv off
if mode == 3:
    #l_prect=[]
    #l_prect.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--deep-off--daily-output/timselsum-daily.output.PRECT.nc')

    #l_preccdzm=[]
    #l_preccdzm.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--deep-off--daily-output/timselsum-daily.output.PRECC.nc')

    #l_precc=[]
    #l_precc.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--deep-off--daily-output/timselsum-daily.output.PRECC.nc')

    #l_precl=[]
    #l_precl.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--deep-off--daily-output/timselsum-daily.output.PRECL.nc')

    #l_precsc=[]
    #l_precsc.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--deep-off--daily-output/timselsum-daily.output.PRECSC.nc')

    #l_ta=[]
    #l_ta.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--deep-off--daily-output/pressure-levels/timselsum-daily.output.T.nc')

    #l_hus=[]
    #l_hus.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--deep-off--daily-output/pressure-levels/timselsum-daily.output.Q.nc')

    #l_u=[]
    #l_u.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--deep-off--daily-output/timselsum-daily.output.U.nc')

    #l_v=[]
    #l_v.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--deep-off--daily-output/timselsum-daily.output.V.nc')

    #l_omega=[]
    #l_omega.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--deep-off--daily-output/pressure-levels/timselsum-daily.output.OMEGA.nc')

    l_prect=[]
    l_prect.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--deep-off--daily-output/output.PRECT.nc')

    l_preccdzm=[]
    l_preccdzm.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--deep-off--daily-output/output.PRECCDZM.nc')

    l_precc=[]
    l_precc.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--deep-off--daily-output/output.PRECC.nc')

    l_precl=[]
    l_precl.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--deep-off--daily-output/output.PRECL.nc')

    l_precsc=[]
    l_precsc.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--deep-off--daily-output/output.PRECSC.nc')


    l_ta=[]
    l_ta.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--deep-off--daily-output/pressure-levels/output.T.nc')

    l_hus=[]
    l_hus.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--deep-off--daily-output/pressure-levels/output.Q.nc')

    l_omega=[]
    l_omega.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--deep-off--daily-output/pressure-levels/output.OMEGA.nc')



#tgroup='12h'
#
#l_pr=[]
#l_pr.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/timselsum-'+tgroup+'.output.PRECT.nc')
#
#l_ta=[]
#l_ta.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/pressure-levels/timselsum-'+tgroup+'.output.T.nc')
#
#l_hus=[]
#l_hus.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/pressure-levels/timselsum-'+tgroup+'.output.Q.nc')
#
#l_u=[]
#l_u.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/pressure-levels/timselsum-'+tgroup+'.output.U.nc')
#
#l_v=[]
#l_v.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/pressure-levels/timselsum-'+tgroup+'.output.V.nc')
#
#l_omega=[]
#l_omega.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/pressure-levels/timselsum-'+tgroup+'.output.OMEGA.nc')

#l_pr=[]
#l_pr.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/timselsum-daily.output.PRECT.nc')
#
#l_ta=[]
#l_ta.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/pressure-levels/timselsum-daily.output.T.nc')
#
#l_hus=[]
#l_hus.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/pressure-levels/timselsum-daily.output.Q.nc')
#
#l_u=[]
#l_u.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/pressure-levels/timselsum-daily.output.U.nc')
#
#l_v=[]
#l_v.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/pressure-levels/timselsum-daily.output.V.nc')
#
#l_omega=[]
#l_omega.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/pressure-levels/timselsum-daily.output.OMEGA.nc')

#l_pr=[]
#l_pr.append('/g/data1b/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/timselsum-daily.output.PRECT.nc')
#
#l_ta=[]
#l_ta.append('/g/data1b/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/pressure-levels/timselsum-daily.output.T.nc')
#
#l_hus=[]
#l_hus.append('/g/data1b/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/pressure-levels/timselsum-daily.output.Q.nc')
#
#l_u=[]
#l_u.append('/g/data1b/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/pressure-levels/timselsum-daily.output.U.nc')
#
#l_v=[]
#l_v.append('/g/data1b/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/pressure-levels/timselsum-daily.output.V.nc')
#
#l_omega=[]
#l_omega.append('/g/data1b/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/pressure-levels/timselsum-daily.output.OMEGA.nc')

hours_grouped=1

#ds_omega=xr.open_mfdataset(l_omega,combine='by_coords')
#ds_u=xr.open_mfdataset(l_u,combine='by_coords')
#ds_v=xr.open_mfdataset(l_v,combine='by_coords')
ds_hus=xr.open_mfdataset(l_hus,combine='by_coords')
ds_pr = xr.open_mfdataset(l_prect,combine='by_coords')
ds_ta = xr.open_mfdataset(l_ta,combine='by_coords')

#print(ds_u.variables)
#sys.exit()
hus = ds_hus['Q'].sel(lat = slice(-15,15)) 
ta = ds_ta['T'].sel(lat = slice(-15,15)) 
precip = ds_pr['PRECT'].sel(lat = slice(-15,15)) 
pressure = ds_hus['plev']
#omega = ds_omega['OMEGA'].sel(lat = slice(-15,15)) 
lat =ds_hus['lat'].sel(lat = slice(-15,15)) 
lon =ds_hus['lon']
print(lat[10])
print(lon[10])
pressure = 100*pressure.data
#print(pr1)
pr = np.zeros(shape=(17,16,len(lon)))

for i in range(0,17):
    pr[i,:,:] = pressure[i].data

pr = pr*units.pascal
lon_grid, lat_grid = np.meshgrid(np.add(lon.load(),-180),lat.load())
mask = globe.is_land(lat_grid, lon_grid) 

lag = 1
counts = 0
precip_lower =0
csf_lower=0
precip_higher=100
csf_higher=1
bins=30
csf_binvalues = np.linspace(csf_lower,csf_higher,bins)
precip_binvalues = np.linspace(precip_lower,precip_higher,bins)
counts=np.zeros(shape=(bins+1,bins+1))
dec_count=np.zeros(shape=(bins+1,bins+1))
temp_ocean = np.zeros(shape=(731,17,1))
hus_ocean = np.zeros(shape=(731,17,1))
lat = lat.load()
lon_grid, lat_grid = np.meshgrid(np.add(lon.load(),-180),lat[:])
mask = globe.is_land(lat_grid, lon_grid)
print(np.sum(mask))
#sys.exit()

print(np.shape(pressure))
tempmean = np.zeros(shape=(30,16,144))
tempmean1 = np.nanmean(np.nanmean(np.flip(ta[10,:,:,:].load().data,axis=0),axis=2),axis=1)

p = np.divide(pressure,100)*units.hectopascals
print(p)
T = np.subtract(np.divide(ta[10,:,10,10].load().data,24),273.15)*units.degC
print(T)
Td = mpcalc.dewpoint(mpcalc.vapor_pressure(p,np.divide(hus[10,:,10,10],24))).to('degC')
print(Td)
#prof = mpcalc.parcel_profile(p,T[0],Td[0]).to('degC')
#print(prof)
#fig = plt.figure(figsize=(9, 9))
#skew = SkewT(fig, rotation=45)
#
## Plot the data using normal plotting functions, in this case using
## log scaling in Y, as dictated by the typical meteorological plot.
#skew.plot(p, T, 'r')
#skew.plot(p, Td, 'g')
#skew.ax.set_ylim(1000, 10)
#skew.ax.set_xlim(-40,60)
#skew.plot(p, prof, 'k', linewidth=2)
#
## Shade areas of CAPE and CIN
#skew.shade_cin(p, T, prof)
#skew.shade_cape(p, T, prof)
#
## An example of a slanted line at constant T -- in this case the 0
## isotherm
#skew.ax.axvline(0, color='c', linestyle='--', linewidth=2)
#
## Add the relevant special lines
#skew.plot_dry_adiabats()
#skew.plot_moist_adiabats()
#skew.plot_mixing_lines()
#
## Show the plot
#plt.savefig('parcelprofile.png')
#plt.close()
#print(mpcalc.cape_cin(p,T,Td,prof)[0].magnitude)
#print(temp1)
#sys.exit()
cape = np.zeros(shape=(16,144))

for i in range(0,600):
    f = open("capetime.txt","r")
    t = int(f.readline())
    f.close()
    print(t)
    t = t*12
    #temp1 = np.subtract(np.divide(ta[t,:,:,:].load().data,24),273.15)*units.degC
    #hus1 = np.divide(hus[t,:,:,:].load().data,24)
    temp1 = np.subtract(ta[t,:,:,:].load().data,273.15)*units.degC
    hus1 = hus[t,:,:,:].load().data
    Td1 = mpcalc.dewpoint(mpcalc.vapor_pressure(pr,hus1)).to('degC')
    for lon in tqdm(range(0,144)):
        for lat in range(0,16):
            prof = mpcalc.parcel_profile(p,temp1[0,lat,lon],Td1[0,lat,lon]).to('degC')
            cape[lat,lon] = mpcalc.cape_cin(p,temp1[:,lat,lon],Td1[:,lat,lon],prof)[0].magnitude
    t = int((t)/12)
    np.save('/g/data/up6/jl2538/cape_data_hourly/cape-deep-off-'+str(t)+'-single.npy',cape)
    print('done')
    gc.collect()
    f = open("capetime.txt", "w")
    t = t+1
    f.write(str(t))
    f.close()

#np.save('cape30b.npy', cape)
#cape = np.load('cape3single.npy')
#print(np.shape(cape))
#print(cape[1,:])
#cape = np.ma.masked_array(cape,mask).compressed()
#precip1 = np.divide(1000*24*3600*precip[1,:,:].load().data,hours_grouped)
#precip1 = np.ma.masked_array(precip1,mask).compressed()
#plt.scatter(cape,precip1)
#plt.savefig('cape_precip.png',dpi=300)
