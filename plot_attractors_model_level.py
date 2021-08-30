# We will use humidity (or csf), vertical wind as predictors for the change of precip (dP/dt), then construct wolding attractor plots

# This code is take from precentage_blah_blah.py
import netCDF4 as nc4
import xarray as xr
import sys
from netCDF4 import Dataset
from pathlib import Path
import numpy as np
import metpy.calc as mpcalc
from metpy.units import units
from scipy import stats
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn import linear_model, svm
from global_land_mask import globe
from sklearn.tree import plot_tree, DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score, train_test_split,  RepeatedKFold
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
np.seterr(divide='ignore', invalid='ignore')

def csf_calc(t, start_level,ta, hus, mask, pressure_diff, true_pressure_midpoint, hours_grouped, bottom_lower_bound, bottom_upper_bound, middle_lower_bound, middle_upper_bound, top_lower_bound, top_upper_bound):
    temp = np.divide(ta[t,start_level:,:,:].load(),hours_grouped)*units.kelvin
    sat_current = mpcalc.saturation_mixing_ratio(true_pressure_midpoint,temp.data)
    sat_weighted = (sat_current.data * pressure_diff)
    sat_sum = np.sum(sat_weighted,axis=0)
    hus_current = hus[t,start_level:,:,:].load()
    hus_weighted= (hus_current.data * pressure_diff)
    hus_sum = np.sum(hus_weighted,axis=0)
    csf=hus_sum/sat_sum
    csf=np.divide(csf.magnitude,hours_grouped)
    csf_all_ocean = np.ma.masked_array(csf,mask)
    csf_all_ocean = csf_all_ocean.filled(np.nan)

    hus_bottom_sum = np.sum(hus_weighted[bottom_lower_bound:bottom_upper_bound],axis=0)
    sat_bottom_sum = np.sum(sat_weighted[bottom_lower_bound:bottom_upper_bound],axis=0)
    csf_bottom = hus_bottom_sum/sat_bottom_sum
    csf_bottom = np.divide(csf_bottom.magnitude,hours_grouped)
    csf_bottom_ocean = np.ma.masked_array(csf_bottom,mask)

    hus_middle_sum = np.sum(hus_weighted[middle_lower_bound:middle_upper_bound],axis=0)
    sat_middle_sum = np.sum(sat_weighted[middle_lower_bound:middle_upper_bound],axis=0)
    csf_middle = hus_middle_sum/sat_middle_sum
    csf_middle = np.divide(csf_middle.magnitude,hours_grouped)
    csf_middle_ocean = np.ma.masked_array(csf_middle,mask)

    hus_top_sum = np.sum(hus_weighted[top_lower_bound:top_upper_bound],axis=0)
    sat_top_sum = np.sum(sat_weighted[top_lower_bound:top_upper_bound],axis=0)
    csf_top = hus_top_sum/sat_top_sum
    csf_top = np.divide(csf_top.magnitude,hours_grouped)
    csf_top_ocean = np.ma.masked_array(csf_top,mask)

    return csf_all_ocean, csf_bottom_ocean, csf_middle_ocean, csf_top_ocean

def hus_calc(t, start_level,ta, hus, mask, pressure_diff, true_pressure_midpoint, hours_grouped, bottom_lower_bound, bottom_upper_bound, middle_lower_bound, middle_upper_bound, top_lower_bound, top_upper_bound):
    hus_current = hus[t,start_level:,:,:].load()
    hus_weighted= (hus_current.data * pressure_diff)/9.8
    hus_sum = np.sum(hus_weighted,axis=0)
    hus_sum=np.divide(hus_sum.magnitude,hours_grouped)
    hus_all_ocean = np.ma.masked_array(hus_sum,mask)
    hus_all_ocean = hus_all_ocean.filled(np.nan)

    hus_bottom_sum = np.sum(hus_weighted[bottom_lower_bound:bottom_upper_bound],axis=0)
    hus_sum_bottom = np.divide(hus_bottom_sum.magnitude,hours_grouped)
    hus_sum_bottom_ocean = np.ma.masked_array(hus_sum_bottom,mask)

    hus_middle_sum = np.sum(hus_weighted[middle_lower_bound:middle_upper_bound],axis=0)
    hus_sum_middle = np.divide(hus_middle_sum.magnitude,hours_grouped)
    hus_sum_middle_ocean = np.ma.masked_array(hus_sum_middle,mask)

    hus_top_sum = np.sum(hus_weighted[top_lower_bound:top_upper_bound],axis=0)
    hus_sum_top = np.divide(hus_top_sum.magnitude,hours_grouped)
    hus_sum_top_ocean = np.ma.masked_array(hus_sum_top,mask)

    return hus_all_ocean, hus_sum_bottom_ocean, hus_sum_middle_ocean, hus_sum_top_ocean

def sat_calc(t, start_level,ta, hus, mask, pressure_diff, true_pressure_midpoint, hours_grouped, bottom_lower_bound, bottom_upper_bound, middle_lower_bound, middle_upper_bound, top_lower_bound, top_upper_bound):
    temp = np.divide(ta[t,start_level:,:,:].load(),hours_grouped)*units.kelvin
    sat_current = mpcalc.saturation_mixing_ratio(true_pressure_midpoint,temp.data)
    sat_weighted = (sat_current.data * pressure_diff)/9.8
    sat_sum = np.sum(sat_weighted,axis=0)
    sat_all_ocean = np.ma.masked_array(sat_sum,mask)
    sat_all_ocean = sat_all_ocean.filled(np.nan)

    sat_bottom_sum = np.sum(sat_weighted[bottom_lower_bound:bottom_upper_bound],axis=0)
    sat_sum_bottom_ocean = np.ma.masked_array(sat_bottom_sum,mask)

    sat_middle_sum = np.sum(sat_weighted[middle_lower_bound:middle_upper_bound],axis=0)
    sat_sum_middle_ocean = np.ma.masked_array(sat_middle_sum,mask)

    sat_top_sum = np.sum(sat_weighted[top_lower_bound:top_upper_bound],axis=0)
    sat_sum_top_ocean = np.ma.masked_array(sat_top_sum,mask)

    return sat_all_ocean, sat_sum_bottom_ocean, sat_sum_middle_ocean, sat_sum_top_ocean

def satur(pressure,temperature):
    func = mpcalc.saturation_mixing_ratio(pressure,temperature)
    return xr.apply_ufunc(func,pressure,temperature)

def inputNumber(message):
  while True:
    try:
       userInput = int(input(message))       
    except ValueError:
       print("Not an integer! Try again.")
       continue
    else:
       return userInput 
       break


def load_data(mode,rain_setting):
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

    # hdsettings
    #mode = inputNumber('Please enter setting for scheme:')
    #rain_setting = inputNumber('Please enter setting for rain:')
    ## Daily Control
    if mode == 1:
        l_prect=[]
        l_prect.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/timselsum-daily.output.PRECT.nc')

        l_preccdzm=[]
        l_preccdzm.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/timselsum-daily.output.PRECCDZM.nc')

        l_precc=[]
        l_precc.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/timselsum-daily.output.PRECC.nc')

        l_precl=[]
        l_precl.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/timselsum-daily.output.PRECL.nc')

        l_precsc=[]
        l_precsc.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/timselsum-daily.output.PRECSC.nc')


        l_ta=[]
        l_ta.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/timselsum-daily.output.T.nc')

        l_ps=[]
        l_ps.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/timselsum-daily.output.PS.nc')

        l_hus=[]
        l_hus.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/timselsum-daily.output.Q.nc')

        l_u=[]
        l_u.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/pressure-levels/timselsum-daily.output.U.nc')

        l_v=[]
        l_v.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/pressure-levels/timselsum-daily.output.V.nc')

        l_omega=[]
        l_omega.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/pressure-levels/timselsum-daily.output.OMEGA.nc')

        #l_prect=[]
        #l_prect.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/output.PRECT.nc')

        #l_preccdzm=[]
        #l_preccdzm.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/output.PRECCDZM.nc')

        #l_precc=[]
        #l_precc.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/output.PRECC.nc')

        #l_precl=[]
        #l_precl.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/output.PRECL.nc')

        #l_precsc=[]
        #l_precsc.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/output.PRECSC.nc')

        #l_ta=[]
        #l_ta.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/output.T.nc')

        #l_hus=[]
        #l_hus.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/output.Q.nc')

        #l_omega=[]
        #l_omega.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--ctrl-daily-output/output.OMEGA.nc')


    ## Shallow Off
    if mode == 2:
        l_prect=[]
        l_prect.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/timselsum-daily.output.PRECT.nc')

        l_preccdzm=[]
        l_preccdzm.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/timselsum-daily.output.PRECCDZM.nc')

        l_precc=[]
        l_precc.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/timselsum-daily.output.PRECC.nc')

        l_precl=[]
        l_precl.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/timselsum-daily.output.PRECL.nc')

        l_precsc=[]
        l_precsc.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/timselsum-daily.output.PRECSC.nc')

        l_ta=[]
        l_ta.append('/g/data1b/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/timselsum-daily.output.T.nc')

        l_hus=[]
        l_hus.append('/g/data1b/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/timselsum-daily.output.Q.nc')

        l_ps=[]
        l_ps.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/timselsum-daily.output.PS.nc')

        l_u=[]
        l_u.append('/g/data1b/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/pressure-levels/timselsum-daily.output.U.nc')

        l_v=[]
        l_v.append('/g/data1b/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/pressure-levels/timselsum-daily.output.V.nc')

        l_omega=[]
        l_omega.append('/g/data1b/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/pressure-levels/timselsum-daily.output.OMEGA.nc')

        #l_prect=[]
        #l_prect.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/output.PRECT.nc')

        #l_preccdzm=[]
        #l_preccdzm.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/output.PRECCDZM.nc')

        #l_precc=[]
        #l_precc.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/output.PRECC.nc')

        #l_precl=[]
        #l_precl.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/output.PRECL.nc')

        #l_precsc=[]
        #l_precsc.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/output.PRECSC.nc')

        #l_ta=[]
        #l_ta.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/output.T.nc')

        #l_hus=[]
        #l_hus.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/output.Q.nc')

        #l_omega=[]
        #l_omega.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--shallow-off-daily-output/output.OMEGA.nc')


    # Deep conv off
    if mode == 3:
        l_prect=[]
        l_prect.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--deep-off--daily-output/timselsum-daily.output.PRECT.nc')

        l_preccdzm=[]
        l_preccdzm.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--deep-off--daily-output/timselsum-daily.output.PRECC.nc')

        l_precc=[]
        l_precc.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--deep-off--daily-output/timselsum-daily.output.PRECC.nc')

        l_precl=[]
        l_precl.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--deep-off--daily-output/timselsum-daily.output.PRECL.nc')

        l_precsc=[]
        l_precsc.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--deep-off--daily-output/timselsum-daily.output.PRECSC.nc')

        l_ta=[]
        l_ta.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--deep-off--daily-output/timselsum-daily.output.T.nc')

        l_hus=[]
        l_hus.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--deep-off--daily-output/timselsum-daily.output.Q.nc')

        l_ps=[]
        l_ps.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--deep-off--daily-output/timselsum-daily.output.PS.nc')
        l_u=[]
        l_u.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--deep-off--daily-output/timselsum-daily.output.U.nc')

        l_v=[]
        l_v.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--deep-off--daily-output/timselsum-daily.output.V.nc')

        l_omega=[]
        l_omega.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--deep-off--daily-output/pressure-levels/timselsum-daily.output.OMEGA.nc')

        #l_prect=[]
        #l_prect.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--deep-off--daily-output/output.PRECT.nc')

        #l_preccdzm=[]
        #l_preccdzm.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--deep-off--daily-output/output.PRECCDZM.nc')

        #l_precc=[]
        #l_precc.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--deep-off--daily-output/output.PRECC.nc')

        #l_precl=[]
        #l_precl.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--deep-off--daily-output/output.PRECL.nc')

        #l_precsc=[]
        #l_precsc.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--deep-off--daily-output/output.PRECSC.nc')

        #l_ta=[]
        #l_ta.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--deep-off--daily-output/output.T.nc')

        #l_hus=[]
        #l_hus.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--deep-off--daily-output/output.Q.nc')

        #l_omega=[]
        #l_omega.append('/g/data/w42/daf561/cesm/1.0.6/postprocess/rF_AMIP_CN_CAM4--deep-off--daily-output/output.OMEGA.nc')


    hours_grouped=24

    print('yes')
    #ds_omega=xr.open_mfdataset(l_omega,combine='by_coords')
    #ds_u=xr.open_mfdataset(l_u,combine='by_coords')
    #ds_v=xr.open_mfdataset(l_v,combine='by_coords')
    ds_hus=xr.open_mfdataset(l_hus,combine='by_coords')
    ds_pr = xr.open_mfdataset(l_prect,combine='by_coords')

    #ds_preccdzm = xr.open_mfdataset(l_preccdzm,combine='by_coords')
    ds_precc = xr.open_mfdataset(l_precc,combine='by_coords')
    ds_precl = xr.open_mfdataset(l_precl,combine='by_coords')
    ds_precsc = xr.open_mfdataset(l_precsc,combine='by_coords')

    ds_ta = xr.open_mfdataset(l_ta,combine='by_coords')
    top_lat = 15
    bot_lat = -15
    lon_bot = 0
    lon_top = 360
    hus = ds_hus['Q'].sel(lat = slice(bot_lat,top_lat), lon = slice(lon_bot,lon_top))
    ta = ds_ta['T'].sel(lat = slice(bot_lat,top_lat), lon = slice(lon_bot,lon_top))

    if rain_setting == 1:
        rain_setting = 'Total'
        precip = ds_pr['PRECT'].sel(lat = slice(bot_lat,top_lat), lon = slice(lon_bot,lon_top))
    elif rain_setting == 2:
        rain_setting = 'Convective'
        precip = ds_precc['PRECC'].sel(lat = slice(bot_lat,top_lat), lon = slice(lon_bot,lon_top))
    elif rain_setting == 3:
        rain_setting = 'Large'
        precip = ds_precl['PRECL'].sel(lat = slice(bot_lat,top_lat), lon = slice(lon_bot,lon_top))

    #if mode == 3:
    #    precip_cdzm = ds_preccdzm['PRECC'].sel(lat = slice(bot_lat,top_lat), lon = slice(lon_bot,lon_top))
    #else:
    #    precip_cdzm = ds_preccdzm['PRECCDZM'].sel(lat = slice(bot_lat,top_lat), lon = slice(lon_bot,lon_top))
        #precip = ds_preccdzm['PRECCDZM'].sel(lat = slice(bot_lat,top_lat))
    precip_c = ds_precc['PRECC'].sel(lat = slice(bot_lat,top_lat), lon = slice(lon_bot,lon_top))
    precip_sc = ds_precsc['PRECSC'].sel(lat = slice(bot_lat,top_lat), lon = slice(lon_bot,lon_top))
    if mode == 1:
        convection_setting = 'Control'
    elif mode == 2:
        convection_setting = 'Shallow-Off'
    elif mode == 3:
        convection_setting = 'Deep-Off'
     
    #omega = ds_omega['OMEGA'].sel(lat = slice(bot_lat,top_lat), lon = slice(lon_bot,lon_top)) 
    lat_data =ds_hus['lat'].sel(lat = slice(bot_lat,top_lat)) 
    lon_data =ds_hus['lon'].sel(lon = slice(lon_bot,lon_top))
    print(lat_data)
    print(lon_data)
    lon_length = len(lon_data)
    lat_length = len(lat_data)
    print(lon_length)
    print(len(lat_data))
    #print(lat.load())
    #lon_grid, lat_grid = np.meshgrid(np.add(lon.load(),-180),lat.load())
    #print(lon.load())
    array_360 = np.zeros(shape=np.shape(lon_data.load().data)) + 360
    lon = np.mod(lon_data + 180, array_360) - 180
    #print(lon)
    lon_grid, lat_grid = np.meshgrid(lon,lat_data.load())
    ## USE THIS TO USE ONLY OCEAN POINTS
    mask = globe.is_land(lat_grid, lon_grid)

    ## USE THIS TO USE ONLY LAND POINTS
    #mask = globe.is_ocean(lat_grid, lon_grid)
    #np.savetxt("mask.csv", mask, delimiter=",")

    true_pressure_midpoint = np.load('true_pressure_midpoint.npy')
    true_pressure_midpoint = units.pascal*true_pressure_midpoint/24
    true_pressure_interface = np.load('true_pressure_interface.npy')
    #print(np.shape(true_pressure_interface))
    pressure_diff = np.diff(true_pressure_interface,axis = 0)/24*units.pascal
    #pr = np.flip(pr,axis=0)
    #pr1 = true_pressure_interface
    #pr1 = true_pressure_midpoint
    #print(pr1)
    start_level = 11
    true_pressure_midpoint = true_pressure_midpoint[start_level:]
    pressure_diff = pressure_diff[start_level:]
    true_pressure_midpoint = true_pressure_midpoint[:,:,:]
    pressure_diff = pressure_diff[:,:,:]
    print(np.shape(pressure_diff))
    print(np.shape(true_pressure_midpoint))
    m = true_pressure_midpoint 
    m = np.mean(m,axis=1)
    m = np.mean(m,axis=1)

    m = pressure_diff
    m = np.mean(m,axis=1)
    m = np.mean(m,axis=1)
    #print(np.shape(pr1))
    bins_csf= 15
    bins_precip = 20
    precip_lower = 0
    csf_lower = 0.3
    csf_higher = 1
    if mode == 3:
        precip_higher = 200
    else:
        precip_higher = 100
    #csf_binvalues = np.arange(csf_lower+csf_higher/bins_csf,csf_higher,(csf_higher-csf_lower)/bins_csf)
    #precip_binvalues = np.arange(precip_lower+precip_higher/bins_precip,precip_higher,(precip_higher-precip_lower)/bins_precip)

    bin_csf_size = (csf_higher-csf_lower)/(bins_csf-1)
    csf_binvalues = np.arange(csf_lower+bin_csf_size/2,csf_higher,bin_csf_size)
    arrow_csf_centre = np.arange(csf_lower,csf_higher+bin_csf_size/2,bin_csf_size)

    bin_precip_size = precip_lower+(precip_higher-precip_lower)/(bins_precip-1)
    precip_binvalues = np.arange(precip_lower+bin_precip_size/2,precip_higher,bin_precip_size)
    arrow_precip_centre = np.arange(precip_lower,precip_higher+bin_precip_size/2,bin_precip_size)

    #print(csf_binvalues)
    #print(len(csf_binvalues))
    #print(arrow_csf_centre)
    #print(len(arrow_csf_centre))
    #print(precip_binvalues)
    #print(len(precip_binvalues))
    #print(arrow_precip_centre)
    #print(len(arrow_precip_centre))
    #precip_binvalues = np.arange(precip_lower+precip_higher/bins_precip,precip_higher,(precip_higher-precip_lower)/bins_precip)
    temp_ocean = np.zeros(shape=(731,17,1))
    hus_ocean = np.zeros(shape=(731,17,1))
    #print(np.sum(mask))
    #sys.exit()

    bins_fine = 20

    bin_csf_size_fine = (csf_higher-csf_lower)/(bins_fine-1)
    csf_binvalues_fine = np.arange(csf_lower+bin_csf_size_fine/2,csf_higher,bin_csf_size_fine)
    arrow_csf_centre_fine = np.arange(csf_lower,csf_higher+bin_csf_size_fine/2,bin_csf_size_fine)

    bin_precip_size_fine = (precip_higher-precip_lower)/(bins_fine-1)
    precip_binvalues_fine = np.arange(precip_lower+bin_precip_size_fine/2,precip_higher,bin_precip_size_fine)
    arrow_precip_centre_fine = np.arange(precip_lower,precip_higher+bin_precip_size_fine/2,bin_precip_size_fine)

    #print(csf_binvalues_fine)
    #print(len(csf_binvalues_fine))
    #print(arrow_csf_centre_fine)
    #print(len(arrow_csf_centre_fine))
    #print(precip_binvalues_fine)
    #print(len(precip_binvalues_fine))
    #print(arrow_precip_centre_fine)
    #print(len(arrow_precip_centre_fine))

    bottom_upper_bound = 25 - start_level
    bottom_lower_bound = 23 - start_level
    middle_upper_bound = 22 - start_level
    middle_lower_bound = 20 - start_level
    top_upper_bound = 19 - start_level
    top_lower_bound = 16 - start_level
    flag=0
    start_level = 11
    lag = 1
    
    storm_precip_list = []
    storm_csf_list = []
    for level_setting in range(1,3):

        counts1=np.zeros(shape=(bins_csf,bins_precip))
        counts2=np.zeros(shape=(bins_csf,bins_precip))
        counts3=np.zeros(shape=(bins_csf,bins_precip))
        
        counts2_fine=np.zeros(shape=(bins_fine,bins_fine))
        precip_positive_tendency_count = np.zeros(shape=(bins_fine,bins_fine))
        csf_positive_tendency_count = np.zeros(shape=(bins_fine,bins_fine))
        temp_count = np.zeros(shape=(bins_fine,bins_fine))

        arrow_dirs = np.zeros(shape=(bins_csf,bins_precip,2))
        arrows = np.zeros(shape=(bins_csf,bins_precip,2))
        #arrow_dirs = np.zeros(shape=(lat_length,lon_length,2))
        bin_loc1=np.zeros(shape=(lat_length,lon_length,2),dtype=int)
        bin_loc2=np.zeros(shape=(lat_length,lon_length,2),dtype=int)
        bin_loc3=np.zeros(shape=(lat_length,lon_length,2),dtype=int)

        bin_loc2_fine=np.zeros(shape=(lat_length,lon_length,2),dtype=int)
        # Lat-Lon space, 2hr-1hr, for the arrows direction
        # then sum these by their bins at 1hr for their arrow location
        # Copied from predtowold
        if level_setting == 0:
            level = 'All'
        elif level_setting==1:
            level = 'Bottom'
        elif level_setting == 2:
            level = 'Middle'
        elif level_setting == 3:
            level = 'Top'

        for t in tqdm(range(30,360)):
            csf1, csf1_bottom, csf1_middle, csf1_top = csf_calc(t-lag, start_level,ta, hus, mask, pressure_diff, true_pressure_midpoint, hours_grouped, bottom_lower_bound, bottom_upper_bound, middle_lower_bound, middle_upper_bound, top_lower_bound, top_upper_bound)
            csf2, csf2_bottom, csf2_middle, csf2_top = csf_calc(t, start_level,ta, hus, mask, pressure_diff, true_pressure_midpoint, hours_grouped, bottom_lower_bound, bottom_upper_bound, middle_lower_bound, middle_upper_bound, top_lower_bound, top_upper_bound)

            csf3, csf3_bottom, csf3_middle, csf3_top = csf_calc(t+lag, start_level,ta, hus, mask, pressure_diff, true_pressure_midpoint, hours_grouped, bottom_lower_bound, bottom_upper_bound, middle_lower_bound, middle_upper_bound, top_lower_bound, top_upper_bound)

            #csf1, csf1_bottom, csf1_middle, csf1_top = hus_calc(t-lag, start_level,ta, hus, mask, pressure_diff, true_pressure_midpoint, hours_grouped, bottom_lower_bound, bottom_upper_bound, middle_lower_bound, middle_upper_bound, top_lower_bound, top_upper_bound)
            #csf2, csf2_bottom, csf2_middle, csf2_top = hus_calc(t, start_level,ta, hus, mask, pressure_diff, true_pressure_midpoint, hours_grouped, bottom_lower_bound, bottom_upper_bound, middle_lower_bound, middle_upper_bound, top_lower_bound, top_upper_bound)
            #csf3, csf3_bottom, csf3_middle, csf3_top = hus_calc(t+lag, start_level,ta, hus, mask, pressure_diff, true_pressure_midpoint, hours_grouped, bottom_lower_bound, bottom_upper_bound, middle_lower_bound, middle_upper_bound, top_lower_bound, top_upper_bound)

            #csf1, csf1_bottom, csf1_middle, csf1_top = sat_calc(t-lag, start_level,ta, hus, mask, pressure_diff, true_pressure_midpoint, hours_grouped, bottom_lower_bound, bottom_upper_bound, middle_lower_bound, middle_upper_bound, top_lower_bound, top_upper_bound)
            #csf2, csf2_bottom, csf2_middle, csf2_top = sat_calc(t, start_level,ta, hus, mask, pressure_diff, true_pressure_midpoint, hours_grouped, bottom_lower_bound, bottom_upper_bound, middle_lower_bound, middle_upper_bound, top_lower_bound, top_upper_bound)
            #csf3, csf3_bottom, csf3_middle, csf3_top = sat_calc(t+lag, start_level,ta, hus, mask, pressure_diff, true_pressure_midpoint, hours_grouped, bottom_lower_bound, bottom_upper_bound, middle_lower_bound, middle_upper_bound, top_lower_bound, top_upper_bound)

            if level_setting == 0:
                level = 'All'
            elif level_setting==1:
                level = 'Bottom'
                csf1 = csf1_bottom
                csf2 = csf2_bottom
                csf3 = csf3_bottom
            elif level_setting == 2:
                level = 'Middle'
                csf1 = csf1_middle
                csf2 = csf2_middle
                csf3 = csf3_middle
            elif level_setting == 3:
                level = 'Top'
                csf1 = csf1_top
                csf2 = csf2_top
                csf3 = csf3_top

            # Data
            temp2 = np.divide(ta[t,11:,:,:].load().data,hours_grouped)
            temp2 = (temp2 * pressure_diff)
            temp2 = np.sum(temp2,axis=0)
            temp2 = temp2.data
            #print(np.nanmean(temp2))
            #print(np.shape(temp2))

            precip1 = np.divide(1000*24*3600*precip[t-lag,:,:].load().data,hours_grouped)
            precip2 = np.divide(1000*24*3600*precip[t,:,:].load().data,hours_grouped)
            precip3 = np.divide(1000*24*3600*precip[t+lag,:,:].load().data,hours_grouped)
            precip1 = np.ma.masked_array(precip1,mask)
            precip2 = np.ma.masked_array(precip2,mask)
            precip3 = np.ma.masked_array(precip3,mask)

            #print(np.nanmean(arrow))
            #print(np.nanmean(precip3-precip1))
            arrow_csf = (csf3-csf1)
            arrow_precip = precip3-precip1
            count = 0
            for lon in range(0,lon_length):
                for lat in range(0,lat_length):

                    if np.isnan(csf1[lat,lon]) == 0:
                        count = count+1
                        #lat_current = lat_data[lat].data
                        #lon_current = (lon_data[lon].data+180)%360-180
                        #is_on_land = globe.is_land(lat_current, lon_current)
                        #if is_on_land == False:
                        #    print('LAND')
                        #    print('lat={}, lon={} is on land: {}'.format(lat_current.data,lon_current.data,is_on_land))
                        #    print(precip[t,lat,lon])
                        #    sys.exit()
                        if (csf2[lat,lon] - precip2[lat,lon]/100 < 0.4) and csf2[lat,lon] > 0.7 and csf2[lat,lon] < 0.8:
                            storm_precip_list.append(arrow_precip[lat,lon])
                            storm_csf_list.append(arrow_csf[lat,lon])

                        bin_loc1[lat,lon,0]=int(np.digitize(csf1[lat,lon],csf_binvalues))
                        bin_loc1[lat,lon,1]=int(np.digitize(precip1[lat,lon],precip_binvalues))
                        bin_loc3[lat,lon,0]=int(np.digitize(csf3[lat,lon],csf_binvalues))
                        bin_loc3[lat,lon,1]=int(np.digitize(precip3[lat,lon],precip_binvalues))
                        bin_loc2[lat,lon,0]=int(np.digitize(csf2[lat,lon],csf_binvalues))
                        bin_loc2[lat,lon,1]=int(np.digitize(precip2[lat,lon],precip_binvalues))
                        arrow_dirs[bin_loc2[lat,lon,0],bin_loc2[lat,lon,1],0]=arrow_csf[lat,lon]+arrow_dirs[bin_loc2[lat,lon,0],bin_loc2[lat,lon,1],0]
                        arrow_dirs[bin_loc2[lat,lon,0],bin_loc2[lat,lon,1],1]=arrow_precip[lat,lon]+arrow_dirs[bin_loc2[lat,lon,0],bin_loc2[lat,lon,1],1]
                        counts1[bin_loc1[lat,lon,0],bin_loc1[lat,lon,1]]=counts1[bin_loc1[lat,lon,0],bin_loc1[lat,lon,1]]+1
                        counts2[bin_loc2[lat,lon,0],bin_loc2[lat,lon,1]]=counts2[bin_loc2[lat,lon,0],bin_loc2[lat,lon,1]]+1
                        counts3[bin_loc3[lat,lon,0],bin_loc3[lat,lon,1]]=counts3[bin_loc3[lat,lon,0],bin_loc3[lat,lon,1]]+1
                   
                        bin_loc2_fine[lat,lon,0]=int(np.digitize(csf2[lat,lon],csf_binvalues_fine))
                        bin_loc2_fine[lat,lon,1]=int(np.digitize(precip2[lat,lon],precip_binvalues_fine))

                        counts2_fine[bin_loc2_fine[lat,lon,0],bin_loc2_fine[lat,lon,1]]=counts2_fine[bin_loc2_fine[lat,lon,0],bin_loc2_fine[lat,lon,1]]+1
                        temp_count[bin_loc2_fine[lat,lon,0],bin_loc2_fine[lat,lon,1]]=temp_count[bin_loc2_fine[lat,lon,0],bin_loc2_fine[lat,lon,1]]+temp2[lat,lon]
                        if arrow_precip[lat,lon] > 0:
                            precip_positive_tendency_count[bin_loc2_fine[lat,lon,0],bin_loc2_fine[lat,lon,1]] = precip_positive_tendency_count[bin_loc2_fine[lat,lon,0],bin_loc2_fine[lat,lon,1]] + 1

                        if arrow_csf[lat,lon] > 0:
                            csf_positive_tendency_count[bin_loc2_fine[lat,lon,0],bin_loc2_fine[lat,lon,1]] = csf_positive_tendency_count[bin_loc2_fine[lat,lon,0],bin_loc2_fine[lat,lon,1]] + 1
        arrow_dirs[:,:,0]=np.divide(arrow_dirs[:,:,0],counts2[:,:])
        arrow_dirs[:,:,1]=np.divide(arrow_dirs[:,:,1],counts2[:,:])
        csf_positive_tendency_fraction = np.divide(csf_positive_tendency_count, counts2_fine[:,:])
        precip_positive_tendency_fraction = np.divide(precip_positive_tendency_count, counts2_fine[:,:])
        temp_count = np.divide(temp_count, counts2_fine[:,:])
        ##print(np.nanmean(arrows[:,:,0]))
        ##print(np.nanmean(arrows[:,:,1])) 
        ##print(np.sum(counts2))
        ##print(np.max(hus2))
        ##print(np.min(hus2))
        for i in range(0,bins_csf):
            for j in range(0,bins_precip):
                if counts2[i,j]<100:
                    arrow_dirs[i,j,:]=0

        #for i in range(0,bins_fine):
        #    for j in range(0,bins_fine):
        #        if counts2_fine[i,j]<5:
        #            csf_positive_tendency_fraction[i,j] = float("nan")
        #            precip_positive_tendency_fraction[i,j] = float("nan")
        #            temp_count[i,j] = float("nan")
        #            counts2_fine[i,j] = float("nan")

        #arrow_dirs[15,15,0]=0.5
        #arrow_dirs[15,15,1]=20
        np.savetxt("counts2-{}-{}-{}.csv".format(convection_setting, level, rain_setting), counts2, delimiter=",")
        np.savetxt("counts2_fine-{}-{}-{}.csv".format(convection_setting, level, rain_setting), counts2_fine, delimiter=",")
        np.savetxt("csf_arrows-{}-{}-{}.csv".format(convection_setting, level, rain_setting), arrow_dirs[:,:,0], delimiter=",")
        np.savetxt("precip_arrows-{}-{}-{}.csv".format(convection_setting, level, rain_setting), arrow_dirs[:,:,1], delimiter=",")

        np.save("paper_plot_data/counts2-{}-{}-{}".format(convection_setting, level, rain_setting), counts2)
        np.save("paper_plot_data/counts2_fine-{}-{}-{}".format(convection_setting, level, rain_setting), counts2_fine)
        np.save("paper_plot_data/csf_arrows-{}-{}-{}".format(convection_setting, level, rain_setting), arrow_dirs[:,:,0])
        np.save("paper_plot_data/precip_arrows-{}-{}-{}".format(convection_setting, level, rain_setting), arrow_dirs[:,:,1])
 
        counts2 = np.log10(np.divide(counts2,np.nansum(counts2)))
        counts2_fine = np.log10(100*np.divide(counts2_fine,np.nansum(counts2_fine)))
        print(np.nanmean(storm_csf_list))
        print(np.nanmean(storm_precip_list))
        plt.hist(storm_csf_list,bins=50)
        plt.savefig('storm_csf',dpi=300)
        plt.close()
        plt.hist(storm_precip_list,bins=50)
        plt.savefig('storm_precip',dpi=300)
        plt.close()
        # Define colormap # 
        X,Y=np.meshgrid(arrow_csf_centre,arrow_precip_centre)
        X_fine,Y_fine=np.meshgrid(arrow_csf_centre_fine,arrow_precip_centre_fine)
        contour_levels = np.linspace(-3,1.5,30)
        CS = plt.contourf(X_fine,Y_fine,np.transpose(counts2_fine),30,levels=contour_levels)
        #c = plt.contourf(X_fine,Y_fine,np.transpose(csf_positive_tendency_fraction),20,levels=np.arange(0, 1.01, .01),cmap=colormap_colors, vmin=0.0, vmax=1.0)
        #c = plt.contourf(X_fine,Y_fine,np.transpose(temp_count),20,levels=np.arange(0, 1.01, .01),cmap=colormap_colors, vmin=0.0, vmax=1.0)
        
        #plt.quiver(X,Y,np.transpose(arrow_dirs[:,:,0]),np.transpose(arrow_dirs[:,:,1]),width=0.007, angles='xy', scale_units='xy', scale=1, pivot='mid',color='r')
        colorbar = plt.colorbar(CS, label = 'Log10(Percent of Total Observations)',orientation='horizontal', aspect=35, ticks=[-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5])
        colorbar.ax.set_xticklabels(['-3','-2.5','-2','-1.5','-1','-0.5','0','0.5','1','1.5'])
        plt.locator_params(axis='y', nbins=5)
        plt.locator_params(axis='x', nbins=5)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=14)
        fontsize_set=14
        plt.xlabel('CSF', fontsize=fontsize_set)
        #plt.xlabel('Precipitable Water [kg m^-2]')
        plt.ylabel('Precipitation Rate [mm day^-1]', fontsize=fontsize_set)
        plt.quiver(X,Y,np.transpose(arrow_dirs[:,:,0]),np.transpose(arrow_dirs[:,:,1]), angles='xy', scale_units='xy', scale=1, pivot='mid', color='crimson',clip_on=False)
        #plt.title(convection_setting+ ', '+rain_setting+', '+ level)
        #plt.savefig('paper_plots/CAM4-{}-{}-{}-00-12-Attractor'.format(convection_setting, level, rain_setting),dpi=300)
        plt.savefig('test',dpi=300)
        plt.close()
        print('CAM4-'+convection_setting+'-'+ level +'-'+ rain_setting+ '-Attractor')
        print(count)
        #sys.exit()
    return

def main():

    load_data(1,1)
    #load_data(1,2)
    #load_data(1,3)

    load_data(2,1)
    #load_data(2,2)
    #load_data(2,3)

    load_data(3,1)
    #load_data(3,2)
    #load_data(3,3)

if __name__ == "__main__":
    main()
