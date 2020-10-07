# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 20:06:53 2020

@author: Bgard
"""
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pyart
import math


import warnings
warnings.filterwarnings("ignore")

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

Figsize = (18, 20)

def plot_radar_ppi_panel(radar, radar_DT, sweep_num, png_file):

    fig = plt.figure(figsize=Figsize)
    
    radar_lon = radar.longitude['data'][:][0]
    radar_lat = radar.latitude['data'][:][0]


# Setting projection and ploting the second tilt

    projection = ccrs.LambertConformal(central_latitude=radar.latitude['data'][0],
                                       central_longitude=radar.longitude['data'][0])

    display = pyart.graph.RadarMapDisplay(radar)
#
# *** Calculate bounding limits for map
#
    dtor = math.pi/180.0
    max_range=150.0
    maxrange_meters = max_range * 1000.
    meters_to_lat = 1. / 111177.
    meters_to_lon =  1. / (111177. * math.cos(radar_lat * dtor))

    min_lat = radar_lat - maxrange_meters * meters_to_lat 
    max_lat = radar_lat + maxrange_meters * meters_to_lat 

    min_lon = radar_lon - maxrange_meters * meters_to_lon 
    max_lon = radar_lon + maxrange_meters * meters_to_lon 

    Pad_lon = -75.471
    Pad_lat =  37.934
    
    PCMK_lon = -75.515
    PCMK_lat =  38.078


    lon_grid = np.arange(-77,-73, 0.5)
    lat_grid = np.arange( 37, 40, 0.5)
#    
#     Fields = ['DBZ2', 'VEL2','ZDR2','RHOHV2','PHIDP2','KDP2']
#     Cmaps  = ['pyart_NWSRef','pyart_NWSVel','pyart_RefDiff', 
#               'pyart_Wild25','pyart_NWSVel','pyart_NWSRef']
#     Vmins   = [0, -20, -1, 0,    0, -1]
#     Vmaxs   = [65, 20, 4,  1,  180,  3]
#     CBarLabels = ['Reflectivity [dBZ]',
#                   'Velocity [m/s]',
#                   'Differential Reflectivity [dB]', 
#                   'Correlation Coefficient',
#                   'Differential Phase [deg]',
#                   'Specific Differential Phase [deg/km]']
#     NFields = len(Fields)   
    
# Extract date/time info from radar_DT as 0-padded strings

    year  = str(radar_DT.year).zfill(4)
    month = str(radar_DT.month).zfill(2)
    day   = str(radar_DT.day).zfill(2)
    hour  = str(radar_DT.hour).zfill(2)
    mint  = str(radar_DT.minute).zfill(2)
    sec   = str(radar_DT.second).zfill(2)

    time_str = month + '/' + day + '/' + year + ' @ ' + hour + ':' + mint + ':' + sec + ' UTC'
    file_time = year + '_' + month + day + '_' + hour + mint + sec
#
# *** Set up figure
#
   # Reflectivity

    ax1 = fig.add_subplot(321,projection=projection)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')

    display.plot_ppi_map('DBZ2', 1, vmin=0, vmax=65,resolution='10m', 
                         projection=projection,
                         cmap='pyart_NWSRef',colorbar_label='DZ [dBZ]',
                         title= 'Reflectivity [dBZ]',
                         min_lon=min_lon, max_lon=max_lon, 
                         min_lat=min_lat, max_lat=max_lat,
                         lon_lines=lon_grid,lat_lines=lat_grid,
                         lat_0=radar.latitude['data'][0],
                         lon_0=radar.longitude['data'][0])

    # Annotate plot
    for rng in range(50,200,50):
        display.plot_range_ring(rng, line_style='k--', lw=0.5)

    for azi in range(0,360,30):
        azimuth = 90. - azi
        dazimuth = azimuth * dtor
        lon_maxrange = radar_lon + math.cos(dazimuth) * meters_to_lon * maxrange_meters
        lat_maxrange = radar_lat + math.sin(dazimuth) * meters_to_lat * maxrange_meters
        display.plot_line_geo([radar_lon, lon_maxrange], [radar_lat, lat_maxrange], 
                              line_style='k--',lw=0.5)
    display.plot_cross_hair(10,npts=100)
    display.plot_point(Pad_lon, Pad_lat, symbol = 'kv', markersize=5)
    display.plot_point(PCMK_lon, PCMK_lat, symbol = 'kv', markersize=5)


    # Radial Velocity
    ax2 = fig.add_subplot(322,projection=projection)
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')

    display.plot_ppi_map('VEL2', 1,resolution='10m', 
                         projection=projection,
                         cmap='pyart_NWSVel',colorbar_label='VR [m/s]',
                         title='Radial Velocity [m/s]',
                         min_lon=min_lon, max_lon=max_lon, 
                         min_lat=min_lat, max_lat=max_lat,
                         lon_lines=lon_grid,lat_lines=lat_grid,
                         lat_0=radar.latitude['data'][0],
                         lon_0=radar.longitude['data'][0])
    display.plot_cross_hair(10,npts=100)
    display.plot_point(Pad_lon, Pad_lat, symbol = 'kv', markersize=5)
    display.plot_point(PCMK_lon, PCMK_lat, symbol = 'kv', markersize=5)
    for rng in range(50,200,50):
        display.plot_range_ring(rng, ls='dashed', lw=0.5)

# Correlation

    ax3 = fig.add_subplot(323,projection=projection)
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')

    display.plot_ppi_map('RHOHV2', 1,resolution='10m', vmin=0.5, vmax=1,
                         cmap='pyart_Wild25',colorbar_label='RH',
                         title='Correlation Coefficient',
                         min_lon=min_lon, max_lon=max_lon, 
                         min_lat=min_lat, max_lat=max_lat,
                         lon_lines=lon_grid,lat_lines=lat_grid,
                         fig=fig,  projection=projection,
                         lat_0=radar.latitude['data'][0],
                         lon_0=radar.longitude['data'][0])
    display.plot_cross_hair(10,npts=100)
    display.plot_point(Pad_lon, Pad_lat, symbol = 'kv', markersize=5)
    display.plot_point(PCMK_lon, PCMK_lat, symbol = 'kv', markersize=5)
    for rng in range(50,200,50):
        display.plot_range_ring(rng, ls='dashed', lw=0.5)

# Differential Phase

    ax4 = fig.add_subplot(324,projection=projection)
    ax4.set_xlabel('Longitude')
    ax4.set_ylabel('Latitude')

    display.plot_ppi_map('ZDR2', 1,resolution='10m', vmin=-1, vmax=4,
                         cmap='pyart_RefDiff',colorbar_label='DR [dB]',
                         title='Differential Reflectivity [dB]',
                         min_lon=min_lon, max_lon=max_lon, 
                         min_lat=min_lat, max_lat=max_lat,
                         lon_lines=lon_grid,lat_lines=lat_grid,
                         fig=fig,  projection=projection,
                         lat_0=radar.latitude['data'][0],
                         lon_0=radar.longitude['data'][0])
    display.plot_cross_hair(10,npts=100)
    display.plot_point(Pad_lon, Pad_lat, symbol = 'kv', markersize=5)
    display.plot_point(PCMK_lon, PCMK_lat, symbol = 'kv', markersize=5)
    for rng in range(50,200,50):
        display.plot_range_ring(rng, ls='dashed', lw=0.5)


# Differential Phase

    ax5 = fig.add_subplot(325,projection=projection)
    ax5.set_xlabel('Longitude')
    ax5.set_ylabel('Latitude')

    display.plot_ppi_map('PHIDP2', 1,resolution='10m', vmin=0, vmax=360,
                         cmap='pyart_NWSRef',colorbar_label='PH [deg]',
                         title='Differential Phase [deg]',
                         min_lon=min_lon, max_lon=max_lon, 
                         min_lat=min_lat, max_lat=max_lat,
                         lon_lines=lon_grid,lat_lines=lat_grid,
                         fig=fig,  projection=projection,
                         lat_0=radar.latitude['data'][0],
                         lon_0=radar.longitude['data'][0])
    display.plot_cross_hair(10,npts=100)
    display.plot_point(Pad_lon, Pad_lat, symbol = 'kv', markersize=5)
    display.plot_point(PCMK_lon, PCMK_lat, symbol = 'kv', markersize=5)
    for rng in range(50,200,50):
        display.plot_range_ring(rng, ls='dashed', lw=0.5)

# Specific Differential Phase

    ax6 = fig.add_subplot(326,projection=projection)
    ax6.set_xlabel('Longitude')
    ax6.set_ylabel('Latitude')

    display.plot_ppi_map('KDP2', 1,resolution='10m', vmin=-1, vmax=3,
                         cmap='pyart_NWSRef',colorbar_label='KD [deg/km]',
                         title='Specific Differential Phase [deg/km]',
                         min_lon=min_lon, max_lon=max_lon, 
                         min_lat=min_lat, max_lat=max_lat,
                         lon_lines=lon_grid,lat_lines=lat_grid,
                         fig=fig,  projection=projection,
                         lat_0=radar.latitude['data'][0],
                         lon_0=radar.longitude['data'][0])
    display.plot_cross_hair(10,npts=100)
    display.plot_point(Pad_lon, Pad_lat, symbol = 'kv', markersize=5)
    display.plot_point(PCMK_lon, PCMK_lat, symbol = 'kv', markersize=5)
    for rng in range(50,200,50):
        display.plot_range_ring(rng, ls='dashed', lw=0.5)

    stitle = 'NPOL  ' + time_str
    print(stitle)
    plt.suptitle(stitle,fontsize=30, va='top', y=0.99)
    plt.tight_layout(pad=5, h_pad=2,w_pad=3)                
    #
    # *** Save to a png file
    #
    plt.savefig(png_file)
    #plt.show()
    plt.close()
    print()

    
def plot_radar_ppi_panel_gf(radar, radar_DT, sweep_num, png_file, gatefilter):

    fig = plt.figure(figsize=Figsize)
    
    radar_lon = radar.longitude['data'][:][0]
    radar_lat = radar.latitude['data'][:][0]


# Setting projection and ploting the second tilt

    projection = ccrs.LambertConformal(central_latitude=radar.latitude['data'][0],
                                       central_longitude=radar.longitude['data'][0])

    display = pyart.graph.RadarMapDisplay(radar)
#
# *** Calculate bounding limits for map
#
    dtor = math.pi/180.0
    max_range=150.0
    maxrange_meters = max_range * 1000.
    meters_to_lat = 1. / 111177.
    meters_to_lon =  1. / (111177. * math.cos(radar_lat * dtor))

    min_lat = radar_lat - maxrange_meters * meters_to_lat 
    max_lat = radar_lat + maxrange_meters * meters_to_lat 

    min_lon = radar_lon - maxrange_meters * meters_to_lon 
    max_lon = radar_lon + maxrange_meters * meters_to_lon 

    Pad_lon = -75.471
    Pad_lat =  37.934
    
    PCMK_lon = -75.515
    PCMK_lat =  38.078



    lon_grid = np.arange(-77,-73, 0.5)
    lat_grid = np.arange( 37, 40, 0.5)
#    
#     Fields = ['DBZ2', 'VEL2','ZDR2','RHOHV2','PHIDP2','KDP2']
#     Cmaps  = ['pyart_NWSRef','pyart_NWSVel','pyart_RefDiff', 
#               'pyart_Wild25','pyart_NWSVel','pyart_NWSRef']
#     Vmins   = [0, -20, -1, 0,    0, -1]
#     Vmaxs   = [65, 20, 4,  1,  180,  3]
#     CBarLabels = ['Reflectivity [dBZ]',
#                   'Velocity [m/s]',
#                   'Differential Reflectivity [dB]', 
#                   'Correlation Coefficient',
#                   'Differential Phase [deg]',
#                   'Specific Differential Phase [deg/km]']
#     NFields = len(Fields)   
    
# Extract date/time info from radar_DT as 0-padded strings

    year  = str(radar_DT.year).zfill(4)
    month = str(radar_DT.month).zfill(2)
    day   = str(radar_DT.day).zfill(2)
    hour  = str(radar_DT.hour).zfill(2)
    mint  = str(radar_DT.minute).zfill(2)
    sec   = str(radar_DT.second).zfill(2)

    time_str = month + '/' + day + '/' + year + ' @ ' + hour + ':' + mint + ':' + sec + ' UTC'
    file_time = year + '_' + month + day + '_' + hour + mint + sec
#
# *** Set up figure
#
   # Reflectivity

    ax1 = fig.add_subplot(321,projection=projection)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')

    display.plot_ppi_map('DBZ2', sweep_num, vmin=0, vmax=65,resolution='10m', 
                         projection=projection,
                         cmap='pyart_NWSRef',colorbar_label='DZ [dBZ]',
                         title= 'Reflectivity [dBZ]',
                         min_lon=min_lon, max_lon=max_lon, 
                         min_lat=min_lat, max_lat=max_lat,
                         lon_lines=lon_grid,lat_lines=lat_grid,
                         lat_0=radar.latitude['data'][0],
                         lon_0=radar.longitude['data'][0],
                         gatefilter=gatefilter)

    # Annotate plot
    for rng in range(50,200,50):
        display.plot_range_ring(rng, line_style='k--', lw=0.5)

    for azi in range(0,360,30):
        azimuth = 90. - azi
        dazimuth = azimuth * dtor
        lon_maxrange = radar_lon + math.cos(dazimuth) * meters_to_lon * maxrange_meters
        lat_maxrange = radar_lat + math.sin(dazimuth) * meters_to_lat * maxrange_meters
        display.plot_line_geo([radar_lon, lon_maxrange], [radar_lat, lat_maxrange], 
                              line_style='k--',lw=0.5)
    display.plot_cross_hair(10,npts=100)
    display.plot_point(Pad_lon, Pad_lat, symbol = 'kv', markersize=5)
    display.plot_point(PCMK_lon, PCMK_lat, symbol = 'kv', markersize=5)


    # Radial Velocity
    ax2 = fig.add_subplot(322,projection=projection)
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')

    display.plot_ppi_map('VEL2', sweep_num,resolution='10m', 
                         projection=projection,
                         cmap='pyart_NWSVel',colorbar_label='VR [m/s]',
                         title='Radial Velocity [m/s]',
                         min_lon=min_lon, max_lon=max_lon, 
                         min_lat=min_lat, max_lat=max_lat,
                         lon_lines=lon_grid,lat_lines=lat_grid,
                         lat_0=radar.latitude['data'][0],
                         lon_0=radar.longitude['data'][0],
                         gatefilter=gatefilter)
    display.plot_cross_hair(10,npts=100)
    display.plot_point(Pad_lon, Pad_lat, symbol = 'kv', markersize=5)
    display.plot_point(PCMK_lon, PCMK_lat, symbol = 'kv', markersize=5)
    for rng in range(50,200,50):
        display.plot_range_ring(rng, ls='dashed', lw=0.5)

# Correlation

    ax3 = fig.add_subplot(323,projection=projection)
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')

    display.plot_ppi_map('RHOHV2', sweep_num,resolution='10m', vmin=0.5, vmax=1,
                         cmap='pyart_Wild25',colorbar_label='RH',
                         title='Correlation Coefficient',
                         min_lon=min_lon, max_lon=max_lon, 
                         min_lat=min_lat, max_lat=max_lat,
                         lon_lines=lon_grid,lat_lines=lat_grid,
                         fig=fig,  projection=projection,
                         lat_0=radar.latitude['data'][0],
                         lon_0=radar.longitude['data'][0],
                         gatefilter=gatefilter)
    display.plot_cross_hair(10,npts=100)
    display.plot_point(Pad_lon, Pad_lat, symbol = 'kv', markersize=5)
    display.plot_point(PCMK_lon, PCMK_lat, symbol = 'kv', markersize=5)
    for rng in range(50,200,50):
        display.plot_range_ring(rng, ls='dashed', lw=0.5)

# Differential Phase

    ax4 = fig.add_subplot(324,projection=projection)
    ax4.set_xlabel('Longitude')
    ax4.set_ylabel('Latitude')

    display.plot_ppi_map('ZDR2', sweep_num,resolution='10m', vmin=-1, vmax=4,
                         cmap='pyart_RefDiff',colorbar_label='DR [dB]',
                         title='Differential Reflectivity [dB]',
                         min_lon=min_lon, max_lon=max_lon, 
                         min_lat=min_lat, max_lat=max_lat,
                         lon_lines=lon_grid,lat_lines=lat_grid,
                         fig=fig,  projection=projection,
                         lat_0=radar.latitude['data'][0],
                         lon_0=radar.longitude['data'][0],
                         gatefilter=gatefilter)
    display.plot_cross_hair(10,npts=100)
    display.plot_point(Pad_lon, Pad_lat, symbol = 'kv', markersize=5)
    display.plot_point(PCMK_lon, PCMK_lat, symbol = 'kv', markersize=5)
    for rng in range(50,200,50):
        display.plot_range_ring(rng, ls='dashed', lw=0.5)


# Differential Phase

    ax5 = fig.add_subplot(325,projection=projection)
    ax5.set_xlabel('Longitude')
    ax5.set_ylabel('Latitude')

    display.plot_ppi_map('PHIDP2', sweep_num,resolution='10m', vmin=0, vmax=360,
                         cmap='pyart_NWSRef',colorbar_label='PH [deg]',
                         title='Differential Phase [deg]',
                         min_lon=min_lon, max_lon=max_lon, 
                         min_lat=min_lat, max_lat=max_lat,
                         lon_lines=lon_grid,lat_lines=lat_grid,
                         fig=fig,  projection=projection,
                         lat_0=radar.latitude['data'][0],
                         lon_0=radar.longitude['data'][0],
                         gatefilter=gatefilter)
    display.plot_cross_hair(10,npts=100)
    display.plot_point(Pad_lon, Pad_lat, symbol = 'kv', markersize=5)
    display.plot_point(PCMK_lon, PCMK_lat, symbol = 'kv', markersize=5)
    for rng in range(50,200,50):
        display.plot_range_ring(rng, ls='dashed', lw=0.5)

# Specific Differential Phase

    ax6 = fig.add_subplot(326,projection=projection)
    ax6.set_xlabel('Longitude')
    ax6.set_ylabel('Latitude')

    display.plot_ppi_map('KDP2', sweep_num,resolution='10m', vmin=-1, vmax=3,
                         cmap='pyart_NWSRef',colorbar_label='KD [deg/km]',
                         title='Specific Differential Phase [deg/km]',
                         min_lon=min_lon, max_lon=max_lon, 
                         min_lat=min_lat, max_lat=max_lat,
                         lon_lines=lon_grid,lat_lines=lat_grid,
                         fig=fig,  projection=projection,
                         lat_0=radar.latitude['data'][0],
                         lon_0=radar.longitude['data'][0],
                         gatefilter=gatefilter)
    display.plot_cross_hair(10,npts=100)
    display.plot_point(Pad_lon, Pad_lat, symbol = 'kv', markersize=5)
    display.plot_point(PCMK_lon, PCMK_lat, symbol = 'kv', markersize=5)
    for rng in range(50,200,50):
        display.plot_range_ring(rng, ls='dashed', lw=0.5)

    stitle = 'NPOL  ' + time_str
    print(stitle)
    plt.suptitle(stitle,fontsize=30, va='top', y=0.99)
    plt.tight_layout(pad=5, h_pad=2,w_pad=3)                
    #
    # *** Save to a png file
    #
    plt.savefig(png_file)
    #plt.show()
    plt.close()
    print()

# ****************************************************************************************************


if __name__ == '__main__':

    # Read the radar file.
    in_dir = 'C:\\Users\\Bgard\\Documents\\GPM Fall 2020\\GPM Fall 2020\\'
    file = in_dir + 'np1200430212716.RAWAJCE' # PH fold

    radar = pyart.io.read(file, file_field_names=True)
    # See what moments are present.
    radar_DT = pyart.util.datetime_from_radar(radar)
    print(radar.fields.keys())
    
    # Get DateTime from radar
    year  = str(radar_DT.year).zfill(4)
    month = str(radar_DT.month).zfill(2)
    day   = str(radar_DT.day).zfill(2)
    hour  = str(radar_DT.hour).zfill(2)
    mint  = str(radar_DT.minute).zfill(2)
    sec   = str(radar_DT.second).zfill(2)

    #Make the File Time
    file_time = year + '_' + month + day + '_' + hour + mint + sec
        

    sweep_num = 0
    
    # Create a gatefilter from radar
    gatefilter = pyart.filters.GateFilter(radar)

    # Lets remove reflectivity values below a threshold (5)
    dbz_thresh = 5
    rh_thresh = 0.75
    gatefilter = pyart.correct.despeckle_field(radar, 'VEL2', label_dict=None, threshold=-100,
                                                size=10, gatefilter=None, delta=5.0)
    gatefilter.exclude_below('DBZ2', dbz_thresh)
    gatefilter.exclude_below('RHOHV2', rh_thresh)
    
    #Save file directory for ppi panel
    png_file = 'C:\\Users\\Bgard\\Documents\\GPM Fall 2020\\GPM Fall 2020\\' + file_time + '_filtered_PPI.png'
    print('    --> ' + png_file)
    plot_radar_ppi_panel_gf(radar, radar_DT, sweep_num, png_file, gatefilter) 
    
    
    print('Done.')