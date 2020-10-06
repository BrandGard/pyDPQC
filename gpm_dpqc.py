import numpy as np
import math
from copy import deepcopy
import pyart
import pandas as pd
import gpm_dpqc_utils as gu
from csu_radartools.csu_liquid_ice_mass import linearize
from csu_radartools import (csu_fhc, csu_liquid_ice_mass, csu_blended_rain,
                            csu_dsd, csu_kdp, csu_misc, fundamentals)

# ***************************************************************************************
def unfold_phidp(radar, ref_field_name, phi_field_name):

    """
    Function for unfolding phidp
    Written by: David A. Marks, NASA/WFF/SSAI

    Parameters:
    radar: pyart radar object
    ref_field_name: name of reflectivty field (should be QC'd)
    phi_field_name: name of PhiDP field (should be unfolded)

    Return
    radar: radar object with unfolded PHM field included
    """
    
    BAD_DATA       = -9999.0        # Fill in bad data values
    FIRST_GATE     = 5000           # First gate to begin unfolding
    MAX_PHIDP_DIFF = 270.0          # Set maximum phidp gate-to-gate difference allowed

    # Copy current PhiDP field to phm_field
    phm_field = radar.fields[phi_field_name]['data'].copy()

    # Get gate spacing info and determine start gate for unfolding
    # Start at 5 km from radar to avoid clutter gates with bad phase 
    gate_spacing = radar.range['meters_between_gates']
    start_gate = int(FIRST_GATE / gate_spacing)
    nsweeps = radar.nsweeps
    nrays = phm_field.data.shape[0]

    # Loop through the rays and perform unfolding if needed
    # By specifying iray for data and mask provides gate info
    # for iray in range(0, 1):

    for iray in range(0, nrays-1):
        gate_data = phm_field.data[iray]
        ngates = gate_data.shape[0]

        # Conditional where for valid data -- NPOL only.
        # Phase data from other radars should be evaluated for phase range values
        good = np.ma.where(gate_data >= 0)
        bad = np.ma.where(gate_data < 0)
        final_data = gate_data[good]
        num_final = final_data.shape[0]
        #print("Num_Final = ", str(num_final))

        folded_gates = 0
        for igate in range(start_gate,num_final-2):
            diff = final_data[igate+1] - final_data[igate]
            if abs(diff) > MAX_PHIDP_DIFF:
                #print('igate: igate+1: ',final_data[igate],final_data[igate+1])
                final_data[igate+1] += 360
                folded_gates += 1

        # Put corrected data back into ray
        gate_data[good] = final_data
        gate_data[bad] = BAD_DATA

        # Replace corrected data in phm_field. Original phidp remains the same
        phm_field.data[iray] = gate_data

    # Create new field for corrected PH -- name it phm
    radar = gu.add_field_to_radar_object(phm_field, radar, field_name='PHM', 
		units='deg',
		long_name=' Differential Phase (Marks)',
		standard_name='Specific Differential Phase (Marks)',
		dz_field=ref_field_name)
    return radar

# ***************************************************************************************



def calculate_kdp(radar, ref_field_name, phi_field_name):

    """
    Wrapper for calculating Kdp using csu_kdp.calc_kdp_bringi from CSU_RadarTools
    Thank Timothy Lang et al.
    Parameters:
    -----------
    radar: pyart radar object
    ref_field_name: name of reflectivty field (should be QC'd)
    phi_field_name: name of PhiDP field (should be unfolded)

    Return
    ------
    radar: radar object with KDPB, PHIDPB and STDPHIDP added to original

    NOTE: KDPB: Bringi Kdp, PHIDPB: Bringi-filtered PhiDP, STDPHIB: Std-dev of PhiDP
    """

    radar_lon = radar.longitude['data'][:][0]
    radar_lat = radar.latitude['data'][:][0]

    DZ = gu.extract_unmasked_data(radar, ref_field_name)
    DP = gu.extract_unmasked_data(radar, phi_field_name)

    # Range needs to be supplied as a variable, with same shape as DZ
    rng2d, az2d = np.meshgrid(radar.range['data'], radar.azimuth['data'])

    KDPB, PHIDPB, STDPHIB = csu_kdp.calc_kdp_bringi(dp=DP, dz=DZ, rng=rng2d/1000.0, 
                                                    thsd=12, gs=125.0, window=5)

    radar =gu.add_field_to_radar_object(KDPB, radar, field_name='KDPB', 
		units='deg/km',
		long_name='Specific Differential Phase (Bringi)',
		standard_name='Specific Differential Phase (Bringi)',
		dz_field=ref_field_name)

    radar = gu.add_field_to_radar_object(PHIDPB, radar, 
		field_name='PHIDPB', units='deg',
		long_name='Differential Phase (Bringi)',
		standard_name='Differential Phase (Bringi)',
		dz_field=ref_field_name)

    radar = gu.add_field_to_radar_object(STDPHIB, radar, 
		field_name='STDPHIB', units='deg',
		long_name='STD Differential Phase (Bringi)',
		standard_name='STD Differential Phase (Bringi)',
		dz_field=ref_field_name)

    return radar

# ***************************************************************************************

def calibrate(radar, ref_field_name, zdr_field_name, ref_cal, zdr_cal):

    """
    Applies calibration adjustments to DBZ and ZDR fields.

    returns:  radar with calibrated fields.

    Written by: Jason L. Pippitt, NASA/GSFC/SSAI
    """

    fill_value = -9999
    #Calibrate Reflectivity field
    ref_field = radar.fields[ref_field_name]['data'].copy()
    corr_dbz = pyart.correct.correct_bias(radar, bias=ref_cal, field_name=ref_field_name)
    corz_dict = {'data': corr_dbz['data'], 'units': '', 'long_name': 'DBZ2',
                 '_FillValue': fill_value, 'standard_name': 'DBZ2'}
    radar.add_field('DBZ2', corz_dict, replace_existing=True)
    
    #Calibrate ZDR field
    zdr_field = radar.fields[zdr_field_name]['data'].copy()
    corr_zdr = pyart.correct.correct_bias(radar, bias=zdr_cal, field_name=zdr_field_name)
    corzdr_dict = {'data': corr_zdr['data'], 'units': '', 'long_name': 'ZDR2',
                 '_FillValue': fill_value, 'standard_name': 'ZDR2'}
    radar.add_field('ZDR2', corzdr_dict, replace_existing=True)

    return radar

# ***************************************************************************************

def get_ruc_sounding(radar, snd_dir, site):

    """
    Finds correct RUC hourly sounding based on radar time stamp.
    Reads text sounding file and creates dictionary for input to SkewT.
    returns:  sounding dictionary

    Written by: Jason L. Pippitt, NASA/GSFC/SSAI
    """
    
    radar_DT = pyart.util.datetime_from_radar(radar)
     
    month = str(radar_DT.month).zfill(2)
    day = str(radar_DT.day).zfill(2)
    year = str(radar_DT.year).zfill(4)
    hh = str(radar_DT.hour).zfill(2)
    mm = str(radar_DT.minute).zfill(2)
    hour =  radar_DT.hour

    mdays = [00,31,28,31,30,31,30,31,31,30,31,30,31]
                
    if radar_DT.minute >= 30: hour = radar_DT.hour + 1
    if hour == 24: 
        mday = radar_DT.day + 1
        hour = 0
        if mday > mdays[radar_DT.month]:
            cmonth = radar_DT.month + 1
            mday = 1
            if(cmonth > 12):
                cmonth = 1
                mday = 1
                cyear = radar_DT.year + 1
                year = str(cyear).zfill(4)
            month = str(cmonth).zfill(2)
        day = str(mday).zfill(2)
    hh = str(hour).zfill(2)
    sounding_dir = snd_dir + year + '/' + month + day + '/' + site + '/' + site + '_' + year + '_' + month + day + '_' + hh + 'UTC.txt'
    
    print()
    print('Sounding file -->  ' + sounding_dir)
    print()

    headings = ["PRES","HGHT","TEMP","DWPT","RELH","MIXR","DRCT","SKNT","THTA","THTE","THTV"]
    colspecs = [(3, 9), (11, 18), (20, 26), (28, 34), (36, 38), (40, 42),
                (44, 46), (48, 50), (52, 54), (56, 58), (60, 62)]
    
    sound = pd.read_fwf(sounding_dir, names=headings, header=None, colspecs=colspecs,skiprows=2)

    presssure_pa = sound.PRES
    height_m = sound.HGHT
    temperature_c = sound.TEMP
    dewpoint_c = sound.DWPT

    mydata=dict(zip(('hght','pres','temp','dwpt'),(height_m,presssure_pa,temperature_c,dewpoint_c)))

    return mydata

# ***************************************************************************************

def mask_cone_of_silence(radar, field_sector, sector):

    """
    filter out any data inside the cone of silence

    Parameters
    ----------
    radar : radar object
            the radar object where the data is
    fieldname : str
            name of the field to filter
    sector : dict
            a dictionary defining the region of interest

    Returns
    -------
    cos_flag : ndarray
    a field array with ones in gates that are in the Region of Interest

    Written by: Jason L. Pippitt, NASA/GSFC/SSAI
    """

    cos_flag = np.ma.ones((radar.nrays, radar.ngates), dtype=int)

    # check for altitude limits
    if sector['hmin'] is not None:
        cos_flag[radar.gate_altitude['data'] < sector['hmin']] = 0
    if sector['hmax'] is not None:
        cos_flag[radar.gate_altitude['data'] > sector['hmax']] = 0

    # check for range limits

    if sector['rmin'] is not None:
        cos_flag[:, radar.range['data'] < sector['rmin']] = 0

    if sector['rmax'] is not None:
        cos_flag[:, radar.range['data'] > sector['rmax']] = 0

    # check elevation angle limits
    if sector['elmin'] is not None:
        cos_flag[radar.elevation['data'] < sector['elmin'], :] = 0

    if sector['elmax'] is not None:
        cos_flag[radar.elevation['data'] > sector['elmax'], :] = 0

    # check min and max azimuth angle
    if sector['azmin'] is not None and sector['azmax'] is not None:
        if sector['azmin'] <= sector['azmax']:
            cos_flag[radar.azimuth['data'] < sector['azmin'], :] = 0
            cos_flag[radar.azimuth['data'] > sector['azmax'], :] = 0
        if sector['azmin'] > sector['azmax']:
            cos_flag[np.logical_and(
            radar.azimuth['data'] < sector['azmin'],
            radar.azimuth['data'] > sector['azmax']), :] = 0
    elif sector['azmin'] is not None:
        cos_flag[radar.azimuth['data'] < sector['azmin'], :] = 0
    elif sector['azmax'] is not None:
        cos_flag[radar.azimuth['data'] > sector['azmax'], :] = 0
    return cos_flag

# ***************************************************************************************

def sector_wipeout(radar, field_sector, sector):
    
    """
    filter out any data inside the region of interest defined by sector

    Parameters
    ----------
    radar : radar object
            the radar object where the data is
    fieldname : str
            name of the field to filter
    sector : dict
            a dictionary defining the region of interest

    Returns
    -------
    sector_wipeout : ndarray
    a field array with ones in gates that are in the Region of Interest

    Written by: Jason L. Pippitt, NASA/GSFC/SSAI
    """

    sector_wipeout = np.ma.ones((radar.nrays, radar.ngates), dtype=int)

    # check for altitude limits
    if sector['hmin'] is not None:
        sector_wipeout[radar.gate_altitude['data'] < sector['hmin']] = 0

    if sector['hmax'] is not None:
        sector_wipeout[radar.gate_altitude['data'] > sector['hmax']] = 0

    # check for range limits
    if sector['rmin'] is not None:
        sector_wipeout[:, radar.range['data'] < sector['rmin']] = 0

    if sector['rmax'] is not None:
        sector_wipeout[:, radar.range['data'] > sector['rmax']] = 0

    # check elevation angle limits
    if sector['elmin'] is not None:
        sector_wipeout[radar.elevation['data'] < sector['elmin'], :] = 0

    if sector['elmax'] is not None:
        sector_wipeout[radar.elevation['data'] > sector['elmax'], :] = 0

    # check min and max azimuth angle
    if sector['azmin'] is not None and sector['azmax'] is not None:
        if sector['azmin'] <= sector['azmax']:
            sector_wipeout[radar.azimuth['data'] < sector['azmin'], :] = 0
            sector_wipeout[radar.azimuth['data'] > sector['azmax'], :] = 0
        if sector['azmin'] > sector['azmax']:
            sector_wipeout[np.logical_and(
            radar.azimuth['data'] < sector['azmin'],
            radar.azimuth['data'] > sector['azmax']), :] = 0
    elif sector['azmin'] is not None:
        sector_wipeout[radar.azimuth['data'] < sector['azmin'], :] = 0
    elif sector['azmax'] is not None:
        sector_wipeout[radar.azimuth['data'] > sector['azmax'], :] = 0
    return sector_wipeout

# ***************************************************************************************

def rh_sector(radar, field_sector, sector):
    
    """
    filter out any data inside the region of interest that is < rh_sector

    Parameters
    ----------
    radar : radar object
            the radar object where the data is
    fieldname : str
            name of the field to filter
    sector : dict
            a dictionary defining the region of interest

    Returns
    -------
    sector_rh : ndarray
    a field array with ones in gates that are in the Region of Interest

    Written by: Jason L. Pippitt, NASA/GSFC/SSAI
    """
    sector_wipeout = np.ma.ones((radar.nrays, radar.ngates), dtype=int)

    # check for altitude limits
    if sector['hmin'] is not None:
        sector_wipeout[radar.gate_altitude['data'] < sector['hmin']] = 0

    if sector['hmax'] is not None:
        sector_wipeout[radar.gate_altitude['data'] > sector['hmax']] = 0

    # check for range limits
    if sector['rmin'] is not None:
        sector_wipeout[:, radar.range['data'] < sector['rmin']] = 0

    if sector['rmax'] is not None:
        sector_wipeout[:, radar.range['data'] > sector['rmax']] = 0

    # check elevation angle limits
    if sector['elmin'] is not None:
        sector_wipeout[radar.elevation['data'] < sector['elmin'], :] = 0

    if sector['elmax'] is not None:
        sector_wipeout[radar.elevation['data'] > sector['elmax'], :] = 0

    # check min and max azimuth angle
    if sector['azmin'] is not None and sector['azmax'] is not None:
        if sector['azmin'] <= sector['azmax']:
            sector_wipeout[radar.azimuth['data'] < sector['azmin'], :] = 0
            sector_wipeout[radar.azimuth['data'] > sector['azmax'], :] = 0
        if sector['azmin'] > sector['azmax']:
            sector_wipeout[np.logical_and(
            radar.azimuth['data'] < sector['azmin'],
            radar.azimuth['data'] > sector['azmax']), :] = 0
    elif sector['azmin'] is not None:
        sector_wipeout[radar.azimuth['data'] < sector['azmin'], :] = 0
    elif sector['azmax'] is not None:
        sector_wipeout[radar.azimuth['data'] > sector['azmax'], :] = 0
    
    rh = radar.fields['RHOHV2']['data'].copy()
    sector_r = np.ones(rh.shape)
    rh_sec = sector['rh_sec']
    rh_lt = np.ma.where(rh < rh_sec , 1, 0)
    sec_f = np.logical_and(rh_lt == 1 , sector_wipeout == 1)
    sector_r[sec_f] = 0
    return sector_r

# ***************************************************************************************

def sd_sector(radar, field_sector, sector):

    """
    filter out any data inside the region of interest that is < sd_sector

    Parameters
    ----------
    radar : radar object
            the radar object where the data is
    fieldname : str
            name of the field to filter
    sector : dict
            a dictionary defining the region of interest

    Returns
    -------
    sector_sd : ndarray
    a field array with ones in gates that are in the Region of Interest

    Written by: Jason L. Pippitt, NASA/GSFC/SSAI
    """
    sector_wipeout = np.ma.ones((radar.nrays, radar.ngates), dtype=int)

    # check for altitude limits
    if sector['hmin'] is not None:
        sector_wipeout[radar.gate_altitude['data'] < sector['hmin']] = 0

    if sector['hmax'] is not None:
        sector_wipeout[radar.gate_altitude['data'] > sector['hmax']] = 0

    # check for range limits
    if sector['rmin'] is not None:
        sector_wipeout[:, radar.range['data'] < sector['rmin']] = 0

    if sector['rmax'] is not None:
        sector_wipeout[:, radar.range['data'] > sector['rmax']] = 0

    # check elevation angle limits
    if sector['elmin'] is not None:
        sector_wipeout[radar.elevation['data'] < sector['elmin'], :] = 0

    if sector['elmax'] is not None:
        sector_wipeout[radar.elevation['data'] > sector['elmax'], :] = 0

    # check min and max azimuth angle
    if sector['azmin'] is not None and sector['azmax'] is not None:
        if sector['azmin'] <= sector['azmax']:
            sector_wipeout[radar.azimuth['data'] < sector['azmin'], :] = 0
            sector_wipeout[radar.azimuth['data'] > sector['azmax'], :] = 0
        if sector['azmin'] > sector['azmax']:
            sector_wipeout[np.logical_and(
            radar.azimuth['data'] < sector['azmin'],
            radar.azimuth['data'] > sector['azmax']), :] = 0
    elif sector['azmin'] is not None:
        sector_wipeout[radar.azimuth['data'] < sector['azmin'], :] = 0
    elif sector['azmax'] is not None:
        sector_wipeout[radar.azimuth['data'] > sector['azmax'], :] = 0
    
    sd = radar.fields['STDPHIB']['data'].copy()
    sector_s = np.ones(sd.shape)
    sd_sec = sector['sd_sec']
    sd_lt = np.ma.where(sd > sd_sec , 1, 0)
    sec_f = np.logical_and(sd_lt == 1 , sector_wipeout == 1)
    sector_s[sec_f] = 0
    return sector_s

# ***************************************************************************************

def ph_sector(radar, field_sector, sector):
    
    """
    filter out any data inside the region of interest that is < ph_sector

    Parameters
    ----------
    radar : radar object
            the radar object where the data is
    fieldname : str
            name of the field to filter
    sector : dict
            a dictionary defining the region of interest

    Returns
    -------
    sector_ph : ndarray
    a field array with ones in gates that are in the Region of Interest

    Written by: Jason L. Pippitt, NASA/GSFC/SSAI
    """
    sector_wipeout = np.ma.ones((radar.nrays, radar.ngates), dtype=int)

    # check for altitude limits
    if sector['hmin'] is not None:
        sector_wipeout[radar.gate_altitude['data'] < sector['hmin']] = 0

    if sector['hmax'] is not None:
        sector_wipeout[radar.gate_altitude['data'] > sector['hmax']] = 0

    # check for range limits
    if sector['rmin'] is not None:
        sector_wipeout[:, radar.range['data'] < sector['rmin']] = 0

    if sector['rmax'] is not None:
        sector_wipeout[:, radar.range['data'] > sector['rmax']] = 0

    # check elevation angle limits
    if sector['elmin'] is not None:
        sector_wipeout[radar.elevation['data'] < sector['elmin'], :] = 0

    if sector['elmax'] is not None:
        sector_wipeout[radar.elevation['data'] > sector['elmax'], :] = 0

    # check min and max azimuth angle
    if sector['azmin'] is not None and sector['azmax'] is not None:
        if sector['azmin'] <= sector['azmax']:
            sector_wipeout[radar.azimuth['data'] < sector['azmin'], :] = 0
            sector_wipeout[radar.azimuth['data'] > sector['azmax'], :] = 0
        if sector['azmin'] > sector['azmax']:
            sector_wipeout[np.logical_and(
            radar.azimuth['data'] < sector['azmin'],
            radar.azimuth['data'] > sector['azmax']), :] = 0
    elif sector['azmin'] is not None:
        sector_wipeout[radar.azimuth['data'] < sector['azmin'], :] = 0
    elif sector['azmax'] is not None:
        sector_wipeout[radar.azimuth['data'] > sector['azmax'], :] = 0
    
    ph = radar.fields['PHM']['data'].copy()
    sector_p = np.ones(ph.shape)
    ph_sec = sector['ph_sec']
    ph_lt = np.ma.where(ph < ph_sec , 1, 0)
    sec_f = np.logical_and(ph_lt == 1 , sector_wipeout == 1)
    sector_p[sec_f] = 0
    return sector_p

# ***************************************************************************************

def threshold_qc_dpfields(radar, dbz_thresh, rh_thresh, 
                          dr_min, dr_max, 
                          sq_thresh, 
                          sec, cos,
                          thresh_dict):

    """
    Use gatefilter to apply QC by looking at various thresholds of field values.
    Written by: Jason L. Pippitt, NASA/GSFC/SSAI

    Parameters
    ----------
    radar : radar object
            the radar object where the data is

    Thresholds for qc'ing data: dbz_thresh, rh_thresh,  dr_min, dr_max, kdp_min,
                                 kdp_max, sq_thresh, sd_thresh, sec, cos

    Returns
    -------
    radar: QC'd radar with gatefilters applied.

    """
    
    # Create a pyart gatefilters from radar
    dbzfilter = pyart.filters.GateFilter(radar)
    gatefilter = pyart.filters.GateFilter(radar)

    # Apply dbz, sector, and SQI thresholds regardless of Temp 
    if thresh_dict['do_dbz'] == 'yes': dbzfilter.exclude_below('DBZ2', dbz_thresh)
    if thresh_dict['do_sector'] == 'yes': dbzfilter.exclude_not_equal('SEC', cos)
    if thresh_dict['do_rh_sector'] == 'yes': dbzfilter.exclude_not_equal('SECRH', sec) 
    if thresh_dict['do_cos'] == 'yes': dbzfilter.exclude_not_equal('COS', cos)
    if thresh_dict['do_sq'] == 'yes': dbzfilter.exclude_below('SQI2', sq_thresh)
    
    # Apply gate filters to radar
    for fld in radar.fields:
        nf = deepcopy(radar.fields[fld])
        nf['data'] = np.ma.masked_where(dbzfilter.gate_excluded, nf['data'])
        radar.add_field(fld, nf, replace_existing=True) 

    # Create AP filter variables
    if thresh_dict['do_ap'] == 'yes':
        dz = radar.fields['DBZ2']['data'].copy()
        dr = radar.fields['ZDR2']['data'].copy()
        ap = np.ones(dz.shape)
        dz_lt = np.ma.where(dz <= 45 , 1, 0)
        dr_lt = np.ma.where(dr >= 3 , 1, 0)
        ap_t = np.logical_and(dz_lt == 1 , dr_lt == 1)
        ap[ap_t] = 0
        gu.add_field_to_radar_object(ap, radar, field_name='AP', 
                                     units='0 = Z < 0, 1 = Z >= 0',
                                     long_name='AP Mask', 
                                     standard_name='AP Mask', 
                                     dz_field='DBZ2')

    # Call gatefliters for each field based on temperature or beam height
    if thresh_dict['use_qc_height'] == 'yes':
        qc_height = thresh_dict['qc_height'] * 1000
        gatefilter.exclude_all()
        gatefilter.include_below('HEIGHT', qc_height)
        if thresh_dict['do_rh'] == 'yes': gatefilter.exclude_below('RHOHV2', rh_thresh)
        if thresh_dict['do_zdr'] == 'yes': gatefilter.exclude_outside('ZDR2', dr_min, dr_max)
        if thresh_dict['do_ap'] == 'yes': gatefilter.exclude_not_equal('AP', sec)
        gatefilter.include_above('HEIGHT', qc_height)
    elif thresh_dict['use_qc_height'] == 'no':
        gatefilter.exclude_all()
        gatefilter.include_above('TEMP', 1.5)
        if thresh_dict['do_rh'] == 'yes': gatefilter.exclude_below('RHOHV2', rh_thresh)
        if thresh_dict['do_zdr'] == 'yes': gatefilter.exclude_outside('ZDR2', dr_min, dr_max)
        if thresh_dict['do_ap'] == 'yes': gatefilter.exclude_not_equal('AP', sec)
        gatefilter.include_below('TEMP', 1.6)   

    # Apply gate filters to radar
    for fld in radar.fields:
        nf = deepcopy(radar.fields[fld])
        nf['data'] = np.ma.masked_where(gatefilter.gate_excluded, nf['data'])
        radar.add_field(fld, nf, replace_existing=True)
    return radar



# ***************************************************************************************
def threshold_qc_calfields(radar, kdp_min, kdp_max, sd_thresh, sec, ph_thresh, thresh_dict):

    """
    Use gatefilter to apply QC by looking at thresholds of calculated field values.
    Written by: Jason L. Pippitt, NASA/GSFC/SSAI

    Parameters
    ----------
    radar : radar object
            the radar object where the data is

    Thresholds for qc'ing data: kdp_min, kdp_max, sd_thresh

    Returns
    -------
    radar: QC'd radar with gatefilters applied.

    """

    # Create a pyart gatefilter from radar
    secfilter = pyart.filters.GateFilter(radar)
    gatefilter_cal = pyart.filters.GateFilter(radar)

    # Apply sector thresholds regardless of temp 
    if thresh_dict['do_sd_sector'] == 'yes': secfilter.exclude_not_equal('SECSD', sec)
    if thresh_dict['do_ph_sector'] == 'yes': secfilter.exclude_not_equal('SECPH', sec)
    
    # Apply gate filters to radar
    for fld in radar.fields:
        nf = deepcopy(radar.fields[fld])
        nf['data'] = np.ma.masked_where(secfilter.gate_excluded, nf['data'])
        radar.add_field(fld, nf, replace_existing=True)

    # Call gatefliters for calculated fields based on temperature or beam height
    if thresh_dict['use_qc_height'] == 'yes':
        qc_height = thresh_dict['qc_height'] * 1000
        gatefilter_cal.exclude_all()
        gatefilter_cal.include_below('HEIGHT', qc_height)
        if thresh_dict['do_sd'] == 'yes': gatefilter_cal.exclude_above('STDPHIB', sd_thresh)
        if thresh_dict['do_kdp'] == 'yes': gatefilter_cal.exclude_outside('KDP2', kdp_min, kdp_max)
        if thresh_dict['do_ph'] == 'yes': gatefilter_cal.exclude_below('PHM', ph_thresh)
        gatefilter_cal.include_above('HEIGHT', qc_height)
    elif thresh_dict['use_qc_height'] == 'no':
        gatefilter_cal.exclude_all()
        gatefilter_cal.include_above('TEMP', 1.5)
        if thresh_dict['do_sd'] == 'yes': gatefilter_cal.exclude_above('STDPHIB', sd_thresh)
        if thresh_dict['do_kdp'] == 'yes': gatefilter_cal.exclude_outside('KDP2', kdp_min, kdp_max)
        if thresh_dict['do_ph'] == 'yes': gatefilter_cal.exclude_below('PHM', ph_thresh)
        gatefilter_cal.include_below('TEMP', 1.6)    

    # Apply gate filters to radar
    for fld in radar.fields:
        nf = deepcopy(radar.fields[fld])
        nf['data'] = np.ma.masked_where(gatefilter_cal.gate_excluded, nf['data'])
        radar.add_field(fld, nf, replace_existing=True)
    return radar

# ***************************************************************************************
def rename_fields_in_radar(radar, old_fields, new_fields):

    """
    Rename fields we want to keep with GPM, 2-letter IDs (e.g. CZ, DR, KD)
    Written by: David B. Wolff, NASA/WFF

    Parameters:
    -----------
    radar: pyart radar object
    old_fields: List of current field names that we want to change
    new_fields: List of field names we want to change the name to

    Return:
    -------
    radar: radar with more succinct field names

    """    

    # Change names of old fields to new fields using pop
    nl = len(old_fields)
    for i in range(0,nl):
        old_field = old_fields[i]
        new_field = new_fields[i]
        radar.fields[new_field] = radar.fields.pop(old_field)
        i += 1   
    return radar

# ***************************************************************************************
def remove_fields_from_radar(radar, drop_fields):

    """
    Remove fields from radar that are not needed.
    Written by: David B. Wolff, NASA/WFF

    Parameters:
    -----------
    radar: pyart radar object
    drop_fields: List of fields to drop from radar object

    Return:
    -------
    radar: Pruned radar

    """    

    # Remove fields we no longer need.
    for field in drop_fields:
        radar.fields.pop(field)
    return radar

def calc_field_diff(radar, field1, field2):

    """
    Compute the difference between two fields.
    Written by: Charanjit S. Pabla, NASA/WFF/SSAI

    Parameters:
    -----------
    radar: pyart radar object
    field1: radar moment (str)
    field2: radar moment (str)

    Return:
    -------
    radar: pyart radar object with difference field included
    """
    
    #make sure fields are included in the radar object
    if field1 and field2 in radar.fields.keys():
    
        #copy fields
        f1 = radar.fields[field1]['data'].copy()
        f2 = radar.fields[field2]['data'].copy()
    
        #compute difference
        diff = f2 - f1
    
        #put back into radar objective
        radar.add_field_like(field, field+'_diff')
    
        return radar
    else:
        print(radar.fields.keys())
        raise Exception("{} {} fields are not in radar object".format(field1, field2))    

def calc_dsd_sband_tokay_2020(dz, zdr, loc='wff', d0_n2=False):
    """
    Compute dm and nw or (d0 and n2) following the methodology of Tokay et al. 2020
    Works for S-band radars only
    Written by: Charanjit S. Pabla, NASA/WFF/SSAI

    Parameters:
    -----------
    dz: Reflectivity (numpy 2d array)
    zdr: Differential Reflectivity (numpy 2d array)
    
    Keywords:
    -----------
    loc: wff (default, string); region or field campaign name (DSD depends on environment)
         user options: wff, alabama, ifloods, iphex, mc3e, olympex
    d0_n2: False (default, bool)
        if true then function will return d0 and n2

    Return:
    -------
    dm and nw (default, numpy array)
    if d0_n2 set to True then return d0 and n2 (numpy array)
    """ 
    missing = -32767.0
    dm = np.zeros(dz.shape)
    nw = np.zeros(dz.shape)
    dz_lin = linearize(dz)
    
    #force input string to lower case
    loc = loc.lower()
    
    if not d0_n2:
        
        #compute dm
        if loc == 'wff':
            high = zdr > 3.5
            low = zdr <= 3.5
            dm[high] = 0.0188 * zdr[high]**3 - 0.1963 * zdr[high]**2 + 1.1796 * zdr[high] + 0.7183
            dm[low] = 0.0990 * zdr[low]**3 - 0.6141 * zdr[low]**2 + 1.8364 * zdr[low] + 0.4559
        elif loc == 'alabama':
            high = zdr > 3.8
            low = zdr <= 3.8
            dm[high] = 0.0188 * zdr[high]**3 - 0.1963 * zdr[high]**2 + 1.1796 * zdr[high] + 0.7183
            dm[low] = 0.0453 * zdr[low]**3 - 0.3236 * zdr[low]**2 + 1.2939 * zdr[low] + 0.7065
        elif loc == 'ifloods':
            high = zdr > 3.1
            low = zdr <= 3.1
            dm[high] = 0.0188 * zdr[high]**3 - 0.1963 * zdr[high]**2 + 1.1796 * zdr[high] + 0.7183
            dm[low] = 0.1988 * zdr[low]**3 - 1.0747 * zdr[low]**2 + 2.3786 * zdr[low] + 0.3623
        elif loc == 'iphex':
            high = zdr > 2.9
            low = zdr <= 2.9
            dm[high] = 0.0188 * zdr[high]**3 - 0.1963 * zdr[high]**2 + 1.1796 * zdr[high] + 0.7183
            dm[low] = 0.1887 * zdr[low]**3 - 1.0024 * zdr[low]**2 + 2.3153 * zdr[low] + 0.3834
        elif loc == 'mc3e':
            high = zdr > 3.1
            low = zdr <= 3.1
            dm[high] = 0.0188 * zdr[high]**3 - 0.1963 * zdr[high]**2 + 1.1796 * zdr[high] + 0.7183
            dm[low] = 0.1861 * zdr[low]**3 - 1.0453 * zdr[low]**2 + 2.3804 * zdr[low] + 0.3561
        elif loc == 'olpymex':
            high = zdr > 2.7
            low = zdr <= 2.7
            dm[high] = 0.0188 * zdr[high]**3 - 0.1963 * zdr[high]**2 + 1.1796 * zdr[high] + 0.7183
            dm[low] = 0.2209 * zdr[low]**3 - 1.1577 * zdr[low]**2 + 2.3162 * zdr[low] + 0.3486
    
        #compute nw
        nw = np.log10(35.43 * dz_lin * dm**-7.192)
    
        #set dm and nw missing based on acceptable zdr range
        zdr_bad = np.logical_or(zdr <= 0.0, zdr > 4.0)
        dm[zdr_bad] = missing
        nw[zdr_bad] = missing
    
        #set dm and nw missing based on acceptable dm range
        dm_bad = np.logical_or(dm < 0.5, dm > 4.0)
        dm[dm_bad] = missing
        nw[dm_bad] = missing
    
        #set dm and nw missing based on acceptable nw range
        bad_nw = np.logical_or(nw < 0.5, nw > 6.0)
        nw[bad_nw] = missing
        dm[bad_nw] = missing
        
        return dm, nw
    else:
        #user request d0 and n2
        
        d0 = dm
        n2 = nw
        
        d0 = 0.0215 * zdr**3 - 0.0836 * zdr**2 + 0.7898 * zdr + 0.8019
        n2 = np.log10(20.957 * dz_lin * d0**-7.7)
        
        #set d0 and n2 missing
        d0_bad = d0 <= 0
        n2_bad = n2 <= 0
        d0[d0_bad] = missing
        n2[n2_bad] = missing
        
        return d0, n2

