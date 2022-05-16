#Sea Ice Functions used for the analysis of Eureka data
#Claude de Rijke-Thomas
#8th May 2022
import numpy as np
import netCDF4
import h5py
from pyproj import Proj
from scipy import spatial, stats
from scipy.stats import spearmanr, linregress
import csv
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
import cartopy.crs as ccrs
import cartopy
from pyproj import Proj
from scipy.integrate import simps
from shapely.geometry import Point
from to_precision import std_notation
easeProj = Proj("+proj=laea +lat_0=90 +lon_0=0 +x_0=0 +y_0=0 +north +ellps=WGS84 \
                +datum=WGS84 +units=m +no_defs") #using the EaseGrid2.0 projection
warnings.filterwarnings("ignore", category=RuntimeWarning) #ignoring mean of empty slice warning for when there is no colocated data in footprint
from scipy.signal import find_peaks
from scipy.signal import peak_prominences
import cmath
from math import tan, atan, asin, cos, sin, atan2
from shapely.geometry.polygon import Polygon
from scipy import optimize
from typing import Iterable




#WGS84 and Topex-Poseidon coordinate system lengths:
a_wgs = 6378137.00; #m
a_tp = 6378136.30; #m
b_wgs = 6356752.314245; #m
b_tp = 6356751.600563; #m

#ku-band radar file path (for all the _deconv.nc files):
ku_path = '/Users/claudederijkethomas/Desktop/PhD/PYTHON/OIB/19-21Apr2016/ku/'
#ATM laser data path (for all the .h5 files):
atm_path = '/Users/claudederijkethomas/Desktop/PhD/PYTHON/OIB/19-21Apr2016/atm/'

#The following is opening a treemap of all the KDTrees corresponding with particular atm files:
with open("../OIB/19-21Apr2016/fullTreeMap19-21Apr2016.pickle", "rb") as pickle_in:
    tree_map = pickle.load(pickle_in)
#I then make this into a dictionary (so that you can input a atm file name and get its KDTree):
KDTreeDict = {tree_map[i][0]:tree_map[i][1] for i in range(len(tree_map))}
#The following is a 2D array, where each array starts witht the name of an atm file, and all coincident ku files:
with open("../OIB/19-21Apr2016/full_coincidence_arr19-21Apr2016.pickle", "rb") as pickle_in:
    coincidence_arr = pickle.load(pickle_in)

#class to make the return of a function object-like:
class result_container:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

absarr = lambda arr: np.array([abs(arr[j]) for j in range(len(arr))])

def steamroll(items):
    """Yield items from any nested iterable.
    For example:
    l = np.array([[1,2,4],[[1,2]]], dtype='object')
    or
    l = [[[[[1,2,4]]]],[[1,2]]]
    if you want a list from it do:
    my_list = list(steamroll(original_array))
    """
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from np.ravel(x)
        else:
            yield x

def fflat(items): #makes any list really, really flat
    return list(steamroll(items))

#first-estimate retracking of Ku-band surface:
def KuBandSurface(filename, threshold_fraction, plot=False,respectability=False, RHS_criteria = False, plot_no=0):
    """
    Calculates the surface height of the sea ice from Ku-Band Data:))
    
        inputs:
            filename: a .nc file containing Ku-Band data.
            threshold_fraction: whatever fraction of the maximum peak power you would like the threshold to be
                                (e.g a 40% threshold would require a 0.4 threshold fraction.      
            plot: (boolean) whether you want one of the waveforms with the threshold to be plotted.
            
        output:
            lon_ku: a 1-D array of aircraft longitudes during measurement taking, in degrees
            lat_ku: a 1-D array of aircraft latitudes during measurement taking, in degrees
            alt_ku: a 1-D array of ice-interface measurement altitudes, in metres   
            pitch_ku: a 1-D array of aircraft pitch angles, measured in degrees
            roll_ku: a 1-D array of aircraft roll angles, measured in degrees
            aircraft_alt_ku: the aircraft altitude during measurement taking, in metres"""
    ku_subpath = "ku/"
    try:
        nc = netCDF4.Dataset(filename) #making an nc file in the format that Python can understand
    except FileNotFoundError:
        try:
            nc = netCDF4.Dataset(ku_subpath+filename)
        except FileNotFoundError:
            nc = netCDF4.Dataset(ku_path+filename)
    lat_ku = nc['lat'][:] #measurement latitude
    lon_ku = nc['lon'][:] #measurement longitude
    delta_h = ((a_wgs - a_tp)*np.cos(lat_ku*np.pi/180)**2 + (b_wgs - b_tp)*np.sin(lat_ku*np.pi/180)**2) # in metres
    aircraft_alt_ku = nc['altitude'][:] - delta_h #the aircraft altitude relative to WGS84 (in metres)
    pitch_ku = nc['pitch'][:]*np.pi/180 #in radians
    roll_ku = nc['roll'][:]*np.pi/180 #in radians
    fasttime_ku = nc['fasttime'][:][:] #the range bin time (two-way travel time) equivalent for a waveform in 
    # microseconds (not actually all the same value (slightly increases with increasing range bin))
    

    log_amplitude = nc['amplitude'] #a 2-D array of stacked waveforms, with the first index being waveform number 
                                    #and the second being the range bin number of a particular waveform. 
                                  #Contains log-amplitude waveform data
                                    #Shape = [number of waveforms x number of range bins in each waveform]
    interp_range_bin_arr = [] #array of interpolated range bins for where the 70% threshold of each echo is
    #looping over the total number of echoes:
    
    respectable_indices = np.array([], dtype='int')
    indices_of_working_echoes = np.array([],dtype='int')
    for i in range(len(log_amplitude)):
        #removing the logarithm scale from the echo amplitudes:
        unlog_amp = 10**(np.array(log_amplitude[i])/10)
        
        #finding the effective start of the signal beyond the noise floor:
        
        try:
            #truncating the waveform/echo so that only parts that could realistically come from the snow layer are included:
            # (distance between range bins =1.27 cm which is roughly 1cm within snow so only taking 1.5m from the start of the signal):
            signal_threshold_constant = np.mean(unlog_amp[:250])+6*np.std(unlog_amp[:250], ddof=1)
            start_of_signal  = [index for index,value in enumerate(unlog_amp) if value > signal_threshold_constant][10]
            unlog_amp = unlog_amp[:start_of_signal+250]
        except:
            try:
                signal_threshold_constant = np.mean(unlog_amp[:250])+4*np.std(unlog_amp[:250], ddof=1)
                start_of_signal  = [index for index,value in enumerate(unlog_amp) if value > signal_threshold_constant][10]
                unlog_amp = unlog_amp[:start_of_signal+250]
            except:
                pass
        
        #finding the indices of peaks for an echo:
        peak_indices,peak_dict = find_peaks(unlog_amp, width=2,height=0.03) #finding all the indices of all the peaks in the echo
             #finding all the indices of all the peaks in the echo, as well as the values of their heights
        peak_heights = peak_dict['peak_heights']
        
    
        #determining whether the echo is free of any prominent sidelobes near the max power:
        try:
            if isItARespectableEcho(unlog_amp, peak_indices, RHS_criteria):
                respectable_indices = np.append(respectable_indices, i)
        except:
            pass
        major_peak_index = np.argmax(unlog_amp) #finding the index of the largest peak in the echo
        #finding the peak-to-trough depth of the largest (major) peak in the echo:

        if len(peak_indices)>0 and max(unlog_amp)>0:
            if np.min(abs(peak_indices-major_peak_index))<=2:
                #finding what scipy thinks is the maximal index of the largest peak:
                major_gaus_peak_index = peak_indices[np.argmin(abs(peak_indices-major_peak_index))]
                pass
            else:
                continue #this will not be an index of a working echo
        elif len(peak_indices)==0:
            continue # this will not be the index of a working echo
        elif max(unlog_amp)<=0:
            continue #this will not be the index of a working echo
        max_peak_depth = peak_prominences(unlog_amp,peak_indices)[0][list(peak_indices).index(major_gaus_peak_index)]
        #calculating the peak threshold value:
        #peak_threshold = np.max(unlog_amp)-(1-threshold_fraction)*max_peak_depth
        highest_peak_index = peak_indices[np.argmax(peak_heights)]
        peak_threshold = np.mean(unlog_amp[:250]) + threshold_fraction*(unlog_amp[highest_peak_index] - np.mean(unlog_amp[:250]))

        j = np.argmax(unlog_amp)-250
        #finding the index just above the % peak threshold value::
        while unlog_amp[j]<peak_threshold:
            j = j+1
        #finding the interpolated range bin corresponding to the % threshold value:
        interp_range_bin = j-1 + abs(peak_threshold - unlog_amp[j-1])/abs(unlog_amp[j] - unlog_amp[j-1])
        #plotting one of the echoes (of index i):
        if i ==plot_no and plot==True:
            plt.scatter(np.argmax(unlog_amp), np.max(unlog_amp), color = '#d62728')
            plt.plot(unlog_amp)
            if abs(interp_range_bin-j)<abs(interp_range_bin-(j-1)):
                plt.scatter(j, unlog_amp[j], color = '#d62728')
            else:
                plt.scatter(j-1, unlog_amp[j-1], color = '#d62728')
        #appending to an array of interpolated range bins corresponding to the % threshold value:
        interp_range_bin_arr.append(interp_range_bin) #interp_range_bin
        indices_of_working_echoes = np.append(indices_of_working_echoes,i)
    #calculate the interpolated fasttimes for each waveform corresponding to the % range bin thresholds:
    fasttime_interp = np.interp(interp_range_bin_arr, range(len(fasttime_ku)), fasttime_ku)
    #time it takes signal to reach % threshold surface in seconds:
    time_to_surf_arr = np.array(fasttime_interp)*10**-6/2
    #calculate the distance to the ice surface:
    cosarr = lambda arr: np.array([cos(arr[j]) for j in range(len(arr))])
    #converting echo delay time to space
    space_to_surf_arr = time_to_surf_arr*299792458                #/1.000293
    #calculating the altitude of the ice interface, accounting for pitch and roll of satellite:
    alt_ku = aircraft_alt_ku[indices_of_working_echoes] - space_to_surf_arr#*cosarr(pitch_ku)*cosarr(roll_ku)
    if respectability==True:
        return lon_ku[respectable_indices],lat_ku[respectable_indices],alt_ku[respectable_indices], \
                pitch_ku[respectable_indices], roll_ku[respectable_indices], aircraft_alt_ku[respectable_indices], respectable_indices
    else:
        return lon_ku[indices_of_working_echoes], lat_ku[indices_of_working_echoes], alt_ku, pitch_ku[indices_of_working_echoes], roll_ku[indices_of_working_echoes], aircraft_alt_ku[indices_of_working_echoes], indices_of_working_echoes

def correctEastNorthKu(filename, coord_sys, threshold_fraction=0.70000, snow_thickness = 0, respectability=False):
    """Returns the correct easting and northing of the centres of the Ku footprint measurements, accounting for 
    the pitch and roll of the aircraft
    
    REQUIRES AT LEAST TWO SEQUENTIAL KU-BAND DATA POINTS
     ONLY WORKS WITH RESPECTABILITY = FALSE BECAUSE OF INDICES OF WORKING ECHOES"""
    if respectability:
        lon_ku, lat_ku, alt_ku, pitch_angle, roll_angle, aircraft_alt_ku, appropriate_indices = KuBandSurface(filename, threshold_fraction, plot = False, respectability=respectability)
    else:
        lon_ku, lat_ku, alt_ku, pitch_angle, roll_angle, aircraft_alt_ku, appropriate_indices = KuBandSurface(filename, threshold_fraction, plot = False, respectability=respectability)
    try:
        if coord_sys == "utm" or coord_sys == "UTM":
            _,_,zone_number,zone_designator =utm.from_latlon(lat_ku[0], lon_ku[0])
            myProj = Proj("+proj=utm +zone="+str(zone_number)+str(zone_designator)+", +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
        elif coord_sys == "Ease2.0" or coord_sys == "EASE2.0" or coord_sys =="ease2.0" or coord_sys=="ease":
            myProj = Proj("+proj=laea +lat_0=90 +lon_0=0 +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
        elif coord_sys == "stereo" or coord_sys == "stereopolar":
            myProj = Proj("+proj=laea +lat_0=90 +lat_ts=70 +lon_0=-45 +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    except: raise InputError("Need to input \"UTM\" or \"EASE2.0\" ")
    ku_x, ku_y = myProj(lon_ku,lat_ku)
    east_ku, north_ku, azi_angle = np.array([]), np.array([]), np.array([])
    
    for i in range(len(ku_x)):     
        #calculating the dimensions of each footprint accounting for aircraft to air-snow interface height:::
        H = (aircraft_alt_ku[i] - alt_ku[i] - snow_thickness)  
        #for every index but the last (so that 'i+1' is still within bounds):
        if i<len(ku_x)-1:
            #angle from north – direction aircraft is travelling
            azi_ang = atan2(ku_x[i+1]- ku_x[i], ku_y[i+1]- ku_y[i])
        else:
            azi_ang = atan2(ku_x[i]- ku_x[i-1], ku_y[i]- ku_y[i-1])
            
        #making a systematic across track error parameter:
        systematic_acc_tr_correction = 0 #how much systematically the radar is rightwards accross track of where it should be
        systematic_al_tr_correction = 0 #how much systematically the radar is upwards along track of where it should be
        
        #making systematic easting and northing error parameters:
        systematic_east_correction = 0 #how much the radar is systematically east of where it should be
        systematic_north_correction = 0 #how much teh radar is systematically norrth of where it should be
        
        #used to include. -cos(azi_ang)*tan(roll_angle[i])*H + sin(azi_ang)*tan(pitch_angle[i])*H
        east_correction = 0 + cos(azi_ang)*systematic_acc_tr_correction \
                            + sin(azi_ang)*systematic_al_tr_correction + systematic_east_correction
        #used to include +sin(azi_ang)*tan(roll_angle[i])*H + cos(azi_ang)*tan(pitch_angle[i])*H
        north_correction = 0 -sin(azi_ang)*systematic_acc_tr_correction \
                            + cos(azi_ang)*systematic_al_tr_correction + systematic_north_correction
        
        east_ku = np.append(east_ku, ku_x[i] + east_correction)
        north_ku = np.append(north_ku, ku_y[i] + north_correction)
        azi_angle = np.append(azi_angle, azi_ang)
    return east_ku,north_ku,alt_ku, pitch_angle, roll_angle, aircraft_alt_ku, azi_angle, appropriate_indices

def footprintYielder(east_ku, north_ku, alt_ku, pitch_ku, roll_ku, aircraft_alt_ku, azi_ang):
    """ Yields the correct ku-band footprint for each measurement, accounting for satellite orientation, 
    pitch and roll. Requires an input of at least two measurements to determine orientation of satellite
    
    Output:
            Ku footprint in a shapely polygon format"""
        #for each ku-band footprint:
    for i in range(len(east_ku)):
        
        #calculating the dimensions of each footprint:::
        H = (aircraft_alt_ku[i] - alt_ku[i])
        #pulse-limited across track dimension (pp. 12/17 of Ku SAR doc):
        across_tr_dim = 2*np.sqrt(299792458*1.5*H/(3.5*10**9))
        along_tr_dim =  H*tan(asin((299792458)/(2*1.12*14.75*10**9)))# used to be H*tan(asin(np.sqrt((299792458)/(2*H*14.75*10**9))))
        pitch_angle = pitch_ku[i] #in radians
        roll_angle = roll_ku[i]
        azi_angle = azi_ang[i]
        
        acc_tr_proj = across_tr_dim#/cos(roll_angle) # m across track footprint projection onto the ice
        al_tr_proj = along_tr_dim#/cos(pitch_angle) # m along track footprint projection onto the ice
        
        rect_diag = np.sqrt(acc_tr_proj**2 + al_tr_proj**2)
        
        #making the angles from the center to the corners of a rectanlge:
        azimuth1 = atan(acc_tr_proj/al_tr_proj) + azi_angle
        azimuth2 = atan(-acc_tr_proj/al_tr_proj) + azi_angle
        azimuth3 = atan(acc_tr_proj/al_tr_proj)+np.pi + azi_angle # first point + 180 degrees
        azimuth4 = atan(-acc_tr_proj/al_tr_proj)+np.pi + azi_angle 
        
        #making the corners of the footprint:
        corner_1_easting, corner_1_northing = [east_ku[i] + (rect_diag/2)*sin(azimuth1),
                                               north_ku[i] + (rect_diag/2)*cos(azimuth1)]
        corner_2_easting, corner_2_northing = [east_ku[i] + (rect_diag/2)*sin(azimuth2),
                                               north_ku[i] + (rect_diag/2)*cos(azimuth2)]
        corner_3_easting, corner_3_northing = [east_ku[i] + (rect_diag/2)*sin(azimuth3),
                                               north_ku[i] + (rect_diag/2)*cos(azimuth3)]
        corner_4_easting, corner_4_northing = [east_ku[i] + (rect_diag/2)*sin(azimuth4),
                                               north_ku[i] + (rect_diag/2)*cos(azimuth4)]
        
        #making a polygon for the ku footprint:
        polygon = Polygon([(corner_1_easting, corner_1_northing), (corner_2_easting, corner_2_northing),
                           (corner_3_easting, corner_3_northing), (corner_4_easting, corner_4_northing)])
        yield polygon, [east_ku[i], north_ku[i]], rect_diag/2
        
def footprintDataYielder(east_ku, north_ku, alt_ku, pitch_ku, roll_ku, aircraft_alt_ku, azi_ang, atm_coords, kdtree):
    for footprint, centre, radius in footprintYielder(east_ku, north_ku, alt_ku, pitch_ku, roll_ku, aircraft_alt_ku, azi_ang):
        neigh_list = []
        footprint_data = np.array([], dtype = 'int')
        neigh_list.append(kdtree.query_ball_point(centre, r=radius))
        try: 
            neigh_indices = [neigh_list[0][i] for i in range(len(neigh_list[0]))]
        except:
            neigh_indices = []
        if len(neigh_indices)>0:
            for neigh_index in neigh_indices:  
                if footprint.contains(Point(atm_coords[neigh_index])):
                    footprint_data = np.append(footprint_data, neigh_index)
        yield footprint, footprint_data

def projectedLaser(h5file, coord_sys):
    if coord_sys == "utm" or coord_sys == "UTM":
        _,_,zone_number,zone_designator =utm.from_latlon(lat_ku[0], lon_ku[0])
        myProj = Proj("+proj=utm +zone="+str(zone_number)+str(zone_designator)+", +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    elif coord_sys == "Ease2.0" or coord_sys == "EASE2.0" or coord_sys =="ease2.0" or coord_sys=="ease":
        myProj = Proj("+proj=laea +lat_0=90 +lon_0=0 +x_0=0 +y_0=0 +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    elif coord_sys in ("stereo","stereopolar"):
        myProj = Proj("+proj=laea +lat_0=90 +lat_ts=70 +lon_0=-45 +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    else:
        raise InputError("Need to input \"UTM\" or \"EASE2.0\" ")
    atm_subpath = "atm/"
    if not isinstance(h5file, (list, tuple, np.ndarray)) or len(h5file)==1:
        if isinstance(h5file, (list, tuple, np.ndarray)):
            h5file=str(h5file[0])
        try:
            with h5py.File(str(h5file), 'r') as f:
                lat_atm = f['latitude'][:]
                lon_atm = f['longitude'][:]
                alt_atm = f['elevation'][:]
                pitch_atm = f['/instrument_parameters/pitch'][:]*np.pi/180 #converted to radians !!
                roll_atm = f['/instrument_parameters/roll'][:]*np.pi/180 #converted to radians !!
                tra_str = f['/instrument_parameters/xmt_sigstr'][:]
                rec_str = f['/instrument_parameters/rcv_sigstr'][:]
                pulse_width = f['/instrument_parameters/pulse_width'][:]
                gdop = f['/instrument_parameters/gps_pdop'][:]
                f.close()
        except OSError:
            try:
                with h5py.File(atm_subpath+str(h5file), 'r') as f:
                    lat_atm = f['latitude'][:]
                    lon_atm = f['longitude'][:]
                    alt_atm = f['elevation'][:]
                    pitch_atm = f['/instrument_parameters/pitch'][:]*np.pi/180 #converted to radians !!
                    roll_atm = f['/instrument_parameters/roll'][:]*np.pi/180 #converted to radians !!
                    tra_str = f['/instrument_parameters/xmt_sigstr'][:]
                    rec_str = f['/instrument_parameters/rcv_sigstr'][:]
                    pulse_width = f['/instrument_parameters/pulse_width'][:]
                    gdop = f['/instrument_parameters/gps_pdop'][:]
                    f.close()
            except OSError:
                with h5py.File(atm_path+str(h5file), 'r') as f:
                    lat_atm = f['latitude'][:]
                    lon_atm = f['longitude'][:]
                    alt_atm = f['elevation'][:]
                    pitch_atm = f['/instrument_parameters/pitch'][:]*np.pi/180 #converted to radians !!
                    roll_atm = f['/instrument_parameters/roll'][:]*np.pi/180 #converted to radians !!
                    tra_str = f['/instrument_parameters/xmt_sigstr'][:]
                    rec_str = f['/instrument_parameters/rcv_sigstr'][:]
                    pulse_width = f['/instrument_parameters/pulse_width'][:]
                    gdop = f['/instrument_parameters/gps_pdop'][:]
                    f.close()
    else:
        lat_atm = np.array([],dtype='double')
        lon_atm = np.array([],dtype='double')
        alt_atm = np.array([],dtype='double')
        tra_str = np.array([],dtype='double')
        pitch_atm = np.array([],dtype='double')
        roll_atm = np.array([],dtype='double')
        rec_str = np.array([],dtype='double')
        pulse_width = np.array([],dtype='double')
        gdop = np.array([], dtype='double')
        try:
            #making sure that if the same file is inputted twice in the list/array then the data isnt duplicated:
            h5file = list(dict.fromkeys(h5file))
            #looping over each file:
            for subpart in h5file:
                with h5py.File(str(subpart), 'r') as f:
                    lat_atm = np.append(lat_atm, f['latitude'][:])
                    lon_atm = np.append(lon_atm, f['longitude'][:])
                    alt_atm = np.append(alt_atm, f['elevation'][:])
                    tra_str = np.append(tra_str, f['/instrument_parameters/xmt_sigstr'][:])
                    rec_str = np.append(rec_str, f['/instrument_parameters/rcv_sigstr'][:])
                    pitch_atm = np.append(pitch_atm, f['/instrument_parameters/pitch'][:]*np.pi/180) #converted to radians!!
                    roll_atm = np.append(roll_atm, f['/instrument_parameters/roll'][:]*np.pi/180) #converted to radians!!
                    gdop = np.append(gdop, f['/instrument_parameters/gps_pdop'][:])
                    pulse_width = np.append(pulse_width, f['/instrument_parameters/pulse_width'][:])
                    f.close()
        except OSError:
            try:
                for subpart in h5file:
                    with h5py.File(atm_subpath+str(subpart), 'r') as f:
                        lat_atm = np.append(lat_atm, f['latitude'][:])
                        lon_atm = np.append(lon_atm, f['longitude'][:])
                        alt_atm = np.append(alt_atm, f['elevation'][:])
                        tra_str = np.append(tra_str, f['/instrument_parameters/xmt_sigstr'][:])
                        rec_str = np.append(rec_str, f['/instrument_parameters/rcv_sigstr'][:])
                        pitch_atm = np.append(pitch_atm, f['/instrument_parameters/pitch'][:]*np.pi/180) #converted to radians!!
                        roll_atm = np.append(roll_atm, f['/instrument_parameters/roll'][:]*np.pi/180) #converted to radians!!
                        gdop = np.append(gdop, f['/instrument_parameters/gps_pdop'][:])
                        pulse_width = np.append(pulse_width, f['/instrument_parameters/pulse_width'][:])
                        f.close()
            except OSError:
                for subpart in h5file:
                    with h5py.File(atm_path+str(subpart), 'r') as f:
                        lat_atm = np.append(lat_atm, f['latitude'][:])
                        lon_atm = np.append(lon_atm, f['longitude'][:])
                        alt_atm = np.append(alt_atm, f['elevation'][:])
                        tra_str = np.append(tra_str, f['/instrument_parameters/xmt_sigstr'][:])
                        rec_str = np.append(rec_str, f['/instrument_parameters/rcv_sigstr'][:])
                        pitch_atm = np.append(pitch_atm, f['/instrument_parameters/pitch'][:]*np.pi/180) #converted to radians!!
                        roll_atm = np.append(roll_atm, f['/instrument_parameters/roll'][:]*np.pi/180) #converted to radians!!
                        gdop = np.append(gdop, f['/instrument_parameters/gps_pdop'][:])
                        pulse_width = np.append(pulse_width, f['/instrument_parameters/pulse_width'][:])
                        f.close()
    atm_coords = list(zip(*myProj(lon_atm, lat_atm)))
    atm_east,atm_north = myProj(lon_atm, lat_atm)
    return result_container(coords = atm_coords, east = atm_east, north = atm_north, alt =alt_atm,
                            rec_str= rec_str, tra_str = tra_str, pulse_width = pulse_width,
                           pitch = pitch_atm, roll = roll_atm, gdop = gdop)

def KDTreePlot(east_ku, north_ku, alt_ku, pitch_ku, roll_ku, aircraft_alt_ku, azi_ang, atm_coords, kdtree):
    ax = plt.subplot(111)
    plt.gca().set_aspect('equal', adjustable='datalim')
    for footprint, data in footprintDataYielder(east_ku, north_ku, alt_ku, pitch_ku, roll_ku, aircraft_alt_ku, azi_ang, atm_coords, kdtree):
        ax.plot(*footprint.exterior.xy)
        within = [atm_coords[datum] for datum in data]
        try:
            ax.scatter(*zip(*(within)))
        except:
            pass
    plt.tight_layout()
    ax.set_yticklabels(["{:.1f}".format(t) for t in ax.get_yticks()])
    ax.set_xticklabels(["{:.1f}".format(t) for t in ax.get_xticks()], rotation='vertical')
    ax.patch.set_facecolor('white')
    plt.xlabel('Easting $/m$', fontsize = 14)
    plt.ylabel("Northing $/m$", fontsize = 14)
    plt.setp([ax.get_xticklines(), ax.get_yticklines(), ax.get_xticklabels(), ax.get_yticklabels(),
             ax.spines.values()], color='black')
    plt.title("ATM Laser Data within Ku radar Footprints", fontsize = 16)
    plt.tight_layout()
#     plt.savefig("plots/MultiFlyoverKdTreeFootprintPlotFYIEureka.png", dpi = 200)

def xysEveryLengthScale(lons,lats, length_scale, track_break_off_length=5000, coordtype='degrees'):
    """calculating a set of coordinates every length scale along the track
    (in order to calculate averages over a certain  length scale)"""
    coords_every_LS_arr = []
    LSxs = np.array([],dtype='double') #length-scale x-coordinates
    LSys = np.array([],dtype = 'double')
    myProj = Proj("+proj=laea +lat_0=90 +lon_0=0 +x_0=0 +y_0=0 +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    if coordtype=='degrees':
        xs, ys  = myProj(lons, lats)
    elif coordtype=='metres':
        xs, ys = lons, lats
    else:
        raise
    x_before_LS, y_before_LS = xs[0], ys[0]
    starting_x, starting_y = xs[0], ys[0]
    for i in range(len(lons)):
        #if the next point is less than the length scale away:
        if np.sqrt(abs((xs[i]-starting_x)**2 + (ys[i]-starting_y)**2))<length_scale:
            x_before_LS, y_before_LS = xs[i], ys[i]
        #if the track has seemingly broken off and started again:
        elif np.sqrt(abs((xs[i]-starting_x)**2 + (ys[i]-starting_y)**2))>track_break_off_length:
            starting_x,starting_y=xs[i], ys[i]
            x_before_LS, y_before_LS = xs[i], ys[i]
        else: #if the next point is more than the length scale away:
            x_after_LS, y_after_LS = xs[i], ys[i]
            distance_to_coords_before_LS = np.sqrt(abs((x_before_LS-starting_x)**2 + (y_before_LS-starting_y)**2))
            distance_left = length_scale - distance_to_coords_before_LS

            #angle from the x-axis ([1,0]) going anticlockwise, between 0 and pi as 0 to 180
            angle_from_x_anticlockwise = cmath.phase(complex((x_after_LS - x_before_LS),(y_after_LS - y_before_LS)))
            #calculating the coordinates of the position one length scale away:
            LSx = x_before_LS + cmath.rect(distance_left,angle_from_x_anticlockwise).real
            LSy = y_before_LS + cmath.rect(distance_left,angle_from_x_anticlockwise).imag

            coords_every_LS_arr.append([LSx, LSy])
            LSxs = np.append(LSxs,LSx)
            LSys = np.append(LSys,LSy)

            #reinitialising the starting position:
            starting_x, starting_y = LSx, LSy
            x_before_LS, y_before_LS = LSx, LSy
    return LSxs,LSys

class Waveform:
    """This is a class for all the information around a single Waveform, including whether the snowdepth is estimatable from it,
    what type of waveform is it (singlet, doublet, merged_doublet, nebulous (the interfaces arent well defined enough))"""
    def __init__(self, unlog_amp, fasttime_ku, easting=None,northing=None,peakiness=None,waveform_type=None,\
                waveform_area=None,first_peak_gradient=None, estimated_snow_ice_alt=None,actual_snow_ice_alt=None,\
                estimated_air_snow_alt=None, actual_air_snow_alt=None,extractable_snowdepth_boolean=None, snow_ice_threshold_interp_range_bin=None,\
                snow_ice_std=None, air_snow_std=None,aircraft_alt=None, airsnow_power=None,snowice_power=None,\
                footprint_polygon = None, footprint_atm_alts = None, insitu_snowdepths= None):
        self.unlog_amp = unlog_amp
        self.fasttime_ku = fasttime_ku
        self.aircraft_alt = aircraft_alt
#     def findSnowIceInterpRangeBin(self):
        #keeping the original unlog_amp (in order to look at the original waveform area if I ever should need to):
        original_unlog_amp = unlog_amp
        self.waveform_area = simps(original_unlog_amp)
        self.snow_ice_threshold_interp_range_bin = np.nan
        self.snow_ice_std = snow_ice_std
        self.air_snow_std = air_snow_std
        self.easting = easting
        self.northing = northing
        self.peakiness = peakiness
        self.space_to_worthy_snow_ice_surf = np.nan
        self.estimated_air_snow_alt = estimated_air_snow_alt
        self.actual_snow_ice_alt = actual_snow_ice_alt
        self.actual_air_snow_alt = actual_air_snow_alt
        self.extractable_snowdepth_boolean = extractable_snowdepth_boolean
        self.snow_ice_threshold_interp_range_bin = snow_ice_threshold_interp_range_bin
        self.airsnow_power = airsnow_power
        self.snowice_power = snowice_power
        self.norm_airsnow_power = np.nan
        self.norm_snowice_power = np.nan
        self.footprint_polygon = footprint_polygon
        self.footprint_atm_alts = footprint_atm_alts
        self.insitu_snowdepths = insitu_snowdepths
        #finding the start of the signal that's definitely above the noise floor
        preliminary_start_of_signal  = [i for i,v in enumerate(unlog_amp) if v > np.mean(unlog_amp[:250])+6*np.std(unlog_amp[:250], ddof=1)][10]
        i = preliminary_start_of_signal
        while unlog_amp[i]>np.mean(unlog_amp[:250])+2.5*np.std(unlog_amp[:250], ddof=1):
            i-=1
        start_of_signal=i
        #WARNING: HERE IS A POTENTIAL SOURCE OF ERROR WHERE IM POTENTIALLY OVERRIDING THE ORIGINAL OBJECTS CHARACTERISTICS (OR NOT):
        unlog_amp = unlog_amp/simps(unlog_amp[start_of_signal:start_of_signal+250])            
        peak_indices,peak_dict = find_peaks(unlog_amp,width=2,height=0.002) #finding all the indices of all the peaks in the echo, as well as the values of their heights
        peak_heights = peak_dict['peak_heights']
        highest_peak_index = peak_indices[np.argmax(peak_heights)]
        second_highest_peak_index = 'nan'
        if len(peak_indices)>1:
            second_highest_peak_index = peak_indices[np.argpartition(peak_heights,-2)[-2]]
        threshold_fraction = 0.7000
        peak_threshold  = np.mean(unlog_amp[:250]) + threshold_fraction*(unlog_amp[highest_peak_index] - np.mean(unlog_amp[:250]))
        j = np.argmax(unlog_amp)-250
        #finding the index just above the % peak threshold value::
        while unlog_amp[j]<peak_threshold:
            j = j+1
        #finding the interpolated range bin corresponding to the % threshold value:
        interp_range_bin = j-1 + abs(peak_threshold - unlog_amp[j-1])/abs(unlog_amp[j] - unlog_amp[j-1])
        
        secondary_peak_found_boolean = False #seeing whether there is a significant secondary peak in the waveform that could either correspond to the air-snow interface
        prepeak_found_boolean = False
        merged_peaks_found = False
                                             # or snow-ice interface
        multiple_strong_prepeaks_found = False
        unlog_amps_of_pre_peaks = np.array([],dtype='double')
        peak_indices_of_pre_peaks = np.array([],dtype = 'int')
        snowdepth_appended_boolean = False
        singlet_waveform_boolean = False
        usable_singlet_waveform_boolean = False
        index_of_snow_ice_peak='nan'
        index_of_air_snow_peak='nan'
        waveform_type=None
        
        spatial_array = 299792458*fasttime_ku*(10**-6)/2
        inter_range_bin_dist_cm  = (spatial_array[1]-spatial_array[0])*100
        no_of_range_bins_equiv_to_80cm_of_snow = 80/(inter_range_bin_dist_cm/1.238)
                         
        #looking for potential air-snow peaks before the largest peak in the waveform:
        for peak_index in peak_indices[(peak_indices<highest_peak_index)]:
            #seeing whether it's a prominent enough peak that we can say it could corespond to the air-snow interface, 
            #and not just some backscatter from some off-nadir facet: 
            if unlog_amp[peak_index]>0.05*max(unlog_amp) and peak_index+no_of_range_bins_equiv_to_80cm_of_snow>highest_peak_index:
                unlog_amps_of_pre_peaks = np.append(unlog_amps_of_pre_peaks,unlog_amp[peak_index])
                peak_indices_of_pre_peaks = np.append(peak_indices_of_pre_peaks,peak_index)
        the_tryhard_boolean = False
        if (np.any(unlog_amps_of_pre_peaks) and len(unlog_amps_of_pre_peaks[unlog_amps_of_pre_peaks>=0.3*unlog_amp[highest_peak_index]])==0):
            the_tryhard_boolean = True
            prepeak_index = peak_indices_of_pre_peaks[0]
        elif (np.any(unlog_amps_of_pre_peaks) and len(unlog_amps_of_pre_peaks[unlog_amps_of_pre_peaks>=0.3*unlog_amp[highest_peak_index]]))==1 and \
             len(unlog_amps_of_pre_peaks[unlog_amps_of_pre_peaks>=0.5*unlog_amp[highest_peak_index]])==0:
            the_tryhard_boolean = True
            prepeak_index = peak_indices_of_pre_peaks[0]
        elif (np.any(unlog_amps_of_pre_peaks) and len(unlog_amps_of_pre_peaks[unlog_amps_of_pre_peaks>=0.5*unlog_amp[highest_peak_index]]))==1:
            the_tryhard_boolean = True
            prepeak_index = peak_indices_of_pre_peaks[np.argmax(unlog_amps_of_pre_peaks)]

        if the_tryhard_boolean:
            # prepeak_index = peak_indices_of_pre_peaks[0]
            prepeak_threshold = np.mean(unlog_amp[:250]) + threshold_fraction*(unlog_amp[prepeak_index] - np.mean(unlog_amp[:250]))
            j = prepeak_index-250
            while unlog_amp[j]<prepeak_threshold:
                j+=1
            prepeak_interp_range_bin = j-1 + abs(prepeak_threshold - unlog_amp[j-1])/abs(unlog_amp[j] - unlog_amp[j-1])
            #seeing whether the TFMRA is accidentally hitting the smaller air-snow peak before the more prominent (in this scenario) snow-ice peak:
            prepeak_found_boolean = True

            #seeing if there is enough of a dip between the two peaks to say that they're well seperated enough 
            #to properly estimate the interp_range_bin for the snow-ice_interface:
            if (np.min(unlog_amp[prepeak_index:highest_peak_index+1]) < 0.75*unlog_amp[prepeak_index] and \
            np.min(unlog_amp[prepeak_index:highest_peak_index+1]) < 0.5*unlog_amp[highest_peak_index]): #or (np.min(unlog_amp[prepeak_index:highest_peak_index+1]) < 0.3*unlog_amp[highest_peak_index])
                index_of_air_snow_peak = prepeak_index
                waveform_type = 'doublet'
                #avoiding the main 70% threshold accidentally hitting the peak beforehand:
                #this +1 is definitely needed when adding two indices together in order to find the total index:
                j = np.argmin(unlog_amp[prepeak_index:highest_peak_index+1])+prepeak_index +1
                while unlog_amp[j]<peak_threshold:
                    j+=1
                #making sure the interp_range_bin corresponding to the snow-ice-peak is after the large air-snow peak:
                interp_range_bin = j-1 + abs(peak_threshold - unlog_amp[j-1])/abs(unlog_amp[j] - unlog_amp[j-1])

                self.snow_ice_threshold_interp_range_bin = interp_range_bin
                self.air_snow_threshold_interp_range_bin = prepeak_interp_range_bin
                self.airsnow_power = self.unlog_amp[index_of_air_snow_peak]
                self.snowice_power = self.unlog_amp[highest_peak_index]
                self.extractable_snowdepth_boolean = True
                self.norm_airsnow_power = unlog_amp[index_of_air_snow_peak]
                self.norm_snowice_power = unlog_amp[highest_peak_index]

            #there isnt enough dip between them to estimate the snow-ice interface reliably:
            else:
                waveform_type = 'merged'
                merged_peaks_found = True

        elif np.any(unlog_amps_of_pre_peaks) and len(unlog_amps_of_pre_peaks[unlog_amps_of_pre_peaks>0.3*unlog_amp[highest_peak_index]])>1:
            prepeak_found_boolean = True
            multiple_strong_prepeaks_found = True
            waveform_type = 'strongly multi-prepeaked'
                
        if not prepeak_found_boolean and second_highest_peak_index!='nan':
            #if the secondary peak is as least 50% of the amplitude of the highest peak and is quite prominent:
            #and if the number of peaks after the highest peak that are above 30% of the amplitude of the highest peak is just one:
            if (unlog_amp[second_highest_peak_index]>0.5*unlog_amp[highest_peak_index]) and \
            peak_prominences(unlog_amp,[second_highest_peak_index])[0]>0.002\
            and len(unlog_amp[peak_indices[(peak_indices>highest_peak_index) & (peak_indices<highest_peak_index+no_of_range_bins_equiv_to_80cm_of_snow)]][np.where(unlog_amp[peak_indices[(peak_indices>highest_peak_index) & (peak_indices<highest_peak_index+no_of_range_bins_equiv_to_80cm_of_snow)]]\
                                                                                        > 0.4*unlog_amp[highest_peak_index])])==1:
                #if the 50% threshold of the secondary peak isnt ensumed within the highest peak:
                if np.min(unlog_amp[highest_peak_index:second_highest_peak_index+1])<0.5*unlog_amp[second_highest_peak_index]:
                    secondary_peak_found_boolean=True

                    #starting the iterator at the local minimum between the two points:
                    j = np.argmin(unlog_amp[highest_peak_index:second_highest_peak_index+1])+highest_peak_index+1
                    #finding the index just above the % peak threshold value::
                    second_peak_threshold= np.mean(unlog_amp[:250]) + threshold_fraction*(unlog_amp[second_highest_peak_index] - np.mean(unlog_amp[:250]))

                    while unlog_amp[j]<second_peak_threshold: 
                        j = j+1
                    #finding the interpolated range bin corresponding to the % threshold value of the second biggest peak (presumably from the snow-ice interface 
                    #with the air-snow peak as the largest peak in this case):
                    second_interp_range_bin = j-1 + abs(second_peak_threshold - unlog_amp[j-1])/abs(unlog_amp[j] - unlog_amp[j-1])
#                     interp_range_bin_arr.append(second_interp_range_bin)
                    self.snow_ice_threshold_interp_range_bin = second_interp_range_bin
                    self.air_snow_threshold_interp_range_bin = interp_range_bin
                    self.extractable_snowdepth_boolean = True
#                     extractable_snowdepth_from_waveform_counter+=1
                    self.airsnow_power = self.unlog_amp[highest_peak_index]
                    self.snowice_power = self.unlog_amp[second_highest_peak_index]
                    self.norm_airsnow_power = unlog_amp[highest_peak_index]
                    self.norm_snowice_power = unlog_amp[second_highest_peak_index]
                    waveform_type = 'doublet'

                else:
                    pass
                    waveform_type = 'merged'
                    #print("The peaks arent well seperated enough")
                    merged_peaks_found = True
            else:
                singlet_waveform_boolean = True
                waveform_type = 'singlet'
#                 singlet_waveform_counter+=1
                self.snow_ice_threshold_interp_range_bin = interp_range_bin
                max_peak_depth = peak_prominences(unlog_amp,[highest_peak_index])[0][0]

                peak_10percent_threshold = np.max(unlog_amp)-(1-0.1)*max_peak_depth
                peak_90percent_threshold = np.max(unlog_amp)-(1-0.9)*max_peak_depth
                j = np.argmax(unlog_amp)-250
                while unlog_amp[j]<peak_10percent_threshold:
                    j+=1
                peak10percent_interp_range_bin = j-1 + abs(peak_10percent_threshold - unlog_amp[j-1])/abs(unlog_amp[j] - unlog_amp[j-1])
                j = np.argmax(unlog_amp)-250
                while unlog_amp[j]<peak_90percent_threshold:
                    j+=1
                peak90percent_interp_range_bin = j-1 + abs(peak_90percent_threshold - unlog_amp[j-1])/abs(unlog_amp[j] - unlog_amp[j-1])
        
                fasttimes_of_10and90_percent_threshold_interp = np.interp([peak10percent_interp_range_bin,peak90percent_interp_range_bin], range(len(fasttime_ku)), fasttime_ku)
                times_to_10and90_percent_threshold_arr = np.array(fasttimes_of_10and90_percent_threshold_interp)*10**-6/2 #converting back into microseconds and then dividing by two because the echo has to travel there and back again
                spaces_to_10and90_percent_threshold_arr = times_to_10and90_percent_threshold_arr*299792458
                
                #if the singlet peak has an amplitude>0.053 and the distance between the 10% and 90% of the leading edge is <10cm:
                if unlog_amp[highest_peak_index]>0.053 and (spaces_to_10and90_percent_threshold_arr[1]-spaces_to_10and90_percent_threshold_arr[0])<.10:
                    usable_singlet_waveform_boolean = True
                    waveform_type = 'usable singlet'
        worthy_snow_ice_interface_extracted_boolean = False
        """calculating the snow-ice interface for waveforms with singlet or prominent doublet peaks"""    
        if ((not singlet_waveform_boolean) or usable_singlet_waveform_boolean==True) and not multiple_strong_prepeaks_found and not merged_peaks_found:

            self.extractable_snowdepth_boolean = True
            if aircraft_alt is not None:
    
                snow_ice_interface_worthy_fasttime_interp = np.interp(self.snow_ice_threshold_interp_range_bin, range(len(fasttime_ku)), fasttime_ku)
                time_to_worthy_snow_ice_surf = snow_ice_interface_worthy_fasttime_interp*10**-6/2 #converting back from microseconds to seconds
                #and then dividing by two because the echo has to travel there and back again
                self.space_to_worthy_snow_ice_surf = time_to_worthy_snow_ice_surf*299792458
                ignorant_of_refractive_index_estimated_snow_ice_alt = aircraft_alt - self.space_to_worthy_snow_ice_surf
                if self.footprint_atm_alts is not None:
                    self.estimated_snowdepth = (np.nanmean([alt for alt in self.footprint_atm_alts]) - ignorant_of_refractive_index_estimated_snow_ice_alt)/1.238 
                    self.estimated_snow_ice_alt = np.nanmean([alt for alt in self.footprint_atm_alts]) - self.estimated_snowdepth
                    self.estimated_air_snow_alt = np.nan
                else:
                    self.air_snow_threshold_interp_range_bin = np.nan
                    self.estimated_snowdepth = np.nan
                    self.estimated_snow_ice_alt = np.nan
                    self.estimated_air_snow_alt = np.nan
                    
                if waveform_type == 'doublet':
                    try:
                        #I will soon make this a try: except: statement to catch the error of when there is no self.air_snow_threshold_interp_range_bin:
                        air_snow_interface_worthy_fasttime_interp = np.interp(self.air_snow_threshold_interp_range_bin, range(len(fasttime_ku)), fasttime_ku)
                        time_to_worthy_air_snow_surf = air_snow_interface_worthy_fasttime_interp*10**-6/2 #convert it back from microseconds to seconds
                        space_to_worthy_air_snow_surf = time_to_worthy_air_snow_surf*299792458
                        self.estimated_air_snow_alt = aircraft_alt - space_to_worthy_air_snow_surf #the air-snow altitude estimated by the ku air-snow peak in doublet waveforms
                        self.estimated_snowdepth = (np.nanmean([alt for alt in self.footprint_atm_alts]) - ignorant_of_refractive_index_estimated_snow_ice_alt)/1.238
                        self.estimated_snow_ice_alt = np.nanmean([alt for alt in self.footprint_atm_alts]) - self.estimated_snowdepth
                    except: #this is for when the air_snow_threshold_interp_range_bin is not defined:
                        self.air_snow_threshold_interp_range_bin = np.nan
                        self.estimated_snowdepth = np.nan
                        self.estimated_air_snow_alt = np.nan #the air-snow altitude estimated by the ku air-snow peak in doublet waveforms
                        self.estimated_snow_ice_alt = np.nan
                                                        
            else:
                self.air_snow_threshold_interp_range_bin = np.nan
                self.estimated_snowdepth = np.nan
                self.estimated_air_snow_alt = np.nan #the air-snow altitude estimated by the ku air-snow peak in doublet waveforms
                self.estimated_snow_ice_alt = np.nan 
                                                             
        else:
            self.snow_ice_threshold_interp_range_bin = np.nan
            #I can always add the 'selfs' to the end if i need to (e.g self.snow_ice_threshold_interp_range_bin = snow_ice_threshold_interp_range_bin)
            self.extractable_snowdepth_boolean = False
            self.estimated_air_snow_alt = np.nan
            self.estimated_snow_ice_alt = np.nan
            self.estimated_snowdepth = np.nan
            self.snow_power = None
            self.snow_power = None
            self.air_snow_threshold_interp_range_bin = np.nan
            self.estimated_air_snow_alt = np.nan
        if waveform_type is None:
            waveform_type = 'nebulous'
        self.waveform_type = waveform_type

def errorPlot(x_vals,y_vals,y_errs,point_labels=False,x_errs =False, fill_between_errorbars = True, annotate=True, label=False, colour='#1f77b4'):
    if label:
        line, = plt.plot(x_vals,y_vals,color=colour, linestyle = '-', marker='.', label=label)
    else:
        line, = plt.plot(x_vals,y_vals,color=colour, linestyle = '-', marker = '.')
    plt.errorbar(x_vals,y_vals, yerr=y_errs, capthick=1,capsize=5, ecolor=colour, color=colour)
    if x_errs:
        plt.errorbar(x_vals,y_vals, xerr=x_errs, capthick=1,capsize=5, ecolor=colour, color=colour)
    #make sure that you've already filtered out all nan data that might contribute to the point_label number values
    if annotate and np.any(point_labels):
        for i, txt in enumerate(point_labels):
            if i==0:
                if y_vals[i+1]>y_vals[i]:
                    plt.annotate(str(int(txt)), (x_vals[i] - (np.nanmax(x_vals)-np.nanmin(x_vals))/50 - len(str(txt))*(np.nanmax(x_vals)-np.nanmin(x_vals))/80, \
                                       [np.nanmean(y_val) for y_val in y_vals][i]+(np.nanmax(y_vals)-np.nanmin(y_vals))/30))
                else:
                    plt.annotate(str(int(txt)), (x_vals[i] - (np.nanmax(x_vals)-np.nanmin(x_vals))/50 - len(str(txt))*(np.nanmax(x_vals)-np.nanmin(x_vals))/80, \
                                       [np.nanmean(y_val) for y_val in y_vals][i]-(np.nanmax(y_vals)-np.nanmin(y_vals))/10))
            elif y_vals[i]> y_vals[i-1]:
                plt.annotate(str(int(txt)), (x_vals[i] - (np.nanmax(x_vals)-np.nanmin(x_vals))/50 - len(str(txt))*(np.nanmax(x_vals)-np.nanmin(x_vals))/80, \
                                   [np.nanmean(y_val) for y_val in y_vals][i]+(np.nanmax(y_vals)-np.nanmin(y_vals))/30))
            else:
                plt.annotate(str(int(txt)), (x_vals[i] - (np.nanmax(x_vals)-np.nanmin(x_vals))/50 - len(str(txt))*(np.nanmax(x_vals)-np.nanmin(x_vals))/80, \
                                   [np.nanmean(y_val) for y_val in y_vals][i]-(np.nanmax(y_vals)-np.nanmin(y_vals))/10))
    plt.xlim(np.nanmin(x_vals)-(np.nanmax(x_vals)-np.nanmin(x_vals))/12,)
    plt.ylim(np.nanmin(y_vals)-np.nanmax(y_errs) - (np.nanmax(y_vals)-np.nanmin(y_vals))/12, \
             np.nanmax(y_vals)+np.nanmax(y_errs) + (np.nanmax(y_vals)-np.nanmin(y_vals))/12) 
    if fill_between_errorbars:
        plt.fill_between(x_vals, np.subtract(y_vals,y_errs, where=~np.isnan(np.array(y_vals)+np.array(y_errs))), \
                         np.add(y_vals,y_errs,where=~np.isnan(np.array(y_vals)+np.array(y_errs))), color='gray', alpha=0.2)

def nthTupleBoxPlotter(x,y,n, percentiles = True): 
    """inputs:
            x: x values (will be sectioned into n different sections)
            y: y values (will be rearranged to correspond to the n different sections of x)
            n: integer number of box plots you want to split into
            percentiles (optional): determines whether to split boxplots into 0-20%,20-40%,40-60%,60-80%,80-100% between the min and max of x (=False), or
                                    group the data into the 0-20%, 20-40%, ... percentiles (even splits of data to make the boxplots) (=True)
        returns:
            box_plot_center_xs: the x values that the box plots are centered on
            y_blocks: a 2D array of y-values that correspond to the different box plots you want to split your graph up into
          """
    if n>1:
        dividers = []
        if not percentiles:
            for i in range(n+1):
                dividers.append(np.nanmin(x) + (np.nanmax(x)-np.nanmin(x))/n*(i))
        elif percentiles:
            for i in range(n+1):
                dividers.append(np.nanpercentile(x,100/n*(i))) 
        y_blocks = []
        for i in range(n):
            if i==n-1:
                y_blocks.append(np.array(y)[(dividers[i]<=np.array(x)) & (np.array(x)<=dividers[i+1])])
            else:
                y_blocks.append(np.array(y)[(dividers[i]<=np.array(x)) & (np.array(x)<dividers[i+1])])
            
        if not percentiles:
            box_plot_center_xs = [(dividers[i] + dividers[i+1])/2 for i in range(n)]
        elif percentiles:
            box_plot_center_xs = [np.nanpercentile(x,(2*i+1)*100/(2*n)) for i in range(n)]
    elif n==1:
        if not percentiles:
            box_plot_center_xs = [np.nanmin(x) + (np.nanmax(x)-np.nanmin(x))/2]  
        else:
            box_plot_center_xs = [np.nanpercentile(x,100/2)]
        y_blocks = [y]    
    return box_plot_center_xs,y_blocks
