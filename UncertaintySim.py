#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:13:18 2019

@author: ZhexingLi
"""
###################################################################################
#
# Script that creates multi-planet synthetic data points in the future and tests 
# how different scenarios improve the orbital ephemeries of the planets.
# Based on orbital phase of the outer most planet (if the longest period provided
# in the list below is the period of the outermost planet.
#
# Author: Zhexing Li
#
# Purpose: Creates synthetic data points that are spread out for the entire period
#          for orbital ephemerides refinement work.
#
# Varible: - (1) Different number of synthetic data points
#          - (2) Gaps from the first synthetic data to the last real data in terms 
#               of number of orbital period.
#          - (3) Different coverage (range) of synthetic data points
#
# To do before running the script: 
# - Put this script, radvel setup file and original data at the same location, 
#   then modify changes in the below section.
# - Changes needed to be made: In radvel setup file:
#   * Add 'Syn' to 'instnames' 
#   * Add 'gamma' and 'jitter' value of 'Syn', usually 'gamma' to be '0.0' and 
#     'jitter' to be '0.1'
#   * Set 'params['gamma_Syn'].vary = False'
#   * Add 'path0' for newly created data file, and 'data0' as well, add 'data0'
#     in the end of dataframe list, revise the location indices of data from
#     different instruments and add that for 'Syn'
#   * Turn trend and curvature terms off
# - If running for Variable 1 with 0 new data, add a special radvel setup file
#   and name it with an additional '_0' to the name of the setup file edited above.
#   i.e. if the main setup file is named HD120.py, then name this special one as
#   HD120_0.py. This special setup file does not have newly added Syn information,
#   just like the original one.
# - Changes needed to be made: In this script:
#   * Change 'name' of the radvel setup file to be used for this script
#   * Change 'no' for number of new data points needed or other variables
#   * Change orbital parameters of the system
#   * Change the sources/locations of existing data to be used for this script
#     (have to be at the same location as this script and the radvel setup file)
#
###################################################################################

##################################################################################
# Import packages
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
import pandas as pd
import os.path
import subprocess
import random

##############################################################################
##############################################################################
# MODIFY CHANGES IN THIS SECTION ONLY

# Name of the radvel setup file
name = 'test.py'

# Which varible to vary? Pick 1, 2, 3 or 4 as indicated from above, and modify variables below
Variable = 1

# Modify variables below for the case desired
if Variable == 1:         # change number of synthetic data points
    # specify if there's an end time, if yes, end times is ref2 below, if not, 
    # then the end of the outer most planet's orbital period
    endtime = 'Yes'
    no=np.arange(1,201,1)            # Number of synthetic data points, always in list
    ref1 = 2517835.8779        # reference time 1, first new data starts around this (2021/01/01)  2459215.500
    ref2 = 2525676.3779        # reference time 2, last new data ends around this (2026/01/01)   2461041.500
elif Variable == 2:       # change gap between last real data and first synthetic data point
    no = 50
    gap = np.array([0.47])         # gap to previous real data, in terms of outer planet's orbital period
    cov = 0.35          # coverage in terms of ORBITAL PERIOD of the outer most planet, from the first (variable 1) uncertainty run (or arbitrary if no first run result)
elif Variable == 3:      # change synthetic data coverage
    no = 50
    ref1 = 2459215.500       # ref1 time from the first (variable 1) uncertainty run (or arbitrary if no first run result)
    cov = np.arange(16.02,20.02,0.02)      # coverage for new data, in terms of ORBITAL PERIOD of the outer planet
elif Variable == 4:      # change synthetic data precision
    no = 50
    ref1 = 2459215.500
    cov = 0.35        # coverage in unit of orbital period of the outer most planet
    eb = np.arange(5.05,10.05,0.05)     # synthetic data rv precision/error bar size in m/s
else:
    pass

# Constants to vary. First element as first planet, second element as second planet etc.
P = [258.224,5227]               # orbital period of the planet(s) (unit of days)
Tp = [2450072.3,2451080.0]          # time of periastron, in unit of days (JD), in the past
e = [0.2347,0.064]                  # eccentricity
K = [49.67,9.8]                  # semi amplitude, in unit of m/s
w = [-0.09,-2.7]                  # argument of periastron of THE STAR, in unit of radians
V_0 = 0.0                     #systematic velocity, in unit of m/s, set here universally as 0 m/s  

# Choose whether to provide stellar jitter (if known) or rms from original fitting for rv scatter.
# Runs for Variable 4 definitely needs 'jit' as the run changes rv uncertainties, which affects
# the rms scatter. Otherwise, if rms is to be held constant throughout the run, pick 'rms'.
scat = 'rms'
if scat == 'jit':
    jitter = 2.876
elif scat == 'rms':
    rms = 3.14               # rms = sqrt(jitter**2 + err**2)
else:
    print ('Please provide a valid input for variable "scat".')
    
# Fetch existing data, order them in the format we want. Can just copy and paste
# this part from the radvel setup file

path1 = '/test/data1.txt'
path2 = '/test/data2.txt'
data1 = pd.read_csv(path1,header=None,skiprows=1,delim_whitespace=True,names=('time','mnvel','errvel'))
data2 = pd.read_csv(path2,header=None,skiprows=1,delim_whitespace=True,names=('time','mnvel','errvel'))
data1['tel'] = 'UCLES'
data2['tel'] = 'HIRES'
dataframe = [data1,data2]
data = pd.concat(dataframe,ignore_index=True)

# Name of the summary files to produce, each one must be in a string, must be one of 
# those parameters in radvel post summary file or derived parameters file.
# Available files to produce: 'per','tc','k','secosw','sesinw','a','mpsini'
sumfile = ['per','tc','k','secosw','sesinw','a','mpsini']

##############################################################################
##############################################################################


#############################################################################
#
# FUNCTION: EXECUTE
#
#############################################################################

def Execute ():
    '''
    Function that executes all below functions in the order it's supposed to.
    Also it's called in the command line section.
    '''

    print('Calculating mean error for new velocity datasets. \n')
    
    error,last_t = GetErr ()
    
    # Path of files
    directory = os.path.dirname(path1)
    datapath = directory + '/data0.txt'
    logpath = directory + '/Log.txt'
    
    # Remove any leftover summary and log files before creating new ones
    for files in os.listdir(directory):
        if files.endswith('Period_Summary.txt'):
            os.remove(directory + '/' + files)
        elif files.endswith('Log.txt'):
            os.remove(directory + '/' + files)
        elif files.endswith('data0.txt'):
            os.remove(directory + '/' + files)
        else:
            pass
    
    # Create summary file
    for item in sumfile:
        sumpath = directory + '/' + item + '_Summary.txt'
        with open (sumpath,'a') as outfile:
            for i in range (len(P)):
                outfile.write(item + str(i+1) + '\t' + item + '_err_up' + str(i+1) + \
                              '\t' + item + '_err_low' + str(i+1) + '\t' + item + \
                              '_err_avg' + str(i+1) + '\t')
            outfile.write('No\t' + 'Gap\t' + 'Cov\t' + 'ErrBar\n')
    
    if Variable == 1:
        Run1(datapath,logpath,error,last_t)
    elif Variable == 2:
        Run2(datapath,logpath,error,last_t)
    elif Variable == 3:
        Run3(datapath,logpath,error,last_t)
    elif Variable == 4:
        Run4(datapath,logpath,last_t)
    else:
        print('Please provide a valid "Variable" number.')

    print('ALL RUNS COMPLETED! \n')
    with open (logpath,'a') as outfile2:
            outfile2.write('ALL RUNS COMPLETED! \n')


#############################################################################
#
# FUNCTION: RUN1
#
#############################################################################

def Run1 (dpath,lpath,err,ltime):
    '''
    Main function for Variable 1 to run all the subfunctions together in order, 
    log, and print outputs.
    
    Input: 1) dpath: Path to the synthetic data file.
           2) lpath: Path to the log file.
           3) err: Uncertainty on each synthetic data point, from GetErr function.
           4) ltime: Time of last available real data.
    Output: None.
    '''
    
    # Run functions below in their order for different new datasets
    for i in range (len(no)):
        if no[i] == 0:
            print('Running RadVel commands for the original dataset. \n')
            with open (dpath,'a') as outfiletemp:
                outfiletemp.write('No new synthetic data required. \n')
            with open (lpath,'a') as outfiletemp2:
                outfiletemp2.write('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%' + \
                                   '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n')
                outfiletemp2.write('Running original dataset. \n\n')
                
            for t in range (5):
                print('Run ' + str(t+1) + '/5 for the original dataset.\n')
                with open (lpath,'a') as outfiletemp2:
                    outfiletemp2.write('\n******************** Run ' + str(t+1) + '/5 ' + \
                                       'for the original dataset. ******************** \n\n' )
                    
                Commands (lpath,no[i])
                
                with open (lpath,'a') as outfiletemp2:
                    outfiletemp2.write('\nRan RadVel commands for run ' + \
                                       str(t+1) + '/5 of the original dataset. \n' )
                print('Retrieving post-MCMC parameters for run ' + str(t+1) + \
                  '/5 of the original dataset and writing to summary files. \n')
                    
                Retrieve (no[i],np.nan,np.nan,err)
                
                with open (lpath,'a') as outfiletemp2:
                    outfiletemp2.write('\nRetrieved post-MCMC parameters for run ' + \
                                       str(t+1) + '/5 of the original dataset and ' + \
                                       'wrote to summary files. \n')
                    outfiletemp2.write('\nRun ' + str(t+1) + '/5 for the original ' + \
                                       'dataset completed.\n\n')
        
        else:
            print('Creating new synthetic dataset ' + str(i+1) + ' for Variable 1.\n')
            
            first,last,spacing,cover = Case1(ltime)
            
            if scat == 'jit':
                scatter = np.sqrt(jitter**2 + err**2)
            elif scat == 'rms':
                scatter = rms
            else:
                pass
            
            Data_Randomize(err,scatter,dpath,first,last,no[i])
            
            with open (lpath,'a') as outfile2:
                outfile2.write('\n----------------------------------------------' + \
                               '-----------------------------\n\n')
                outfile2.write('Created new synthetic dataset ' + str(i+1) + \
                               ' for Variable 1.\n\n')   
            print('Running RadVel commands for new dataset ' + str(i+1) + '.\n')
            
            for j in range (5):
                print('Run ' + str(j+1) + '/5 for dataset ' + str(i+1) + '.\n')           
                with open (lpath,'a') as outfile2:
                    outfile2.write('\n******************** Run ' + str(j+1) + '/5 for dataset ' + \
                                   str(i+1) + '********************\n\n')
                Commands (lpath,no[i])
            
                with open (lpath,'a') as outfile2:
                    outfile2.write('\nRan RadVel commands for run ' + str(j+1) + \
                                   '/5 of new dataset ' + str(i+1) + '.\n')
                print('Retrieving post-MCMC parameters for run ' + str(j+1) + \
                      '/5 of new dataset ' + str(i+1) + ' and writing to summary files. \n')   
            
                Retrieve (no[i],spacing,cover,err)
            
                with open (lpath,'a') as outfile2:
                    outfile2.write('\nRetrieved post-MCMC parameters for run ' + \
                                   str(j+1) + '/5 of new dataset ' + str(i+1) + \
                                   ' and wrote to summary files. \n')
                    outfile2.write('\nRun ' + str(j+1) + '/5 for dataset ' + \
                                   str(i+1) + ' of Variable 1 completed.\n\n')

    
#############################################################################
#
# FUNCTION: RUN2
#
#############################################################################

def Run2 (dpath,lpath,err,ltime):
    '''
    Main function for Variable 2 to run all the subfunctions together in order, 
    log, and print outputs.
    
    Input: 1) dpath: Path to the synthetic data file.
           2) lpath: Path to the log file.
           3) err: Uncertainty on each synthetic data point, from GetErr function
           4) ltime: Time of the last available read data, from GetErr function.
    Output: None.
    '''
    
    # Run functions below in their order for different new datasets
    for i in range (len(gap)):
        print('Creating new synthetic dataset ' + str(i+1) + ' for Variable 2.\n')
            
        first,last = Case2(ltime,gap[i])
        
        if scat == 'jit':
            scatter = np.sqrt(jitter**2 + err**2)
        elif scat == 'rms':
            scatter = rms
        else:
            pass
        
        Data_Randomize(err,scatter,dpath,first,last,no)
        
        with open (lpath,'a') as outfile2:
            outfile2.write('\n----------------------------------------------' + \
                           '-----------------------------\n\n')
            outfile2.write('Created new synthetic dataset ' + str(i+1) + \
                           ' for Variable 2.\n\n')   
        print('Running RadVel commands for new dataset ' + str(i+1) + '.\n')
        
        for j in range (5):
            print('Run ' + str(j+1) + '/5 for dataset ' + str(i+1) + '.\n')           
            with open (lpath,'a') as outfile2:
                outfile2.write('\n******************** Run ' + str(j+1) + '/5 for dataset ' + \
                               str(i+1) + '********************\n\n')
            Commands (lpath,no)
        
            with open (lpath,'a') as outfile2:
                outfile2.write('\nRan RadVel commands for run ' + str(j+1) + \
                               '/5 of new dataset ' + str(i+1) + '.\n')
                print('Retrieving post-MCMC parameters for run ' + str(j+1) + \
                      '/5 of new dataset ' + str(i+1) + ' and writing to summary files. \n')   
        
            Retrieve (no,gap[i],cov,err)
        
            with open (lpath,'a') as outfile2:
                outfile2.write('\nRetrieved post-MCMC parameters for run ' + \
                               str(j+1) + '/5 of new dataset ' + str(i+1) + \
                               ' and wrote to summary files. \n')
                outfile2.write('\nRun ' + str(j+1) + '/5 for dataset ' + \
                               str(i+1) + ' of Variable 2 completed.\n\n')
        

#############################################################################
#
# FUNCTION: RUN3
#
#############################################################################

def Run3 (dpath,lpath,err,ltime):
    '''
    Main function for Variable 3 to run all the subfunctions together in order, 
    log, and print outputs.
    
    Input: 1) dpath: Path to the synthetic data file.
           2) lpath: Path to the log file.
           3) err: Uncertainty on each synthetic data point, from GetErr function
           4) ltime: Time of the last real data.
    Output: None.
    '''
    
    # Run functions below in their order for different new datasets
    for i in range (len(cov)):
        print('Creating new synthetic dataset ' + str(i+1) + ' for Variable 3.\n')
            
        first,last,spacing = Case3(ltime,cov[i])
        
        if scat == 'jit':
            scatter = np.sqrt(jitter**2 + err**2)
        elif scat == 'rms':
            scatter = rms
        else:
            pass
        
        Data_Randomize(err,scatter,dpath,first,last,no)
        
        with open (lpath,'a') as outfile2:
            outfile2.write('\n----------------------------------------------' + \
                           '-----------------------------\n\n')
            outfile2.write('Created new synthetic dataset ' + str(i+1) + \
                           ' for Variable 3.\n\n')
        print('Running RadVel commands for new dataset ' + str(i+1) + '.\n')
        
        for j in range (5):
            print('Run ' + str(j+1) + '/5 for dataset ' + str(i+1) + '.\n')           
            with open (lpath,'a') as outfile2:
                outfile2.write('******************** Run ' + str(j+1) + '/5 for dataset ' + \
                               str(i+1) + '********************\n\n')
            Commands (lpath,no)
        
            with open (lpath,'a') as outfile2:
                outfile2.write('\nRan RadVel commands for run ' + str(j+1) + \
                               '/5 of new dataset ' + str(i+1) + '.\n')
                print('Retrieving post-MCMC parameters for run ' + str(j+1) + \
                      '/5 of new dataset ' + str(i+1) + ' and writing to summary files. \n')   
        
            Retrieve (no,spacing,cov[i],err)
        
            with open (lpath,'a') as outfile2:
                outfile2.write('\nRetrieved post-MCMC parameters for run ' + \
                               str(j+1) + '/5 of new dataset ' + str(i+1) + \
                               ' and wrote to summary files. \n')
                outfile2.write('\nRun ' + str(j+1) + '/5 for dataset ' + \
                               str(i+1) + ' of Variable 3 completed.\n\n')
                
    
#############################################################################
#
# FUNCTION: RUN4
#
#############################################################################

def Run4 (dpath,lpath,ltime):
    '''
    Main function for Variable 4 to run all the subfunctions together in order, 
    log, and print outputs.
    
    Input: 1) dpath: Path to the synthetic data file.
           2) lpath: Path to the log file.
           3) ltime: Time of the last real data.
    Output: None.
    '''
    
    # Run functions below in their order for different new datasets
    for i in range(len(eb)):
        print('Creating new synthetic dataset ' + str(i+1) + ' for Variable 4.\n')
    
        first,last,spacing = Case4(ltime)
        
        if scat == 'jit':
            scatter = np.sqrt(jitter**2 + eb[i]**2)
        elif scat == 'rms':
            scatter = rms
        else:
            pass
        
        Data_Randomize(eb[i],scatter,dpath,first,last,no)
        
        with open (lpath,'a') as outfile2:
            outfile2.write('\n----------------------------------------------' + \
                           '-----------------------------\n\n')
            outfile2.write('Created new synthetic dataset ' + str(i+1) + \
                           ' for Variable 4.\n\n')
        print('Running RadVel commands for new dataset ' + str(i+1) + '.\n')
        
        for j in range (5):
            print('Run ' + str(j+1) + '/5 for dataset ' + str(i+1) + '.\n')
            with open (lpath,'a') as outfile2:
                outfile2.write('******************** Run ' + str(j+1) + '/5 for dataset ' + \
                               str(i+1) + '********************\n\n')
            
            Commands (lpath,no)
            
            with open (lpath,'a') as outfile2:
                outfile2.write('\nRan RadVel commands for run ' + str(j+1) + \
                               '/5 of new dataset ' + str(i+1) + '.\n')
                print('Retrieving post-MCMC parameters for run ' + str(j+1) + \
                      '/5 of new dataset ' + str(i+1) + ' and writing to summary files. \n')
                
            Retrieve (no,spacing,cov,eb[i])
        
            with open (lpath,'a') as outfile2:
                outfile2.write('\nRetrieved post-MCMC parameters for run ' + \
                               str(j+1) + '/5 of new dataset ' + str(i+1) + \
                               ' and wrote to summary files. \n')
                outfile2.write('\nRun ' + str(j+1) + '/5 for dataset ' + \
                               str(i+1) + ' of Variable 4 completed.\n\n')
        
    
#############################################################################
#
# FUNCTION: GETERR
#
#############################################################################

def GetErr ():
    '''
    Function to calculate average velocity errors for synthetic data points.
    
    Input: None
    Output: 1) mean_err: Average velocity error.
            2) last_time: Last real data time.
    '''

    err = data['errvel']
    t = data['time']

    mean_err = np.mean(err)
    last_time = np.max(t)
    
    return mean_err, last_time

    
#############################################################################
#
# FUNCTION: CASE1
#
#############################################################################

def Case1 (last):
    '''
    Function to determine the starting and ending time stamps required for synthetic 
    data point creation, for Variable 1.
    
    Input: 1) Time of the last real data.
    Output: 1) startT: Starting time (1st data) for synthetic data points.
            2) endT: Ending time (last data) for synthetic data points.
            3) space: Gap info between real and synthetic data.
            4) coverage: Coverage info about synthetic data.
    '''

    # Determine data starting and ending time
    
    outerP = np.max(P)
    
    startT = ref1
    
    # Check if there's an end time specified, if not, simply the end of one period of outer most planet
    if endtime == 'Yes' or 'yes':
        endT = ref2
    elif endtime == 'No' or 'no':
        endT = startT + outerP
    else:
        print("'endtime' option not valid, please select 'Yes' or 'No'.")
        
    coverage = (endT-startT)/outerP          # coverage info in unit of outer planet's orbital period
    space = (startT - last)/outerP      # gap info in unit of outer planet's orbital period

    return startT, endT, space, coverage


#############################################################################
#
# FUNCTION: CASE2
#
#############################################################################

def Case2 (last,space):
    '''
    Function to determine the starting and ending time stamps required for synthetic 
    data point creation, for Variable 2.
    
    Input: 1) last: Time of the last real data point, from GetErr function.
           2) space: Time gap (in orbital period) since the last real data point 
              to create first synthetic data point.
    Output: 1) startT: Starting time (1st data) for synthetic data points.
            2) endT: Ending time (last data) for synthetic data points.
    '''
    
    outerP = np.max(P)
    
    # Start time based on the spacing input, and end time based on coverage input
    startT = last + space*outerP
    endT = startT + cov*outerP
    
    return startT, endT


#############################################################################
#
# FUNCTION: CASE3
#
#############################################################################

def Case3 (last,cover):
    '''
    Function to determine the starting and ending time stamps required for synthetic 
    data point creation, for Variable 3.
    
    Input: 1) last: Time of the last real data.
           2) cover: Coverage of synthetic data point.
    Output: 1) startT: Starting time (1st data) for synthetic data points.
            2) endT: Ending time (last data) for synthetic data points.
            3) space: Gap info between real and synth data.
    '''
    
    outerP = np.max(P)
    
    # Starting time the same as ref1, and ending time based on coverage inputs
    
    startT = ref1
    endT = startT + cover*outerP
    
    space = (startT - last)/outerP              # gap in outer planet's period

    return startT, endT, space


#############################################################################
#
# FUNCTION: CASE4
#
#############################################################################

def Case4 (last):
    '''
    Function to determine the starting and ending time stamps required for synthetic 
    data point creation, for Variable 4.
    
    Input: 1) last: Time of the last real data.
    Output: 1) startT: Starting time (1st data) for synthetic data points.
            2) endT: Ending time (last data) for synthetic data points.
    '''
    
    outerP = np.max(P)
    
    startT = ref1
    endT = startT + cov*outerP
    
    space = (startT - last)/outerP
    
    return startT, endT, space


#############################################################################
#
# FUNCTION: DATA_RANDOMIZE
#
#############################################################################

def Data_Randomize (error,scatter,dpath,start,end,number):
    '''
    Function to randomize the generated rv data points from Synth_Data to increase 
    both rv scatter and temporal sampling randomness to avoid false periodicity in a periodogram.
    
    Input: 1) error: Data error from GetErr function
           2) scatter: rms of the original fit
           3) dpath: Path to where the data file will be stored
           4) start: Starting time (1st data) of synthetic data point
           5) end: Finishing time (last data) of synthetic data point
           6) num: Number of data points
    Output: One text file that stores generated synthetic data with time, rv, and err 
            in the same directory as other data files and setup file.
    '''
    
    # randomize the temporal sampling when there're more than 2 data points
    if number > 2:
        add = int(number*1)    # extra number of data points added and subtracted later for randomness
        number += add
        rv, t_random = Synth_Data(start,end,number)
        
        rv = list(rv)
        t_random = list(t_random)
        
        # randomly exclude number of data points that are added earlier to create more
        # randomness in the temporal sampling
        ind = np.arange(1,len(rv)-1,1)     # velocity indices to be randomly selected except the first and the last data points
        exclude = random.sample(list(ind),add)       # list containing the indice of the data points to be excluded
        
        for item in exclude:
            rv[item] = 'NaN'
            t_random[item] = 'NaN'
        
        rv = [x for x in rv if x != 'NaN']
        t_random = [x for x in t_random if x != 'NaN']
    
        rv = np.array(rv)
        t_random = np.array(t_random)
    else:
        rv, t_random = Synth_Data(start,end,number)
    
    # Randonmize calculated total velocity data with the scatter provided using a Gaussian
    # filter.
    rv_random = np.random.normal(rv,scatter)
    
    # store the randomized synthetic rv data into a file
    df = pd.DataFrame()
    df['time'] = t_random
    df['rv'] = rv_random
    df['err'] = error
    
    df.to_csv(dpath,sep='\t',index=False, header=True)
    
    
#############################################################################
#
# FUNCTION: SYNTH_DATA
#
#############################################################################

def Synth_Data (start,end,number):
    '''
    Function to generate synthetic data points based on the inputs.
    
    Input: 1) start: Starting time (1st data) of synthetic data point
           2) end: Finishing time (last data) of synthetic data point
           3) number: Number of rv data points to generate
    Output: 1) vel: an array containing generated rv data points
            2) time_random: an array containing the time stamp information of the
               generated rv data points
    '''
    
    # Determine the actual times to create data points based on the starting time and 
    # number of data points. Actual time depends on the one full period of the outer
    # most planet.
    
    newtime = []              # Synthetic data points time in JD
    if number > 1:
        time_step = (end - start)/(number-1)
        for i in range (number):
            t = start + time_step*i
            newtime.append(t)
    else:
        t = start       # number = 1 case, put data at the start
        newtime.append(t)  
    
    # Slightly randomize the time of observation to avoid false periodicity to
    # be picked up by periodogram
    time_random = np.random.normal(newtime,3)   # sigma of 3 days
    time_random.sort()
    
    # Determine the total velocity values for all planets
    vel = 0
    for p in range (len(P)):
        # Determine orbital phase for each planet
        phi = []
        for t in time_random:
            ratio = (t-Tp[p])/P[p]
            if ratio > 1.0:    # when t is more than one period bigger than tp
                phase = ratio - int(ratio)
                phi.append(phase)
            elif 0 <= ratio <= 1:   # when t is within one period bigger than tp
                phi.append(ratio)
            elif -1 <= ratio < 0:   # when t is within one period less than tp
                phase = 1 + ratio
                phi.append(phase)
            else:    # when t is more than one period less than tp
                phase = 1 + (ratio - int(ratio))
                phi.append(phase)

        phi = np.array(phi)

        # Calculate the corresponding mean anomaly according to orbital phase
        m = 2*(np.pi)*phi/1.0              # mean anomaly

        # Calculate the eccentric anomaly according to mean anomaly
        eps = []
        for i in range(len(phi)):
            def func(x):
                return m[i] - x + e[p]*(np.sin(x))
            sol = root(func,1)
            eps.append(sol.x[0])
        eps = np.array(eps)

        # Calculate the true anomaly according to eccentric anomaly
        nu = []
        for i in range(len(phi)):
            v = np.arccos(((np.cos(eps[i])) - e[p])/(1 - e[p]*(np.cos(eps[i]))))
            if eps[i] <= np.pi:
                nu.append(v)
            else:
                v = np.pi + abs(np.pi - v)
                nu.append(v)
        nu = np.array(nu)

        # Calculate the velocity according to true anomaly
        V = V_0 + K[p]*(np.cos(w[p]+nu)+e[p]*(np.cos(w[p])))
        vel = vel + V

    vel = np.array(vel)
    
    return vel, time_random


#############################################################################
#
# FUNCTION: COMMANDS
#
#############################################################################

def Commands (lpath,number):
    '''
    Function to run subprocesses of radvel command line commands in the script.
    
    Input: 1) lpath: Path to the logfile.
           2) number: number of synthetic data points
    Output: None.
    '''
    
    # Determine if the radvel run is with no new data or with new data, they have
    # different file names
    if number == 0:
        file = str(name[:-3]) + '_0.py'
    else:
        file = name
    
    # Running radvel commands
    a = subprocess.check_output(['radvel','fit', '-s', file])
    a = str(a)
    alog = a.split('\\n')
    with open(lpath,'a') as outfile:
        outfile.write('\n')
        for i in range(len(alog)-2):
            if i != 0:
                outfile.write(alog[i] + '\n')
            else:
                pass
            
    b = subprocess.check_output(['radvel','mcmc','-s',file])
    b = str(b)
    blog = b.split('\\n')
    with open(lpath,'a') as outfile:
        outfile.write('\n')
        for i in range(len(blog)-1):
            if i != 0:
                outfile.write(blog[i] + '\n')
            else:
                pass
    
    c = subprocess.check_output(['radvel','derive','-s',file])
    c = str(c)
    clog = c.split('\\n')
    with open(lpath,'a') as outfile:
        outfile.write('\n')
        for i in range(len(clog)-1):
            if i != 0:
                outfile.write(clog[i] + '\n')
            else:
                pass
    
    d = subprocess.check_output(['radvel','plot','-t','rv','-s',file])
    d = str(d)
    dlog = d.split('\\n')
    with open(lpath,'a') as outfile:
        outfile.write('\n')
        for i in range(len(dlog)-1):
            if i != 0:
                outfile.write(dlog[i] + '\n')
            else:
                pass


#############################################################################
#
# FUNCTION: RETRIEVE
#
#############################################################################

def Retrieve (number,space,coverage,error):
    '''
    Function to retrieve the values of period and their uncertainties after each
    run, and writes to a separate file 'Period_Summary.txt' for summary.
    
    Input: 1) number: Number of data points for each run.
           2) spacing: Spaces between 1st synthetic data and last real data,
              in unit of outer planet's orbital period. 'NaN' if not using this
              variable.
           3) coverage: Duration of synthetic data points, in unit of outer
              planet's orbital period. 'NaN' if not using this variable.
           4) error: instrumental error of the data point in m/s.
    Output: A summary file containing the period and its uncertainty values.
    '''
    
    # Determine if the radvel run is with no new data or with new data, they have
    # different file names
    if number == 0:
        file = str(name[:-3]) + '_0.py'
    else:
        file = name
    
    # Path of the file that store all post mcmc parameters
    
    folder = str(file[:-3])
    directory = os.path.dirname(path1)
    parampath = directory + '/' + folder + '/' + folder + '_post_summary.csv'
    derivedpath = directory + '/' + folder + '/' + folder + '_derived_quantiles.csv'
    
    # Read the param and derived parm files
    param = pd.read_csv(parampath)
    derived = pd.read_csv(derivedpath)
    
    # Name of the columns in radvel summay files
    param_name = set(param.columns)
    derived_name = set(derived.columns)
    
    for item in sumfile:
        sumpath = directory + '/' + item + '_Summary.txt'
        
        # Define how the dataframe that stores final parameters and error will look like
        df2 = pd.DataFrame()
        
        for i in range (len(P)):
            par = item + str(i+1)
            if par in param_name:
                var = np.array([param[par][1]])
                upper = param[par][2]
                lower = param[par][0]
                var_err_up = np.array([upper - var])
                var_err_low = np.array([var - lower])
                var_err_avg = (var_err_up + var_err_low)/2.0
                df2['var' + str(i+1)] = var
                df2['var_err_up' + str(i+1)] = var_err_up
                df2['var_err_low' + str(i+1)] = var_err_low
                df2['var_err_avg' + str(i+1)] = var_err_avg
            elif par in derived_name:
                var = np.array([derived[par][1]])
                upper = derived[par][2]
                lower = derived[par][0]
                var_err_up = np.array([upper - var])
                var_err_low = np.array([var - lower])
                var_err_avg = (var_err_up + var_err_low)/2.0
                df2['var' + str(i+1)] = var
                df2['var_err_up' + str(i+1)] = var_err_up
                df2['var_err_low' + str(i+1)] = var_err_low
                df2['var_err_avg' + str(i+1)] = var_err_avg
            else:
                print(item + 'is not a valid summary file name.')
        
        df2['No'] = number
        df2['Gap'] = space
        df2['Cov'] = coverage
        df2['ErrBar'] = error
        
        # Write to summary file
        df2.to_csv(sumpath,sep='\t',index=False,header=False,mode='a')

    
##############################################################################
##############################################################################

##############################################################################
# 
# Command Line Run Section
#
##############################################################################

'''
Run the above script from the command line. It only calls the Execute () function.
'''

if __name__ == '__main__':
    Execute ()

