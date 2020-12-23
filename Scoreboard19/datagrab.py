"""Functions for getting and analyzing data."""

import scipy.interpolate
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import os


def readobserveddeaths(writetocsv: bool = False)-> pd.DataFrame:
    """Read US cumulative deaths from covid19-forecast-hub
    Args:
        writetocsv (bool), if True writes pd to US_deaths.csv
    Returns:
        US_deaths (pd dataframe) 
    """    
    
    #Read Observed Deaths
    address = "https://raw.githubusercontent.com/reichlab/covid19-forecast-hub/master/data-truth/truth-Cumulative%20Deaths.csv"
    dfOBS = pd.read_csv(address,
                        dtype = {'location': str},parse_dates=['date'],
                        na_values = ['NA', 'no info', '.'])
    US_deaths = dfOBS.copy()    
    US_deaths = US_deaths[(dfOBS['location_name'] == 'US')]
    US_deaths.drop(columns=['location', 'location_name'],inplace=True)
    US_deaths.columns = ['DateObserved', 'Deaths']
    US_deaths.reset_index(drop=True, inplace=True)
    
    if writetocsv == True:
        US_deaths.to_csv('../Data/US_deaths.csv') 
        
    return US_deaths


def readobservedcases(writetocsv: bool = False)-> pd.DataFrame:
    """Read US cumulative cases from covid19-forecast-hub
    Args:
        writetocsv (bool), if True writes pd to US_cases.csv
    Returns:
        WeeklyUS_cases (pd dataframe) 
    """    
    
    #Read Observed Cases
    address = 'https://raw.githubusercontent.com/reichlab/covid19-forecast-hub/master/data-truth/truth-Cumulative%20Cases.csv'
    dfOBS = pd.read_csv(address,
                        dtype = {'location': str},parse_dates=['date'],
                        na_values = ['NA', 'no info', '.'])
    US_cases = dfOBS.copy()    
    US_cases = US_cases[(dfOBS['location_name'] == 'US')]
    US_cases = US_cases.drop(columns=['location', 'location_name'])
    US_cases.columns = ['DateObserved', 'Cases']
    US_cases.reset_index(inplace=True,drop=True)
    
    #Convert from daily to  weekly measured from Sun-Sat
    WeeklyUS_cases = US_cases.iloc[np.arange(US_cases[US_cases["DateObserved"]=="2020-01-25"].index[0],
                                             len(US_cases),7)].copy()
    WeeklyUS_cases['Cases'] = WeeklyUS_cases.Cases.diff()
    WeeklyUS_cases.reset_index(drop=True, inplace=True)
    
    if writetocsv == True:
        US_cases.to_csv('../Data/US_cases.csv') 
        
    return WeeklyUS_cases


def getmodeltypes(Scoreboard, quiet=False)-> pd.DataFrame:
    uniques = Scoreboard.model.unique()
    modeltypes = pd.read_csv('../Data/modeltypes.dat')
    modeltypes['model'] = modeltypes['model'].str.strip()
    modeltypes['modeltype'] = modeltypes['modeltype'].str.strip()
    
    if not quiet:
        print('================================')
        print('Unique models in the scoreboard:')
        for i in range(0,len(uniques)):
            print(str(i)+'. '+uniques[i])
        print('========================================================')
        print("Models in the latest Scoreboard that are not yet in modeltypes.dat:")
        print(np.setdiff1d(np.sort(Scoreboard.model.drop_duplicates()), modeltypes.model))
        print("Edit modeltypes.dat accordingly")
    return modeltypes