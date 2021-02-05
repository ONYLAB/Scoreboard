"""Functions for getting and analyzing data."""

import scipy.interpolate
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import os
from urllib.parse import quote
from . import paths


def read_observed(kind, writetocsv: bool = False, use_cache: bool = False)-> pd.DataFrame:
    """Read US cumulative cases or deaths from covid19-forecast-hub
    Args:
        use_cache (bool), if True use previously written csv file
        writetocsv (bool), if True writes pd to US_deaths.csv
    Returns:
        US_deaths (pd dataframe) 
    """    
    assert kind in ['cases', 'deaths']
    file_path = paths.data_dir / ('US_%s.csv' % kind)
    if use_cache:
        weekly = pd.read_csv(file_path, index_col=0, parse_dates=['DateObserved'])
        return weekly
    
    #Read Observed
    address = "raw.githubusercontent.com/reichlab/covid19-forecast-hub/master/data-truth/truth-Cumulative %s.csv"
    address = "http://" + quote(address % kind.title())
    df = pd.read_csv(address,
                        dtype = {'location': str}, parse_dates=['date'],
                        na_values = ['NA', 'no info', '.'])
    df = df[(df['location_name'] == 'US')]
    df.drop(columns=['location', 'location_name'], inplace=True)
    df.columns = ['DateObserved', kind.title()]
    df.reset_index(drop=True, inplace=True)
    
    #Convert from daily to weekly measured from Sun-Sat
    weekly = df.iloc[np.arange(df[df["DateObserved"]=="2020-01-25"].index[0],
                                             len(df), 7)].copy()
    weekly[kind.title()] = weekly[kind.title()].diff()
    weekly.reset_index(drop=True, inplace=True)
    
    if writetocsv == True:
        weekly.to_csv(file_path) 
        
    return weekly


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