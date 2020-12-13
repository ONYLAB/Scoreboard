"""Functions for plotting data."""

import scipy.interpolate
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.pylab as pl
import matplotlib.dates as mdates
from .scores import *
import os

def plotdifferencescdfpdf(Scoreboard,model_target,figuresdirectory):
    
    model_targets = ['Case', 'Death']
    if model_target not in model_targets:
        raise ValueError("Invalid sim type. Expected one of: %s" % model_targets)  
        
    if model_target == 'Case':
        titlelabel= 'Weekly Incidental Cases'
    elif model_target == 'Death':
        titlelabel= 'Cumulative Deaths'
        
    plt.figure(figsize=(4, 2.5), dpi=180, facecolor='w', edgecolor='k')
    plt.hist(Scoreboard['prange']-Scoreboard['sumpdf'],bins=50)
    plt.xlabel("Difference between integrated pdf and given cdf")
    plt.title('US COVID-19 ' + titlelabel)
    print('===========================')
    print('Maximum % conversion error:')
    print(100*max(Scoreboard['prange']-Scoreboard['sumpdf']))
    plt.savefig(figuresdirectory+'/'+model_target+'_'+'diffcdfpdf.svg',
            bbox_inches = 'tight',
            dpi=300)    

def plotUSCumDeaths(US_deaths,figuresdirectory) -> None:
    plt.figure(figsize=(4, 2.5), dpi=180, facecolor='w', edgecolor='k')
    plt.plot(US_deaths.DateObserved,US_deaths.Deaths)
    plt.xticks(rotation=45)
    plt.title('US Cumulative Deaths')
    plt.ylabel('Deaths')
    plt.savefig(figuresdirectory+'/USDeaths.png', 
                bbox_inches = 'tight',
                dpi=300) 
    plt.savefig(figuresdirectory+'/USDeaths.svg', 
                bbox_inches = 'tight',
                dpi=300)     
    
def plotUSIncCases(US_cases,figuresdirectory) -> None:
    plt.figure(figsize=(4, 2.5), dpi=180, facecolor='w', edgecolor='k')
    plt.plot(US_cases.DateObserved,US_cases.Cases)
    plt.xticks(rotation=45)
    plt.title('US Weekly Incidental Cases')
    plt.ylabel('Cases')
    plt.savefig(figuresdirectory+'/USCases.png', 
                bbox_inches = 'tight',
                dpi=300)
    plt.savefig(figuresdirectory+'/USCases.svg', 
                bbox_inches = 'tight',
                dpi=300)    

def perdelta(start, end, delta):
    """Generate a list of datetimes in an 
    interval for plotting purposes (date labeling)
    Args:
        start (date): Start date.
        end (date): End date.
        delta (date): Variable date as interval.
    Yields:
        Bunch of dates 
    """    
    
    curr = start
    while curr < end:
        yield curr
        curr += delta

def numberofteamsincovidhub(FirstForecasts,figuresdirectory)->None:
    fig = plt.figure(num=None, figsize=(6, 4), dpi=120, facecolor='w', edgecolor='k')
    FirstForecasts['forecast_date']= pd.to_datetime(FirstForecasts['forecast_date'])
    plt.plot(FirstForecasts['forecast_date'],FirstForecasts['cumnumteams'])
    plt.xticks(rotation=70)
    plt.ylabel('Total Number of Modeling Teams')
    plt.xlabel('First Date of Entry')
    plt.title('Number of Teams in Covid-19 Forecast Hub Increases')
    plt.fmt_xdata = mdates.DateFormatter('%m-%d')
    plt.savefig(figuresdirectory+'/numberofmodels.png', 
                bbox_inches = 'tight',
                dpi=300)
    plt.savefig(figuresdirectory+'/numberofmodels.svg', 
            bbox_inches = 'tight',
            dpi=300)
    plt.show(fig)
        
def plotallscoresdist(Scoreboard,figuresdirectory,model_target) -> None:
    
    model_targets = ['Case', 'Death']
    if model_target not in model_targets:
        raise ValueError("Invalid sim type. Expected one of: %s" % model_targets)  
        
    if model_target == 'Case':
        filelabel = 'INCCASE'
        titlelabel= 'Weekly Incidental Cases'
    elif model_target == 'Death':
        filelabel = 'CUMDEATH'
        titlelabel= 'Cumulative Deaths'
        
    fig = plt.figure(figsize=(6, 4), dpi=300, facecolor='w', edgecolor='k')
    Scoreboard.plot.scatter(x='delta', y='score', marker='.')
    plt.xlabel('N-Days Forward Forecast')
    plt.title(titlelabel + ' Forecasts')
    plt.savefig(figuresdirectory+'/'+filelabel+'_'+'ScoreVSx-Days_Forward_Forecast.png',
                bbox_inches = 'tight',
                dpi=300)
    plt.savefig(figuresdirectory+'/'+filelabel+'_'+'ScoreVSx-Days_Forward_Forecast.svg',
                bbox_inches = 'tight',
                dpi=300)    
    plt.show(fig)

    fig = plt.figure(figsize=(6, 4), dpi=300, facecolor='w', edgecolor='k')
    Scoreboard.plot.scatter(x='deltaW', y='score', marker='.')
    plt.xlabel('N-Weeks Forward Forecast')
    plt.title(titlelabel + ' Forecasts')
    plt.savefig(figuresdirectory+'/'+filelabel+'_'+'ScoreVSx-Weeks_Forward_Forecast.png', 
                bbox_inches = 'tight',
                dpi=300)
    plt.savefig(figuresdirectory+'/'+filelabel+'_'+'ScoreVSx-Weeks_Forward_Forecast.svg', 
                bbox_inches = 'tight',
                dpi=300)    
    plt.show(fig)
    
    fig = plt.figure(figsize=(6, 4), dpi=300, facecolor='w', edgecolor='k')
    binwidth = 1
    Scoreboard.delta.hist(bins=range(min(Scoreboard.delta), max(Scoreboard.delta) + binwidth, binwidth))
    #plt.xlim(4, 124)
    plt.title(titlelabel + ' Forecasts')
    plt.xlabel('N-Days Forward Forecast')
    plt.ylabel('Number of forecasts made')   
    plt.xticks(np.arange(min(Scoreboard.delta), max(Scoreboard.delta)+1, 2.0))
    plt.xticks(rotation=90)
    plt.grid(b=None)
    plt.savefig(figuresdirectory+'/'+filelabel+'_x-Days_Forward_Forecast_Hist.png', 
                bbox_inches = 'tight',
                dpi=300)
    plt.show(fig)

    fig = plt.figure(figsize=(6, 4), dpi=300, facecolor='w', edgecolor='k')
    Scoreboard.deltaW.hist(bins=range(1, int(Scoreboard['deltaW'].max()) + binwidth, binwidth))
    #plt.xlim(0, 22)
    plt.title(titlelabel + ' Forecasts')
    plt.xlabel('N-Weeks Forward Forecast')
    plt.ylabel('Number of forecasts made')
    plt.xticks(np.arange(min(Scoreboard.deltaW), max(Scoreboard.deltaW)+1, 1.0))
    plt.xticks(rotation=90)
    plt.grid(b=None)
    plt.savefig(figuresdirectory+'/'+filelabel+'_x-Weeks_Forward_Forecast_Hist.png', 
                bbox_inches = 'tight',
                dpi=300)
    plt.savefig(figuresdirectory+'/'+filelabel+'_x-Weeks_Forward_Forecast_Hist.svg', 
                bbox_inches = 'tight',
                dpi=300)    
    plt.show(fig)
        
def plotlongitudinal(Actual,Scoreboard,scoretype,WeeksAhead,curmod,figuresdirectory) -> None:
    """Plots select model predictions against actual data longitudinally
    Args:
        Actual (pd.DataFrame): The actual data
        Scoreboard (pd.DataFrame): The scoreboard dataframe
        scoretype (str): "Cases" or "Deaths"
        WeeksAhead (int): Forecasts from how many weeks ahead
        curmod (str): Name of the model whose forecast will be shown
    Returns:
        None 
    """    
    
    Scoreboardx = Scoreboard[Scoreboard['deltaW']==WeeksAhead].copy()
    Scoreboardx.sort_values('target_end_date',inplace=True)
    Scoreboardx.reset_index(inplace=True)  
    plt.figure(num=None, figsize=(14, 8), dpi=80, facecolor='w', edgecolor='k')
    models = Scoreboardx['model'].unique()
    colors = pl.cm.jet(np.linspace(0,1,len(models)))
    i = 0

    dates = Scoreboardx[Scoreboardx['model']==curmod].target_end_date
    PE = Scoreboardx[Scoreboardx['model']==curmod].PE
    CIlow = Scoreboardx[Scoreboardx['model']==curmod].CILO
    CIhi = Scoreboardx[Scoreboardx['model']==curmod].CIHI

    modcol = (colors[i].tolist()[0],
              colors[i].tolist()[1],
              colors[i].tolist()[2])

    plt.plot(dates,PE,color=modcol,label=curmod)
    plt.fill_between(dates, CIlow, CIhi, color=modcol, alpha=.1)

    plt.plot(Actual['DateObserved'],Actual[scoretype],color='k',linewidth=3.0)    
    plt.ylim([(Actual[scoretype].min())*0.6, (Actual[scoretype].max())*1.4])
    plt.ylabel('US Cumulative '+scoretype, fontsize=18)
    plt.xlabel('Date', fontsize=18)
    plt.xticks(rotation=45, fontsize=13)
    plt.yticks(fontsize=13)
    plt.fmt_xdata = mdates.DateFormatter('%m-%d')
    plt.title(curmod+': '+str(WeeksAhead)+'-week-ahead Forecasts')
    plt.savefig(figuresdirectory+'/'+scoretype+'_'+curmod+'_'+str(WeeksAhead)+'wk.svg', 
                bbox_inches = 'tight',
                dpi=300)        

def plotlongitudinalUNWEIGHTED(Actual,Scoreboard,scoretype,numweeks,figuresdirectory) -> None:
    """Plots select model predictions against actual data longitudinally
    Args:
        Actual (pd.DataFrame): The actual data
        Scoreboard (pd.DataFrame): The scoreboard dataframe
        scoretype (str): "Cases" or "Deaths"
        numweeks (int): number of weeks to plot
    Returns:
        None 
    """
        
    scoretypes = ['Cases', 'Deaths']
    if scoretype not in scoretypes:
        raise ValueError("Invalid sim type. Expected one of: %s" % scoretypes)  

    if scoretype == 'Cases':
        titlelabel= 'weekly incidental cases'
    elif scoretype == 'Deaths':
        titlelabel= 'cumulative deaths'          
        
    numweeks += 1
    labelp = 'Average Unweighted Forecasts'
    colors = pl.cm.jet(np.linspace(0,1,numweeks))
    plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    i = 0
    
    for WeeksAhead in range(1, numweeks):    
        
        i += 1
        Scoreboardx = Scoreboard[Scoreboard['deltaW']==WeeksAhead].copy()
        Scoreboardx.sort_values('target_end_date',inplace=True)
        Scoreboardx.reset_index(inplace=True)

        MerdfPRED = Scoreboardx.copy()   
        MerdfPRED = (MerdfPRED.groupby(['target_end_date'],
                                       as_index=False)[['CILO','PE','CIHI']].agg(lambda x: np.mean(x)))

        dates = MerdfPRED['target_end_date']
        PE = MerdfPRED['PE']
        CIlow = MerdfPRED['CILO']
        CIhi = MerdfPRED['CIHI']

        modcol = (colors[i].tolist()[0],
                  colors[i].tolist()[1],
                  colors[i].tolist()[2])

        plt.plot(dates,PE,color=modcol,label=str(i)+ ' weeks-ahead')
        #plt.fill_between(dates, CIlow, CIhi, color=modcol, alpha=.1)        

    plt.plot(Actual['DateObserved'],Actual[scoretype],color='k',linewidth=3.0)    
    plt.ylim([(Actual[scoretype].min())*0.6, (Actual[scoretype].max())*1.1])
    plt.ylabel('US '+titlelabel, fontsize=18)
    plt.xlabel('Target End Date', fontsize=18)
    plt.xticks(rotation=45, fontsize=13)
    plt.yticks(fontsize=13)
    plt.fmt_xdata = mdates.DateFormatter('%m-%d')
    #plt.title(labelp, fontsize=18)
    plt.legend(loc="upper left",labelspacing=.9)
    lims = plt.gca().get_xlim()
    plt.savefig(figuresdirectory+'/'+scoretype+'_Forecasts.svg',
                dpi=300,bbox_inches = 'tight')    

    plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    i = 0
    
    for WeeksAhead in range(1, numweeks):    
        
        i += 1
        Scoreboardx = Scoreboard[Scoreboard['deltaW']==WeeksAhead].copy()
        Scoreboardx.sort_values('target_end_date',inplace=True)
        Scoreboardx.reset_index(inplace=True)

        MerdfPRED = Scoreboardx.copy()   
        MerdfPRED = (MerdfPRED.groupby(['target_end_date'],
                                       as_index=False)[['score']].agg(lambda x: np.nanmean(x)))

        dates = MerdfPRED['target_end_date']
        scores = MerdfPRED['score']

        modcol = (colors[i].tolist()[0],
                  colors[i].tolist()[1],
                  colors[i].tolist()[2])

        plt.plot(dates,scores,color=modcol,label=str(i)+ ' weeks-ahead')

    plt.ylabel('Average Scores for US '+titlelabel, fontsize=18)
    plt.xlabel('Target End Date', fontsize=18)
    plt.xticks(rotation=45, fontsize=13)
    plt.yticks(fontsize=13)
    plt.fmt_xdata = mdates.DateFormatter('%m-%d')
    #plt.title('Scores for '+labelp, fontsize=18)
    plt.legend(loc="upper left",labelspacing=.9)
    plt.gca().set_xlim(xmin=lims[0], xmax=lims[1])
    plt.savefig(figuresdirectory+'/'+scoretype+'_Average_Forward_Scores.svg',
                dpi=300,bbox_inches = 'tight')

def plotlongitudinalALL(Actual,Scoreboard,scoretype,WeeksAhead,figuresdirectory) -> None:
    """Plots all predictions against actual data longitudinally
    Args:
        Actual (pd.DataFrame): The actual data
        Scoreboard (pd.DataFrame): The scoreboard dataframe
        scoretype (str): "Cases" or "Deaths"
        WeeksAhead (int): Forecasts from how many weeks ahead
    Returns:
        None 
    """    
    scoretypes = ['Cases', 'Deaths']
    if scoretype not in scoretypes:
        raise ValueError("Invalid sim type. Expected one of: %s" % scoretypes)  
        
    if scoretype == 'Cases':
        titlelabel= 'weekly incidental cases'
    elif scoretype == 'Deaths':
        titlelabel= 'cumulative deaths'         
    
    Scoreboardx = Scoreboard[Scoreboard['deltaW']==WeeksAhead].copy()
    Scoreboardx.sort_values('target_end_date',inplace=True)
    Scoreboardx.reset_index(inplace=True)  
    plt.figure(num=None, figsize=(14, 8), dpi=80, facecolor='w', edgecolor='k')
    models = Scoreboardx['model'].unique()
    colors = pl.cm.jet(np.linspace(0,1,len(models)))
    i = 0
    for curmod in models:

        dates = Scoreboardx[Scoreboardx['model']==curmod].target_end_date
        PE = Scoreboardx[Scoreboardx['model']==curmod].PE
        CIlow = Scoreboardx[Scoreboardx['model']==curmod].CILO
        CIhi = Scoreboardx[Scoreboardx['model']==curmod].CIHI

        modcol = (colors[i].tolist()[0],
                  colors[i].tolist()[1],
                  colors[i].tolist()[2])

        plt.plot(dates,PE,color=modcol,label=curmod)
        plt.fill_between(dates, CIlow, CIhi, color=modcol, alpha=.1)
        i = i+1

    plt.plot(Actual['DateObserved'],Actual[scoretype],color='k',linewidth=3.0)    
    plt.ylim([(Actual[scoretype].min())*0.6, (Actual[scoretype].max())*1.4])
    plt.ylabel('US '+titlelabel, fontsize=18)
    plt.xlabel('Date', fontsize=18)
    plt.xticks(rotation=45, fontsize=13)
    plt.yticks(fontsize=13)
    plt.fmt_xdata = mdates.DateFormatter('%m-%d')
    plt.title(str(WeeksAhead)+'-week-ahead Forecasts', fontsize=18)
    plt.savefig(figuresdirectory+'/'+scoretype+'_All.svg',
                dpi=300,bbox_inches = 'tight')    

def plotgroupsTD(Scoreboard, modeltypes, figuresdirectory, model_target) -> None:
    """Generates modeltype-based score plots in time (Forecast Date)
    Args:
        Scoreboard (pd.DataFrame): Scoreboard
        modeltypes (pd.DataFrame): End date.
        figuresdirectory (str): Directory to save in
        model_target (str): 'Case' or 'Death'
    Returns:
        None 
    """        
    
    model_targets = ['Case', 'Death']
    if model_target not in model_targets:
        raise ValueError("Invalid sim type. Expected one of: %s" % model_targets)  

    if model_target == 'Case':
        filelabel = 'INCCASE'
        titlelabel= 'weekly incidental cases'
    elif model_target == 'Death':
        filelabel = 'CUMDEATH'
        titlelabel= 'cumulative deaths'        
    
    (MerdfPRED,pivMerdfPRED) = givePivotScoreTARGET(Scoreboard,modeltypes)
    
    dateticks = list(perdelta(pivMerdfPRED.index[0] - timedelta(days=14), 
                              pivMerdfPRED.index[-1] + timedelta(days=14), 
                              timedelta(days=7)))
    selectmodels = modeltypes['modeltype'].unique().tolist()
    for selectmodel in selectmodels:
        listmods = modeltypes[modeltypes['modeltype']==selectmodel].model.tolist()
        colors = pl.cm.jet(np.linspace(0,1,len(listmods)))
        plt.figure(figsize=(12, 6), dpi=180, facecolor='w', edgecolor='k')
        
        for i in range(len(listmods)):
            if listmods[i] in pivMerdfPRED.columns:
                if ~pivMerdfPRED[listmods[i]].isnull().all():                
                    pivMerdfPRED[listmods[i]].dropna().plot(color=(colors[i].tolist()[0],
                                                      colors[i].tolist()[1],
                                                      colors[i].tolist()[2]),
                                              marker='o')

        plt.legend(loc=(1.04,0),labelspacing=.9)
        plt.title(selectmodel+' models: Average Forward Scores')
        plt.ylabel('Time-averaged score for ' + titlelabel)
        plt.xlabel('Target End Date')
        plt.ylim([pivMerdfPRED.min().min()-1, pivMerdfPRED.max().max()+1])
        plt.xlim([dateticks[0],dateticks[-1]])
        custom_tick_labels = map(lambda x: x.strftime('%b %d'), dateticks)
        plt.xticks(dateticks,custom_tick_labels)
        plt.xticks(rotation=45)
        plt.savefig(figuresdirectory+'/'+filelabel+'_Average_Forward_Scores_'+selectmodel+'models.svg',
                    dpi=300,bbox_inches = 'tight')


def plotgroupsmodelweek(Scoreboard: pd.DataFrame, modeltypes: pd.DataFrame, 
               figuresdirectory: str, numweeks: int, model_target: str) -> None:
    """Generates modeltype-based score plots in time (Forecast Date)
    Args:
        pivMerdfPRED (pd.DataFrame): Start date.
        modeltypes (pd.DataFrame): model types
        figuresdirectory (str): direcotry to save in
        numweeks (int): number of weeks ahead forecast
        model_target (str): 'Case' or 'Death'
    Returns:
        None 
    """    

    model_targets = ['Case', 'Death']
    if model_target not in model_targets:
        raise ValueError("Invalid sim type. Expected one of: %s" % model_targets)  

    if model_target == 'Case':
        filelabel = 'INCCASE'
        titlelabel= 'weekly incidental cases'
    elif model_target == 'Death':
        filelabel = 'CUMDEATH'
        titlelabel= 'cumulative deaths'       
    
    Scoreboardx = Scoreboard[Scoreboard['deltaW']==numweeks].copy()
    (MerdfPRED,pivMerdfPRED) = givePivotScoreTARGET(Scoreboardx,modeltypes)
    
    dateticks = list(perdelta(pivMerdfPRED.index[0] - timedelta(days=21), 
                              pivMerdfPRED.index[-1] + timedelta(days=21), 
                              timedelta(days=14)))
    selectmodels = modeltypes['modeltype'].unique().tolist()
    for selectmodel in selectmodels:
        models = modeltypes[modeltypes['modeltype']==selectmodel].model.tolist()
        pivMerdfPRED[selectmodel] = pivMerdfPRED.filter(items=models).mean(axis=1)         
        
    listmods = selectmodels
    colors = pl.cm.jet(np.linspace(0,1,len(listmods)))
    plt.figure(figsize=(6, 6), dpi=180, facecolor='w', edgecolor='k')
    for i in range(len(listmods)):
        if listmods[i] in pivMerdfPRED.columns:
            if ~pivMerdfPRED[listmods[i]].isnull().all():                
                pivMerdfPRED[listmods[i]].dropna().plot(color=(colors[i].tolist()[0],
                                                  colors[i].tolist()[1],
                                                  colors[i].tolist()[2]),
                                          marker='o')
    plt.title(str(numweeks)+'-week ahead forecasts')
    plt.legend(loc='lower left',labelspacing=.9)
    plt.ylabel('Model-averaged score for ' + titlelabel)
    plt.xlabel('Target End Date')
    #plt.ylim([pivMerdfPRED.min().min()-1, pivMerdfPRED.max().max()+1])
    plt.xlim([dateticks[0],dateticks[-1]])
    custom_tick_labels = map(lambda x: x.strftime('%b %d'), dateticks)
    plt.xticks(dateticks,custom_tick_labels)
    plt.xticks(rotation=45)
    plt.savefig(figuresdirectory+'/'+filelabel+'_Model_averaged_scores_wk'+str(numweeks)+'models.svg',
                dpi=300,bbox_inches = 'tight')
    
    dateticks = list(perdelta(pivMerdfPRED.index[0] - timedelta(days=14), 
                          pivMerdfPRED.index[-1] + timedelta(days=14), 
                          timedelta(days=7)))   
    for selectmodel in selectmodels:
        listmods = modeltypes[modeltypes['modeltype']==selectmodel].model.tolist()
        colors = pl.cm.jet(np.linspace(0,1,len(listmods)))
        plt.figure(figsize=(12, 6), dpi=180, facecolor='w', edgecolor='k')
        
        for i in range(len(listmods)):
            if listmods[i] in pivMerdfPRED.columns:
                if ~pivMerdfPRED[listmods[i]].isnull().all():                
                    pivMerdfPRED[listmods[i]].dropna().plot(color=(colors[i].tolist()[0],
                                                      colors[i].tolist()[1],
                                                      colors[i].tolist()[2]),
                                              marker='o')

        plt.legend(loc=(1.04,0),labelspacing=.9)
        plt.title(selectmodel+' models')
        plt.ylabel('Score for ' + titlelabel)
        plt.xlabel('Target End Date')
        plt.ylim([pivMerdfPRED.min().min()-1, pivMerdfPRED.max().max()+1])
        plt.xlim([dateticks[0],dateticks[-1]])
        custom_tick_labels = map(lambda x: x.strftime('%b %d'), dateticks)
        plt.xticks(dateticks,custom_tick_labels)
        plt.xticks(rotation=45)
        plt.savefig(figuresdirectory+'/'+filelabel+'_wk_'+str(numweeks)+selectmodel+'models.svg',
                    dpi=300,bbox_inches = 'tight')   
    
        
def plotgroupsFD(Scoreboard: pd.DataFrame, modeltypes: pd.DataFrame, 
               figuresdirectory: str, numweeks: int, model_target: str) -> None:
    """Generates modeltype-based score plots in time (Forecast Date)
    Args:
        pivMerdfPRED (pd.DataFrame): Start date.
        modeltypes (pd.DataFrame): model types
        figuresdirectory (str): direcotry to save in
        numweeks (int): number of weeks ahead forecast
        model_target (str): 'Case' or 'Death'
    Returns:
        None 
    """    

    model_targets = ['Case', 'Death']
    if model_target not in model_targets:
        raise ValueError("Invalid sim type. Expected one of: %s" % model_targets)  

    if model_target == 'Case':
        filelabel = 'INCCASE'
        titlelabel= 'weekly incidental cases'
    elif model_target == 'Death':
        filelabel = 'CUMDEATH'
        titlelabel= 'cumulative deaths'       
    
    Scoreboardx = Scoreboard[Scoreboard['deltaW']==numweeks].copy()
    (MerdfPRED,pivMerdfPRED) = givePivotScoreFORECAST(Scoreboardx,modeltypes)
    
    dateticks = list(perdelta(pivMerdfPRED.index[0] - timedelta(days=14), 
                              pivMerdfPRED.index[-1] + timedelta(days=14), 
                              timedelta(days=7)))
    selectmodels = modeltypes['modeltype'].unique().tolist()
    for selectmodel in selectmodels:
        listmods = modeltypes[modeltypes['modeltype']==selectmodel].model.tolist()
        colors = pl.cm.jet(np.linspace(0,1,len(listmods)))
        plt.figure(figsize=(12, 6), dpi=180, facecolor='w', edgecolor='k')
        
        for i in range(len(listmods)):
            if listmods[i] in pivMerdfPRED.columns:
                if ~pivMerdfPRED[listmods[i]].isnull().all():                
                    pivMerdfPRED[listmods[i]].dropna().plot(color=(colors[i].tolist()[0],
                                                      colors[i].tolist()[1],
                                                      colors[i].tolist()[2]),
                                              marker='o')

        plt.legend(loc=(1.04,0),labelspacing=.9)
        plt.title(selectmodel+' models: '+ str(numweeks) +' wk ahead Scores')
            
        plt.ylabel('Score for '+str(numweeks)+' wk ahead '+titlelabel)
        plt.xlabel('Forecast Date')
        plt.ylim([pivMerdfPRED.min().min()-1, pivMerdfPRED.max().max()+1])
        plt.xlim([dateticks[0],dateticks[-1]])
        custom_tick_labels = map(lambda x: x.strftime('%b %d'), dateticks)
        plt.xticks(dateticks,custom_tick_labels)
        plt.xticks(rotation=45)
        plt.savefig(figuresdirectory+'/'+str(numweeks)+'Week/'+filelabel+'_Forward_Scores_'+selectmodel+'models.svg',
                    dpi=300,
                   bbox_inches = 'tight')        
        
        
def plotscoresvstimeW(Scoreboard,Weeks):
    plt.figure()
    rslt_df = Scoreboard.loc[Scoreboard['deltaW'] == Weeks] 
    df = rslt_df.pivot(index='forecast_date', columns='model', values='score')
    df = df.astype(float)

    plt.figure(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')

    for i in range(len(df.columns)):
        df[df.columns[i]].dropna().interpolate(method='polynomial', order=2).plot(kind='line',marker='o')

    plt.legend(loc=(1.04,0))
    plt.title(str(Weeks)+'-Week Forward Scores')
    plt.ylabel('Score for N wk ahead incident cases')
    plt.xlabel('Date Forecast Made')        
    
    
def plotscoresvstime(Scoreboard,Days):
    plt.figure()
    rslt_df = Scoreboard.loc[Scoreboard['delta'] == Days] 
    df = rslt_df.pivot(index='forecast_date', columns='model', values='score')
    df = df.astype(float)

    plt.figure(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')

    for i in range(len(df.columns)):
        df[df.columns[i]].dropna().interpolate(method='linear').plot(kind='line',marker='o')

    plt.legend(loc=(1.04,0))
    plt.title(str(Days)+'-Day Forward Scores')
    plt.ylabel('Score for N wk ahead incident cases')
    plt.xlabel('Date Forecast Made')    