"""Functions for plotting data."""

import scipy.interpolate
import pandas as pd
from pathlib import Path
import numpy as np
from datetime import date, datetime, timedelta
from tempfile import NamedTemporaryFile
import matplotlib.pyplot as plt
from matplotlib.image import imread
import matplotlib.ticker as ticker
import matplotlib.pylab as pl
import matplotlib.dates as mdates
from .scores import *
from .__init__ import figures_dir, data_dir
import os

def save_figures(name):
    global figures_dir
    figures_dir = Path(figures_dir)
    (figures_dir / name).parent.mkdir(parents=True, exist_ok=True)  # Create all necessary parent dirs
    plt.savefig(figures_dir / (name + '.svg'), 
                bbox_inches = 'tight',
                dpi=300)


def plotdifferencescdfpdf(Scoreboard, model_target):
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
    save_figures(model_target+'_diffcdfpdf') 


def plotUSCumDeaths(US_deaths) -> None:
    plt.figure(figsize=(4, 2.5), dpi=180, facecolor='w', edgecolor='k')
    plt.plot(US_deaths.DateObserved,US_deaths.Deaths)
    plt.xticks(rotation=45)
    plt.title('US Cumulative Deaths')
    plt.ylabel('Deaths')
    save_figures('USDeaths')   

    
def plotUSIncCases(US_cases) -> None:
    plt.figure(figsize=(4, 2.5), dpi=180, facecolor='w', edgecolor='k')
    plt.plot(US_cases.DateObserved,US_cases.Cases)
    plt.xticks(rotation=45)
    plt.title('US Weekly Incidental Cases')
    plt.ylabel('Cases')
    save_figures('USCases')   

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
    

def numberofteamsincovidhub(FirstForecasts, figures_dir)->None:
    fig = plt.figure(num=None, figsize=(6, 4), dpi=120, facecolor='w', edgecolor='k')
    FirstForecasts['forecast_date']= pd.to_datetime(FirstForecasts['forecast_date'])
    plt.plot(FirstForecasts['forecast_date'],FirstForecasts['cumnumteams'])
    plt.xticks(rotation=70)
    plt.ylabel('Total Number of Modeling Teams')
    plt.xlabel('First Date of Entry')
    plt.title('Number of Teams in Covid-19 Forecast Hub Increases')
    plt.fmt_xdata = mdates.DateFormatter('%m-%d')
    save_figures('numberofmodels') 
    plt.show(fig)


def plotallscoresdist(Scoreboard, model_target) -> None:
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
    save_figures(filelabel+'_ScoreVSx-Days_Forward_Forecast')   
    plt.show(fig)

    fig = plt.figure(figsize=(6, 4), dpi=300, facecolor='w', edgecolor='k')
    Scoreboard.plot.scatter(x='deltaW', y='score', marker='.')
    plt.xlabel('N-Weeks Forward Forecast')
    plt.title(titlelabel + ' Forecasts')
    save_figures(filelabel+'_ScoreVSx-Weeks_Forward_Forecast') 
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
    save_figures(filelabel+'_x-Days_Forward_Forecast_Hist') 
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
    save_figures(filelabel+'_x-Weeks_Forward_Forecast_Hist')
    plt.show(fig)
    return 0


def plotlongitudinal(Actual, Scoreboard, scoretype, WeeksAhead, curmodlist) -> None:
    """Plots select model predictions against actual data longitudinally
    Args:
        Actual (pd.DataFrame): The actual data
        Scoreboard (pd.DataFrame): The scoreboard dataframe
        scoretype (str): "Cases" or "Deaths"
        WeeksAhead (int): Forecasts from how many weeks ahead
        curmodlist (list): List of name of the model whose forecast will be shown
    Returns:
        None 
    """    
    
    Scoreboardx = Scoreboard[Scoreboard['deltaW']==WeeksAhead].copy()
    Scoreboardx.sort_values('target_end_date',inplace=True)
    Scoreboardx.reset_index(inplace=True)  
    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    models = Scoreboardx['model'].unique()
    colors = pl.cm.jet(np.linspace(0,1,len(curmodlist)))
    i = 0
    
    for curmod in curmodlist:
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
    plt.ylabel('US Cumulative '+scoretype, fontsize=18)
    plt.xlabel('Date', fontsize=18)
    plt.xticks(rotation=45, fontsize=13)
    plt.yticks(fontsize=13)
    plt.fmt_xdata = mdates.DateFormatter('%m-%d')
    plt.title(str(WeeksAhead)+'-week-ahead Forecasts', fontsize=18)
    plt.legend(loc="upper left",labelspacing=.9, fontsize=16)

    save_figures(scoretype+'_'+''.join(curmodlist)+'_'+str(WeeksAhead)+'wk')
    

def plotlongitudinalUNWEIGHTED(Actual, Scoreboard, scoretype, numweeks) -> None:
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
    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    i = 0
    
    for WeeksAhead in range(1, numweeks):    
        
        i += 1
        Scoreboardx = Scoreboard[Scoreboard['deltaW']==WeeksAhead].copy()
        Scoreboardx.sort_values('target_end_date',inplace=True)
        Scoreboardx.reset_index(inplace=True)

        MerdfPRED = Scoreboardx.copy()   
        MerdfPRED = (MerdfPRED.groupby(['target_end_date'],
                                       as_index=False)[['CILO','PE','CIHI']].agg(lambda x: np.median(x)))

        dates = MerdfPRED['target_end_date']
        PE = MerdfPRED['PE']
        CIlow = MerdfPRED['CILO']
        CIhi = MerdfPRED['CIHI']

        modcol = (colors[i].tolist()[0],
                  colors[i].tolist()[1],
                  colors[i].tolist()[2])
        if i == 1:
            plt.plot(dates,PE,color=modcol,label=str(i)+ ' week ahead')
            #plt.fill_between(dates, CIlow, CIhi, color=modcol, alpha=.1)
        else:
            plt.plot(dates,PE,color=modcol,label=str(i)+ ' weeks ahead')

    plt.plot(Actual['DateObserved'],Actual[scoretype],color='k',linewidth=3.0)    
    plt.ylim([(Actual[scoretype].min())*0.6, (Actual[scoretype].max())*1.1])
    plt.ylabel('US '+titlelabel, fontsize=18)
    plt.xlabel('Target End Date', fontsize=18)
    plt.xticks(rotation=45, fontsize=13)
    plt.yticks(fontsize=13)
    plt.fmt_xdata = mdates.DateFormatter('%m-%d')
    plt.title('Medians of all forecast point estimates', fontsize=18)
    plt.legend(loc="upper left",labelspacing=.9, fontsize=16)
    lims = plt.gca().get_xlim()
    save_figures(scoretype+'_Forecasts.svg')   

    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    i = 0
    
    for WeeksAhead in range(1, numweeks):    
        
        i += 1
        Scoreboardx = Scoreboard[Scoreboard['deltaW']==WeeksAhead].copy()
        Scoreboardx.sort_values('target_end_date',inplace=True)
        Scoreboardx.reset_index(inplace=True)

        MerdfPRED = Scoreboardx.copy()   
        MerdfPRED = (MerdfPRED.groupby(['target_end_date'],
                                       as_index=False)[['score']].agg(lambda x: np.median(x)))

        dates = MerdfPRED['target_end_date']
        scores = MerdfPRED['score']

        modcol = (colors[i].tolist()[0],
                  colors[i].tolist()[1],
                  colors[i].tolist()[2])
        if i == 1:
            plt.plot(dates,scores,color=modcol,label=str(i)+ ' week ahead')
        else:
            plt.plot(dates,scores,color=modcol,label=str(i)+ ' weeks ahead')
    
    plt.title('US '+titlelabel, fontsize=18)
    plt.ylabel('Median of Forecast Scores', fontsize=18)
    plt.xlabel('Target End Date', fontsize=18)
    plt.xticks(rotation=45, fontsize=13)
    plt.yticks(fontsize=13)
    plt.fmt_xdata = mdates.DateFormatter('%m-%d')
    #plt.title('Scores for '+labelp, fontsize=18)
    #plt.legend(loc="upper left",labelspacing=.9, fontsize=16)
    plt.gca().set_xlim(xmin=lims[0], xmax=lims[1])
#     plt.ylim([Scoreboard[Scoreboard['score']!=np.NINF].score.min(),
#               Scoreboard[Scoreboard['score']!=np.NINF].score.max()])

    save_figures(scoretype+'_Average_Forward_Scores.svg')


def plotlongitudinalALL(Actual, Scoreboard, scoretype, WeeksAhead) -> None:
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
    save_figures(scoretype+'_All')    


def plotgroupsTD(Scoreboardx, modeltypes, model_target) -> None:
    """Generates modeltype-based score plots in time (Forecast Date)
    Args:
        Scoreboard (pd.DataFrame): Scoreboard
        modeltypes (pd.DataFrame): End date.
        model_target (str): 'Case' or 'Death'
    Returns:
        None 
    """
    Scoreboard = Scoreboardx.copy()
    Scoreboard.replace([np.inf, -np.inf], np.nan,inplace=True)
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
                              timedelta(days=14)))
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
        save_figures(filelabel+'_Average_Forward_Scores_'+selectmodel+'models')


def plotgroupsmodelweek(Scoreboardx: pd.DataFrame, modeltypes: pd.DataFrame, 
                        numweeks: int, model_target: str) -> None:
    """Generates modeltype-based score plots in time (Forecast Date)
    Args:
        pivMerdfPRED (pd.DataFrame): Start date.
        modeltypes (pd.DataFrame): model types
        numweeks (int): number of weeks ahead forecast
        model_target (str): 'Case' or 'Death'
    Returns:
        None 
    """
    Scoreboard = Scoreboardx.copy()
    Scoreboard.replace([np.inf, -np.inf], np.nan,inplace=True)
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
    
    date0 = datetime.strptime('2020-04-25','%Y-%m-%d')
    dateticks = list(perdelta(date0, 
                              Scoreboardx.target_end_date.max() + timedelta(days=28), 
                              timedelta(days=28)))
    
    selectmodels = modeltypes['modeltype'].unique().tolist()
    for selectmodel in selectmodels:
        models = modeltypes[modeltypes['modeltype']==selectmodel].model.tolist()
        pivMerdfPRED[selectmodel] = pivMerdfPRED.filter(items=models).median(axis=1)         
        
    listmods = selectmodels
    colors = pl.cm.jet(np.linspace(0,1,len(listmods)))
    plt.figure(figsize=(4, 4), dpi=300, facecolor='w', edgecolor='k')
    for i in range(len(listmods)):
        if listmods[i] in pivMerdfPRED.columns:
            if ~pivMerdfPRED[listmods[i]].isnull().all():                
                pivMerdfPRED[listmods[i]].dropna().plot(color=(colors[i].tolist()[0],
                                                  colors[i].tolist()[1],
                                                  colors[i].tolist()[2]),
                                          marker='o')
    plt.title(str(numweeks)+'-week ahead forecasts')
    plt.legend(loc='best',labelspacing=.9)
    plt.ylabel('Model-averaged score for ' + titlelabel)
    plt.xlabel('Target End Date')
    plt.xlim([date0,dateticks[-1]])
    custom_tick_labels = map(lambda x: x.strftime('%Y-%m'), dateticks)
    plt.xticks(dateticks,custom_tick_labels)
    plt.xticks(rotation=45)
    plt.minorticks_off()
    set_size(plt.gcf(), (4, 4))
    save_figures(filelabel+'_Model_averaged_scores_wk'+str(numweeks)+'models')
    
    date0 = datetime.strptime('2020-04-25','%Y-%m-%d')
    dateticks = list(perdelta(date0, 
                              Scoreboardx.target_end_date.max() + timedelta(days=28), 
                              timedelta(days=28)))
    
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
        save_figures(filelabel+'_wk_'+str(numweeks)+selectmodel+'models')   
    
        
def plotgroupsFD(Scoreboardx: pd.DataFrame, modeltypes: pd.DataFrame, 
                 numweeks: int, model_target: str) -> None:
    """Generates modeltype-based score plots in time (Forecast Date)
    Args:
        pivMerdfPRED (pd.DataFrame): Start date.
        modeltypes (pd.DataFrame): model types
        numweeks (int): number of weeks ahead forecast
        model_target (str): 'Case' or 'Death'
    Returns:
        None 
    """
    Scoreboard = Scoreboardx.copy()
    Scoreboard.replace([np.inf, -np.inf], np.nan,inplace=True)
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
                              timedelta(days=14)))
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
        plt.title(selectmodel+' models: '+ str(numweeks) +' week ahead Scores')
            
        plt.ylabel('Score for '+str(numweeks)+' wk ahead '+titlelabel)
        plt.xlabel('Forecast Date')
        plt.ylim([pivMerdfPRED.min().min()-1, pivMerdfPRED.max().max()+1])
        plt.xlim([dateticks[0],dateticks[-1]])
        custom_tick_labels = map(lambda x: x.strftime('%b %d'), dateticks)
        plt.xticks(dateticks,custom_tick_labels)
        plt.xticks(rotation=45)
        save_figures(str(numweeks)+'Week/'+filelabel+'_Forward_Scores_'+selectmodel+'models')
        
        
def plotTD(Scoreboardx, WeeksAhead, listmods) -> None:
    """Generates modeltype-based score plots in time (Forecast Date)
    Args:
        Scoreboard (pd.DataFrame): Scoreboard
    Returns:
        None 
    """
    Scoreboard = Scoreboardx[Scoreboardx['deltaW']==WeeksAhead].copy()
    Scoreboard['model'] = Scoreboard['model'].str.slice(0,21)
    Scoreboard.replace([np.inf, -np.inf], np.nan,inplace=True)

    if 'cases' in Scoreboardx.columns:
        filelabel = 'INCCASE'
        titlelabel= 'weekly incidental cases'
    else:
        filelabel = 'CUMDEATH'
        titlelabel= 'cumulative deaths'        
    
    
    MerdfPRED = Scoreboard.copy()
    MerdfPRED = (MerdfPRED.groupby(['model','target_end_date'],
                                            as_index=False)[['delta','score']].agg(lambda x: list(x)))

    MerdfPRED['median'] = MerdfPRED.apply(lambda row : np.median(row['score']), axis = 1) 
    
    pivMerdfPRED = MerdfPRED.pivot(index='target_end_date', columns='model', values='median') 
    
    date0 = datetime.strptime('2020-04-25','%Y-%m-%d')
    dateticks = list(perdelta(date0, 
                              Scoreboardx.target_end_date.max() + timedelta(days=28), 
                              timedelta(days=28)))

    colors = pl.cm.jet(np.linspace(0,1,len(listmods)))
    plt.figure(figsize=(6, 6), dpi=300, facecolor='w', edgecolor='k')

    for i in range(len(listmods)):
        if listmods[i] in pivMerdfPRED.columns:
            if ~pivMerdfPRED[listmods[i]].isnull().all():                
                pivMerdfPRED[listmods[i]].dropna().plot(color=(colors[i].tolist()[0],
                                                  colors[i].tolist()[1],
                                                  colors[i].tolist()[2]),
                                          marker='o')

    plt.legend(loc='best',labelspacing=.5,fontsize=9)
    plt.title(str(WeeksAhead)+'-week-ahead Scores',fontsize=18)
    plt.ylabel('Scores for ' + titlelabel,fontsize=18)
    plt.xlabel('Target End Date',fontsize=18)
    plt.xlim([date0,dateticks[-1]])
    custom_tick_labels = map(lambda x: x.strftime('%Y-%m'), dateticks)
    plt.xticks(dateticks,custom_tick_labels)
    plt.xticks(rotation=45)
    plt.minorticks_off()
    set_size(plt.gcf(), (6, 6))
    save_figures(str(WeeksAhead)+'Week/'+filelabel+'_top10models')    

def get_size(fig, dpi=100):
    with NamedTemporaryFile(suffix='.png') as f:
        fig.savefig(f.name, bbox_inches='tight', dpi=dpi)
        height, width, _channels = imread(f.name).shape
        return width / dpi, height / dpi

def set_size(fig, size, dpi=100, eps=1e-2, give_up=2, min_size_px=10):
    target_width, target_height = size
    set_width, set_height = target_width, target_height # reasonable starting point
    deltas = [] # how far we have
    while True:
        fig.set_size_inches([set_width, set_height])
        actual_width, actual_height = get_size(fig, dpi=dpi)
        set_width *= target_width / actual_width
        set_height *= target_height / actual_height
        deltas.append(abs(actual_width - target_width) + abs(actual_height - target_height))
        if deltas[-1] < eps:
            return True
        if len(deltas) > give_up and sorted(deltas[-give_up:]) == deltas[-give_up:]:
            return False
        if set_width * dpi < min_size_px or set_height * dpi < min_size_px:
            return False      
    
def plotscoresvstimeW(Scoreboardx, Weeks):
    plt.figure()
    Scoreboard = Scoreboardx.copy()
    Scoreboard.replace([np.inf, -np.inf], np.nan,inplace=True)
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
    
    
def plotscoresvstime(Scoreboardx, Days):
    plt.figure()
    Scoreboard = Scoreboardx.copy()
    Scoreboard.replace([np.inf, -np.inf], np.nan,inplace=True)
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