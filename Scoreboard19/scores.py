"""Functions for preparing scores."""

from tqdm import tqdm
import scipy.interpolate
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
"""Functions for preparing scores."""

from tqdm import tqdm
import scipy.interpolate
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
import os
import os
from . import scoresplots, datagrab


def getleaderboard(Scoreboard, WeeksAhead, leaderboardin, quiet=False):
    Scoreboard4 = Scoreboard[Scoreboard['deltaW']==WeeksAhead].copy()
    
    scoresframe = (Scoreboard4.groupby(['model'],as_index=False)[['score']].agg(lambda x: np.median(x))).sort_values(by=['score'], ascending=False)    
    scoresframe.reset_index(inplace=True,drop=True)
    scoresframe = scoresframe.rename(columns={'score':'median of past scores'})    
    
    ranksframe = (Scoreboard4.groupby(['model'],as_index=False)[['rank']].agg(lambda x: np.mean(x))).sort_values(by=['rank'], ascending=True)    
    ranksframe.reset_index(inplace=True,drop=True)
    ranksframe = ranksframe.rename(columns={'rank':'average of past rankings'})
    
    leaderboard = scoresframe.merge(ranksframe, left_on=['model'], right_on=['model']).copy()
    
    leaderboard['deltaW'] = WeeksAhead
    
    auxstr = ' as of ' + Scoreboard['target_end_date'].max().strftime('%Y-%m-%d')
    if 'cases' in Scoreboard.columns:
        if not quiet:
            print('Leaderboard for ' + str(WeeksAhead) + '-week-ahead weekly incidental case forecasts ' + auxstr)
        leaderboard['forecasttype'] = 'cases'
    else:
        if not quiet:
            print('Leaderboard for ' + str(WeeksAhead) + '-week-ahead cumulative deaths forecasts' + auxstr)
        leaderboard['forecasttype'] = 'deaths'
    leaderboard['asofdate'] = Scoreboard['target_end_date'].max().strftime('%Y-%m-%d')    
    leaderboard = pd.concat([leaderboardin, leaderboard], sort=False)
    
    return leaderboard


def giveweightsformodels(Scoreboardx, datepred, weekcut):
    #str datecut e.g. '2020-07-01'
    #Make sure we take only one prediction per model
    
    datecut = datetime.strptime(datepred,'%Y-%m-%d') - timedelta(days=(weekcut-1)*7)    
    
    Scoreboard = Scoreboardx[Scoreboardx['deltaW']==weekcut].copy()
    
    Scoreboardearly = Scoreboard[Scoreboard['target_end_date']<datecut].copy()

    #Scoreboardearly.dropna(subset=['score'],inplace=True)
    Scoreboardearly.reset_index(inplace=True)    
    
    listofavailablemodels = Scoreboardearly['model'].unique().tolist()
    
    scoresframe = (Scoreboardearly.groupby(['model'],as_index=False)[['score']].agg(lambda x: np.median(x))).sort_values(by=['score'], ascending=False)    
    scoresframe.reset_index(inplace=True,drop=True)
    scoresframe = scoresframe.rename(columns={'score':'pastscores'})
    
#     ranksframe = (Scoreboardearly.groupby(['model'],as_index=False)[['rank']].agg(lambda x: np.mean(x))).sort_values(by=['rank'], ascending=False)    
#     ranksframe.reset_index(inplace=True,drop=True)
#     ranksframe = ranksframe.rename(columns={'rank':'pastranks'})    
    
    return (scoresframe,listofavailablemodels,Scoreboardearly)


def getscoresforweightedmodels(Scoreboardx,datepred,weekcut,case,runtype):
    """Generates all model weighted/unweighted ensembles
    Args:
        scoreboardx (pd.DataFrame): The scoreboard
        datepred (str): Start date on which first ensemble will be formed
        case (str): 'Case' or 'Death'
        weekcut (int): number of weeks ahead forecast ensemble formation
        runtype (str): weighted or unweighted ensemble
    Returns:
        scoreboard (pd.DataFrame): scoreboard with the added ensemble for nwk
    """     
    #str datecut e.g. '2020-07-01'
    #Make sure we take only one prediction per model
    
    Scoreboard = Scoreboardx.copy() 
    
    datepredindate = datetime.strptime(datepred,'%Y-%m-%d')
    datecut = datepredindate - timedelta(days=(weekcut-1)*7)  

    [scoresframe,listofavailablemodels,Scoreboardearly] = giveweightsformodels(Scoreboard,datepred,weekcut)
    predday = Scoreboard[(Scoreboard['target_end_date']==datepred)&(Scoreboard['deltaW']==weekcut)].copy()

#     #We exclude COVIDhub:ensemble from our own ensemble as we know it is an ensemble of the models here
#     predday.drop(predday[predday['model'] == 'COVIDhub:ensemble'].index, inplace = True)
    predday.drop(predday[predday['model'] == 'FDANIH:Sweight'].index, inplace = True) 
    predday.drop(predday[predday['model'] == 'FDANIH:Sunweight'].index, inplace = True) 
    
    preddaymerged = predday.merge(scoresframe, left_on=['model'], right_on=['model']).copy()
    #preddaymerged = tempframe.merge(ranksframe, left_on=['model'], right_on=['model']).copy()
    
    if runtype=='weighted':
        preddaymerged['weights'] = np.exp(preddaymerged['pastscores'].astype(np.float64)/2)
        modelname='FDANIH:Sweight'
    elif runtype=='unweighted':
        preddaymerged['weights'] = 1
        modelname='FDANIH:Sunweight'
        
    sumweights = preddaymerged['weights'].sum()
    preddaymerged['weights'] = preddaymerged['weights']/sumweights
    
    if preddaymerged.empty:
        print('DataFrame is empty!')
        print(datepred)
    else:
        (qso,vso) = givescoreweightedforecast(preddaymerged,case)
        #plt.plot(qso,vso)

        if case=='Cases':
            new_row = {'model':modelname,
                       'target_end_date':datepredindate,
                       'forecast_date':datecut,
                      'delta':weekcut*7,
                      'deltaW':weekcut,
                      'proper':True,
                      'quantile':qso,
                      'value':vso,
                       'CILO':min(vso),
                       'PE':np.median(vso),
                       'CIHI':max(vso),
                      'cases':Scoreboard[(Scoreboard['target_end_date']==datepred)]['cases'].mean()}
        elif case=='Deaths':
            new_row = {'model':modelname,
                       'target_end_date':datepredindate,
                       'forecast_date':datecut,
                      'delta':weekcut*7,
                      'deltaW':weekcut,
                      'proper':True,
                      'quantile':qso,
                      'value':vso,
                      'CILO':min(vso),
                      'PE':np.median(vso),
                      'CIHI':max(vso),
                      'deaths':Scoreboard[(Scoreboard['target_end_date']==datepred)]['deaths'].mean()}

        Scoreboard = Scoreboard.append(new_row, ignore_index=True)

        Index = len(Scoreboard)-1
        result = giveqandscore(Scoreboard,Index)
        Scoreboard.iloc[Index, Scoreboard.columns.get_loc('score')] = result[0]
        Scoreboard.iloc[Index, Scoreboard.columns.get_loc('sumpdf')] = result[1]
        Scoreboard.iloc[Index, Scoreboard.columns.get_loc('prange')] = result[2]
        Scoreboard.iloc[Index, Scoreboard.columns.get_loc('p')] = result[3]
    
    #leaderboard = preddaymerged[['model', 'deltaW', 'pastscores', 'pastranks','weights']].copy()
    #leaderboard['datecut'] = datecut
    #leaderboard['target_end_date'] = datepred
        
    return Scoreboard


def givescoreweightedforecast(Scoreboardx,case):

    Scoreboard = Scoreboardx.copy()
    if case=='Cases':
        mylist = [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]
    elif case=='Deaths':
        mylist = [0.01,0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
         0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]
        
    vso = [0] * len(mylist)

    for Index in range(0,len(Scoreboard)):    

        qs = Scoreboard.iloc[Index, Scoreboard.columns.get_loc('quantile')]
        vs = Scoreboard.iloc[Index, Scoreboard.columns.get_loc('value')]
        wmodel = Scoreboard.iloc[Index, Scoreboard.columns.get_loc('weights')]

        for i in mylist:
            loc = qs.index(i)
            vso[loc] += wmodel * vs[loc]

        qso = mylist
        
    return (qso,vso)


def getweightedmodelalldates(scoreboardx, startdate, case, nwk, runtype):
    """Generates all model weighted/unweighted ensembles for an nwk
    Args:
        scoreboardx (pd.DataFrame): The scoreboard
        startdate (str): Start date on which first ensemble will be formed
        case (str): 'Case' or 'Death'
        nwk (int): number of weeks ahead forecast ensemble formation
        runtype (str): weighted or unweighted ensemble
    Returns:
        scoreboard (pd.DataFrame): scoreboard with the added ensemble for nwk
    """        
    #e.g. startdate '2020-08-01'
    #case e.g. Cases or Deaths
    scoreboard = scoreboardx.copy()
    #cumleaderboard = pd.DataFrame(columns = ['model', 'deltaW', 'pastscores', 'pastranks','weights', 'datecut', 'target_end_date'])
    daterange = pd.date_range(start=startdate, end=pd.to_datetime('today'),freq='W-SAT')
    for datepred in daterange:    
        #(scoreboard,leaderboard) = getscoresforweightedmodels(scoreboard,datepred.strftime('%Y-%m-%d'),nwk,case,runtype)
        #cumleaderboard = pd.concat([cumleaderboard, leaderboard], sort=False)
        scoreboard = getscoresforweightedmodels(scoreboard,datepred.strftime('%Y-%m-%d'),nwk,case,runtype)   
        
    #cumleaderboard.reset_index(inplace=True,drop=True)
    return scoreboard


def getscoreboard(groundtruth,model_target,otpfile='ScoreboardDataCases.pkl')-> pd.DataFrame:
    """Generates primary scores for all model competition entries
    Args:
        groundtruth (pd.DataFrame): The observed data
        model_target (str): 'Case' or 'Death'
        otpfile (str): Name of the scoreboard .pkl output file
    Returns:
        FirstForecasts (pd.DataFrame): check the forecast upload chronology
    """    
    model_targets = ['Case', 'Death']
    if model_target not in model_targets:
        raise ValueError("Invalid sim type. Expected one of: %s" % model_targets)   
    
    #Read the predictions file 
    dfPREDx = pd.read_csv('../Data/all_dataONY.csv',
                         na_values = ['NA', 'no info', '.'], parse_dates=True)
    dfPREDx.drop_duplicates(subset=None, keep = 'first', inplace = True)
    
    #Get the chronology of team entries to the competition
    FirstForecasts = dfPREDx.sort_values('forecast_date').drop_duplicates(subset=['team'], keep='first').copy()
    FirstForecasts['teamexist'] = 1
    FirstForecasts['cumnumteams'] = FirstForecasts['teamexist'].cumsum()
        
    if model_target == 'Case':
        dfPRED = dfPREDx[dfPREDx['target'].str.contains('inc case')].copy()
    elif model_target == 'Death':
        dfPRED = dfPREDx[dfPREDx['target'].str.contains('cum death')].copy()      
    
    dfPRED.reset_index(inplace=True)
    
    #New dataframe with merged values - this forms a single forecast unit (quantiles&corresponding values)
    MerdfPRED = dfPRED.copy()   
    MerdfPRED = (MerdfPRED.groupby(['team','model','forecast_date','target_end_date'],
                                        as_index=False)[['quantile','value']].agg(lambda x: list(x)))
    
    #Develop the ultimate scoreboard including the corresponding observed data
    MerdfPRED['target_end_date'] = pd.to_datetime(MerdfPRED['target_end_date'])
    groundtruth['DateObserved'] = pd.to_datetime(groundtruth['DateObserved'])    
    
    Scoreboard = (MerdfPRED.merge(groundtruth, left_on=['target_end_date'], right_on=['DateObserved'])).copy()    
    Scoreboard.drop(columns=['DateObserved'],inplace=True)
    
    if 'Cases' in Scoreboard.columns:
        Scoreboard.rename(columns={'Cases':'cases'},inplace=True)
        mylist = [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]
    if 'Deaths' in Scoreboard.columns:
        Scoreboard.rename(columns={'Deaths':'deaths'},inplace=True)
        mylist = [0.01,0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
         0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]
    
    Scoreboard['target_end_date']= pd.to_datetime(Scoreboard['target_end_date']) 
    Scoreboard['forecast_date']= pd.to_datetime(Scoreboard['forecast_date'])
    Scoreboard.insert(3,'delta',(Scoreboard.target_end_date-Scoreboard.forecast_date).dt.days)
    new = Scoreboard['model'].copy()
    Scoreboard['team']= Scoreboard['team'].str.cat(new, sep =":")
    Scoreboard.drop(columns=['model'],inplace=True)
    Scoreboard.rename(columns={'team':'model'},inplace=True)
    Scoreboard['deltaW'] = np.ceil(Scoreboard.delta/7)
    
    Scoreboard['proper'] = ''
    for Index in tqdm(range(0,len(Scoreboard))): 
        modellist = Scoreboard['quantile'].iloc[Index]
        proper = all(item in modellist for item in mylist)        
        Scoreboard.iloc[Index, Scoreboard.columns.get_loc('proper')] = proper
    
    #Calculate the scores and merge those with the Scoreboard dataframe
    Scoreboard['score'] = ''
    Scoreboard['sumpdf'] = ''
    Scoreboard['prange'] = ''
    Scoreboard['p'] = ''

    for Index in tqdm(range(0,len(Scoreboard))):    
        result = giveqandscore(Scoreboard,Index)
        Scoreboard.iloc[Index, Scoreboard.columns.get_loc('score')] = result[0]
        Scoreboard.iloc[Index, Scoreboard.columns.get_loc('sumpdf')] = result[1]
        Scoreboard.iloc[Index, Scoreboard.columns.get_loc('prange')] = result[2]
        Scoreboard.iloc[Index, Scoreboard.columns.get_loc('p')] = result[3]

    Scoreboard['CIHI']=pd.DataFrame(Scoreboard['value'].to_list()).max(axis=1)
    Scoreboard['CILO']=pd.DataFrame(Scoreboard['value'].to_list()).min(axis=1)
    Scoreboard['PE']=pd.DataFrame(Scoreboard['value'].to_list()).median(axis=1)
    
    #Scoreboard.replace([np.inf, -np.inf], np.nan,inplace=True)
    #Scoreboard.dropna(inplace=True)
    Scoreboardx = Scoreboard.sort_values('forecast_date').drop_duplicates(subset=['model', 'target_end_date','deltaW'], keep='last').copy()
    Scoreboardx.reset_index(drop=True,inplace=True)
    Scoreboardx['scorecontr']=np.exp(Scoreboardx['score'].astype(np.float64)/2)
    
    Scoreboardx.to_pickle(otpfile)
    
    return FirstForecasts
    

def cdfpdf(df,Index,dV,withplot: bool = False, figuresdirectory: str = ''):
    '''Get pdf from cdf using Scoreboard dataset.
    
    Args:
        df (pandas pd): Scoreboard dataframe
        Index (int): Scoreboard Row index selection (model, forecast date, 
                    target date, quantiles and values)
        dV (int): x-axis grid point delta
        withplot (bool, optional): If True, plot cdf and pdf. Defaults to False.
    Returns:
        xout (int array): x-axis values
        pdfout (float array): pdf at xout
        sum(pdfout) (float): integrated version of calculated pdf
        max(cdf)-min(cdf) (float): the cdf - sum(pdfout) should be close to max(cdf)-min(cdf)
    
    '''    
    
    #Get quantiles and values from the dataset
    mydf = pd.DataFrame(list(zip(df['quantile'].iloc[Index], df['value'].iloc[Index])), 
       columns =['cdf', 'dp'])
    
    #If duplicate forecasts exist for any particular day, then use the average
    mydf = mydf.groupby('cdf', as_index=False).mean() 
    mydf.sort_values(by=['cdf'],inplace=True)
    cdf = mydf.cdf.to_numpy() 
    dp = mydf.dp.to_numpy().round() #number of cases or deaths 
    
    #create grid x-axis
    dpgrid = np.arange(np.round(min(dp))+0.5,np.round(max(dp))+0.5,1)   
        
    if len(dpgrid)<3:
        #Some predictions have an extremely sharp - impulse-like distributions
        xout=[np.nan,np.round(min(dp))]
        pdfout=[np.nan,(max(cdf)-min(cdf))]
        sumpdfout = (max(cdf)-min(cdf))
    else:        
        #Take care of CASE:~dirac/discontinuous CDF
        u, c = np.unique(dp, return_counts=True)
        dup = u[c > 1]
        
        while len(dup)>0:
            for i in range(len(dup)):
                dupindex = np.where(dp==dup[i])
                if len(dupindex[0])>0:
                    for i in range(len(dupindex[0])):
                        dp[dupindex[0][i]] = dp[dupindex[0][i]]-(len(dupindex[0])-1)+(2*i)
          
            u, c = np.unique(dp, return_counts=True)
            dup = u[c > 1]

        dp = np.sort(dp)
        
        #recreate grid
        dpgrid = np.arange(np.round(min(dp))+0.5,np.round(max(dp))+0.5,1)          
        
        #Do PCHIP interpolation
        pchip_obj1 = scipy.interpolate.PchipInterpolator(dp, cdf)
        
        #Get PDF based on the PCHIPed cdf
        pdf2 = np.gradient(np.array(pchip_obj1(dpgrid), dtype=float), 
                    np.array(dpgrid, dtype=float))    

        #Get the integer values xout and corresponding pdf 
        N=2
        xout=np.convolve(dpgrid, np.ones((N,))/N, mode='valid')
        pdfout=np.convolve(pdf2, np.ones((N,))/N, mode='valid')
        sumpdfout=sum(pdfout)
    
    if withplot==True:
        
        plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
        
        print(df['model'].iloc[Index])
        #Linear Interpolation for CDF based on dpgrid
        probgrid =  np.interp(dpgrid, dp, cdf)
        #PDF based on linear interpolation
        pdf = np.gradient(np.array(probgrid, dtype=float), 
            np.array(dpgrid, dtype=float))
        
        if 'deaths' in df.columns:
            xlab = 'Cumulative Deaths'
            actual = df['deaths'].iloc[Index]
        else:
            xlab = 'Weekly Incidental Cases'
            actual = df['cases'].iloc[Index]
        
        #Start figure
        plt.figure(num=None, figsize=(8, 12), dpi=80, facecolor='w', edgecolor='k')

        plt.subplot(2, 1, 1)
        plt.scatter(dp, cdf, s=80, facecolors='none', edgecolors='y')
        plt.scatter(actual, 0.0, s=80, facecolors='k', edgecolors='k', marker="^")
        plt.plot(dpgrid, probgrid, 'r--')
        plt.plot(dpgrid,pchip_obj1(dpgrid), 'g--')
        #plt.xlabel(xlab, fontsize=20)
        plt.ylabel('CDF', fontsize=20) 
        plt.xticks(rotation=45)
        #----------#
        plt.subplot(2, 1, 2)
        plt.plot(dpgrid, pdf, 'g', label='Linear Interpolation')
        plt.plot(dpgrid, pdf2, 'r', label='Monotone Piecewise Cubic Interpolation')
        plt.legend(loc='best', prop={"size":14})
        plt.xlabel(xlab, fontsize=20)    
        plt.ylabel('PDF', fontsize=20) 
        plt.xticks(rotation=45)
        plt.savefig(figuresdirectory+'/exampleconversion.svg', 
            bbox_inches = 'tight',
            dpi=300)
    
    return (xout,pdfout,sum(pdfout),max(cdf)-min(cdf))    


def giveqandscore(df,Index) -> tuple:
    '''Give score for a model.
    
    Args:
        df (pandas pd): Scoreboard dataframe
        Index (int): Scoreboard Row index selection (model, forecast date, 
                    target date, quantiles and values)

    Returns:  -> tuple
        thescore (float): x-axis values
        sumpdfout (float): integrated version of calculated pdf
        prange (float): the cdf - sum(pdfout), should be close to max(cdf)-min(cdf)
        p (float): probability of actual data given model cdf
    '''     
    (xout,pdfout,sumpdfout,prange)=cdfpdf(df,Index,1)
    
    if 'cases' in df.columns:
        actual = df['cases'].iloc[Index]
    if 'deaths' in df.columns:
        actual = df['deaths'].iloc[Index]  
        
    indexofactual = np.where(xout == actual)
    
    if indexofactual[0].size == 0:
        p = 0
        thescore = np.NINF
    else:
        p = pdfout[indexofactual][0]
        thescore = 2*np.log(p)+1+np.log(actual)+np.log(2*np.pi)
    
    return (thescore, sumpdfout, prange, p, xout,pdfout)


def givePivotScoreFORECAST(Scoreboard,modeltypes) -> tuple:
    '''Give pivot table and averaged scoreboard with modeltypes.
    Pivot around forecast_date.
    
    Args:
        Scoreboard (pandas pd): Scoreboard dataframe
        modeltypes (int): Scoreboard Row index selection (model, forecast date, 
                    target date, quantiles and values)

    Returns:  -> tuple
        MerdfPRED (pandas pd): merged scoreboard on model scores vs delta in days
        pivMerdfPRED (pandas pd): pivoted MerdfPRED around forecast_date

    '''
    
    #Drop predictions from the same groups that were made on the same exact date and only 
    #take the final prediction
    Scoreboardxx = Scoreboard.sort_values('forecast_date').drop_duplicates(subset=['model',
                                                                                   'target_end_date'], keep='last').copy()
    
    MerdfPRED = (Scoreboardxx.merge(modeltypes, on=['model'])).copy()
    MerdfPRED.replace([np.inf, -np.inf], np.nan,inplace=True)
    MerdfPRED = (MerdfPRED.groupby(['model','modeltype','forecast_date'],
                                            as_index=False)[['delta','score']].agg(lambda x: list(x)))
    #MerdfPRED.dropna(subset=['score'],inplace=True)
    MerdfPRED['median'] = MerdfPRED.apply(lambda row : np.median(row['score']), axis = 1) 
    MerdfPRED['nanstd'] = MerdfPRED.apply(lambda row : np.nanstd(row['score']), axis = 1) 
    
    pivMerdfPRED = MerdfPRED.pivot(index='forecast_date', columns='model', values='median')
    
    return (MerdfPRED,pivMerdfPRED)


def givePivotScoreTARGET(Scoreboard,modeltypes) -> tuple:
    '''Give pivot table and averaged scoreboard with modeltypes.
    Pivot around target_end_date.
    
    Args:
        Scoreboard (pandas pd): Scoreboard dataframe
        modeltypes (int): Scoreboard Row index selection (model, forecast date, 
                    target date, quantiles and values)

    Returns:  -> tuple
        MerdfPRED (pandas pd): merged scoreboard on model scores vs delta in days
        pivMerdfPRED (pandas pd): pivoted MerdfPRED around target_end_date

    '''    
    
    MerdfPRED = (Scoreboard.merge(modeltypes, on=['model'])).copy()
    MerdfPRED.replace([np.inf, -np.inf], np.nan,inplace=True)
    MerdfPRED = (MerdfPRED.groupby(['model','modeltype','target_end_date'],
                                            as_index=False)[['delta','score']].agg(lambda x: list(x)))
    #MerdfPRED.dropna(subset=['score'],inplace=True)
    MerdfPRED['median'] = MerdfPRED.apply(lambda row : np.median(row['score']), axis = 1) 
    MerdfPRED['nanstd'] = MerdfPRED.apply(lambda row : np.nanstd(row['score']), axis = 1) 
    
    pivMerdfPRED = MerdfPRED.pivot(index='target_end_date', columns='model', values='median') 
    
    return (MerdfPRED,pivMerdfPRED)


def fix_scoreboard(scoreboard, kind='Case', quiet=False, plot=True):
    #Eliminate scores that do not have the proper score quantiles
    delete_row = scoreboard[scoreboard["proper"]==False].index
    scoreboard.drop(delete_row,inplace=True)
    scoreboard.reset_index(drop=True, inplace=True)
    if plot:
        scoresplots.plotdifferencescdfpdf(scoreboard, kind, quiet=quiet)
    modeltypesCases = datagrab.getmodeltypes(scoreboard, quiet=quiet)
    #Get the weekly forecast score rankings
    grouped =  scoreboard.groupby(['target_end_date','deltaW'])
    scoreboard['rank'] = grouped['score'].transform(lambda x: pd.factorize(-x, sort=True)[0]+1)
    return scoreboard
