"""Functions for preparing scores."""

from tqdm import tqdm
import scipy.interpolate
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import os

def getscoreboard(groundtruth,model_target,otpfile='ScoreboardDataCases.pkl')-> pd.DataFrame:
    """Generates primary scores for all model competition entries
    Args:
        groundtruth (pd.DataFrame): The observed data
        model_target (str): 'Case' or 'Death'
        otpfile (str): Name of the scoreboard .pkl output file
    Returns:
        None 
    """    
    model_targets = ['Case', 'Death']
    if model_target not in model_targets:
        raise ValueError("Invalid sim type. Expected one of: %s" % model_targets)   
    
    #Read the predictions file 
    dfPREDx = pd.read_csv('../Data/all_dataONY.csv',
                         na_values = ['NA', 'no info', '.'], parse_dates=True)
    dfPREDx.drop_duplicates(subset=None, keep = 'first', inplace = True)
    
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
    if 'Deaths' in Scoreboard.columns:
        Scoreboard.rename(columns={'Deaths':'deaths'},inplace=True)
    
    Scoreboard['target_end_date']= pd.to_datetime(Scoreboard['target_end_date']) 
    Scoreboard['forecast_date']= pd.to_datetime(Scoreboard['forecast_date'])
    Scoreboard.insert(3,'delta',(Scoreboard.target_end_date-Scoreboard.forecast_date).dt.days)
    new = Scoreboard['model'].copy()
    Scoreboard['team']= Scoreboard['team'].str.cat(new, sep =":")
    Scoreboard.drop(columns=['model'],inplace=True)
    Scoreboard.rename(columns={'team':'model'},inplace=True)
    Scoreboard['deltaW'] = np.ceil(Scoreboard.delta/7)
    
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
    
    Scoreboard.replace([np.inf, -np.inf], np.nan,inplace=True)
    Scoreboard.dropna(inplace=True)
    Scoreboardx = Scoreboard.sort_values('forecast_date').drop_duplicates(subset=['model', 'target_end_date','deltaW'], keep='last').copy()
    Scoreboardx.reset_index(drop=True,inplace=True)
    Scoreboardx['scorecontr']=np.exp(-Scoreboardx['score']/2)
    
    Scoreboardx.to_pickle(otpfile)
    
def cdfpdf(df,Index,dV,withplot: bool = False):
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
    dp = mydf.dp.to_numpy().round() 
    
    #create grid x-axis
    dpgrid = np.arange(np.round(min(dp))+0.5,np.round(max(dp))-0.5,1)   
        
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
        dpgrid = np.arange(np.round(min(dp))+0.5,np.round(max(dp))-0.5,1)          
        
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
        
        print(df['model'].iloc[Index])
        #Linear Interpolation for CDF based on dpgrid
        probgrid =  np.interp(dpgrid, dp, cdf)
        #PDF based on linear interpolation
        pdf = np.gradient(np.array(probgrid, dtype=float), 
            np.array(dpgrid, dtype=float))
        
        #Start figure
        plt.figure(num=None, figsize=(8, 12), dpi=80, facecolor='w', edgecolor='k')

        plt.subplot(2, 1, 1)
        plt.scatter(dp, cdf, s=80, facecolors='none', edgecolors='y')
        plt.plot(dpgrid, probgrid, 'r--')
        plt.plot(dpgrid,pchip_obj1(dpgrid), 'g--')
        plt.xlabel('Cumulative Cases')
        plt.ylabel('CDF') 
        #----------#
        plt.subplot(2, 1, 2)
        plt.plot(dpgrid, pdf, 'g', label='Linear Interpolation')
        plt.plot(dpgrid, pdf2, 'r', label='Monotone Piecewise Cubic Interpolation')
        plt.legend(loc='best')
        plt.xlabel('Cumulative Cases')    
        plt.ylabel('PDF') 
    
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
    
    return (thescore, sumpdfout, prange, p)


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
    
    MerdfPRED['nanmean'] = MerdfPRED.apply(lambda row : np.nanmean(row['score']), axis = 1) 
    MerdfPRED['nanstd'] = MerdfPRED.apply(lambda row : np.nanstd(row['score']), axis = 1) 
    
    pivMerdfPRED = MerdfPRED.pivot(index='forecast_date', columns='model', values='nanmean')
    
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
    
    MerdfPRED['nanmean'] = MerdfPRED.apply(lambda row : np.nanmean(row['score']), axis = 1) 
    MerdfPRED['nanstd'] = MerdfPRED.apply(lambda row : np.nanstd(row['score']), axis = 1) 
    
    pivMerdfPRED = MerdfPRED.pivot(index='target_end_date', columns='model', values='nanmean')
    
    return (MerdfPRED,pivMerdfPRED)