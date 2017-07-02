import os
import re
import pandas as pd
from pandas.tseries.offsets import BDay #pandas has holiday functionality (tbd)
import datetime as dt
from backtest.externaldata.yahoo import get_timeseries_gross_return

def calculate(folderloc, portfoliosymbol):
    '''Return portfolio value history
    
    filenames have to be of the format '{portfoliosymbol}_{YYYYMMDD}.csv',
    the date in the filename is the effective date, i.e. weights are
    open-weights; the files need to be comma-separated and include at
    least the columns 'Symbol' and 'Weight'    
    
    Keyword arguments:
    folderloc -- path to folder that contains composition files (string)
    indexsymbol -- indexsymbol of index to be calculated
    '''
    basevalue = 100
    data = []
    
    datelist = sorted([pd.to_datetime(re.split('[\_\.]',x)[1],format='%Y%m%d').date() for x in os.listdir(folderloc) if x.startswith(portfoliosymbol+'_')])
    
    for filedate in datelist: 
        # define dates
        try:
            nextfiledate = [x for x in datelist if x > filedate][0]
        except:
            nextfiledate = dt.date.today()

        # read composition to dataframe
        fileloc = folderloc + '/' + portfoliosymbol + '_' + dt.datetime.strftime(filedate,'%Y%m%d') + '.csv'
        comp = pd.read_csv(fileloc, dtype={'Symbol':str})
        
        # load total return indices from QAD DataStream tables
        returns = get_timeseries_gross_return(list(comp['Symbol']),(filedate - BDay(1)).date(),(nextfiledate - BDay(1)).date()).fillna(method='ffill')

        # standardize
        returns = returns / returns.iloc[0]
        # apply weights
        for index, row in comp.iterrows():
            returns[row['Symbol']] = returns[row['Symbol']] * row['Weight']
        # sum up to get index timeseries
        portfolio = returns.sum(axis=1)
        # scale to match previous indexvalue
        portfolio = portfolio / portfolio.iloc[0] * basevalue
        # patch together
        data.append(portfolio)
        basevalue = portfolio[-1]

    portfolio = pd.DataFrame(pd.concat(data)).drop_duplicates()
    portfolio.columns = [portfoliosymbol+'_TR']

    return portfolio