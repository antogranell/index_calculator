import pandas as pd
import numpy as np
import datetime as dt
import requests
from pandas_datareader.data import DataReader


def find_stock(dfs, sto_ = ''): 
    #receives master table as dfs
    dfs = dfs[-dfs.Name.isnull()].sort('Name').reset_index(drop=True)
    stofo = [(sto_.lower() in dfs.loc[i,'Name'].lower()) for i in range(len(dfs))]
    return dfs[stofo].reset_index(drop=True)[[0,1,2,3,4]]
    
def getmaster(sheet):
    #sheet= 'Stock', 'ETF', 'Future', 'Index', 'Mutual Fund', 'Currency', 'Warrant', 'Bond'
    ploc = 'D:/Python/AntoTradingSystem/indexifyLib/idxfy/docs/'
    df = pd.read_excel(ploc + 'Yahoo Ticker Symbols - Jan 2016.xlsx', sheetname=sheet)
    cols = list(df.iloc[2,:-2])
    df = df.iloc[3:,:-2]
    df.columns = cols
    return df

def find_tag(tag_ = ''):
    ploc = 'D:/Python/AntoTradingSystem/indexifyLib/idxfy/docs/'
    dftags = pd.read_excel(ploc + 'yahootags.xlsx', header=None)

    dft = pd.DataFrame()
    for df0 in (dftags.iloc[:,[0,1]], dftags.iloc[:,[2,3]], dftags.iloc[:,[4,5]]):
        df0.columns = ['tag', 'value']
        dft = pd.concat([dft, df0], axis=0)

    dft = dft[-dft.tag.isnull()].sort('value').reset_index(drop=True)
    tagfo = [(tag_.lower() in dft.loc[i,'value'].lower()) for i in range(len(dft))]
    
    return dft[tagfo]
    
def get_replist(rows):
    import re
    rep = []
    i=0
    for line in rows:
        res = re.findall(r'\"(.+?)\"', str(line))
        for item in res:
            rep.append([item, item.replace("'","").replace(",","")])
            i=i+1
    lstrep = []
    for pair in rep:
        if pair[0]!=pair[1]:
            lstrep.append(pair)
    return lstrep

def rep_all(lstrep, row):
    for rep in lstrep:
        row = row.replace(rep[0], rep[1])
    return row

def get_funda_url(lst_tik, lst_codes): #'http://finance.yahoo.com/d/quotes.csv?s=GE+PTR+MSFT&f=snd1l1yr'
    url1 = "http://finance.yahoo.com/d/quotes.csv?"
    url2 = "s="; url3 = "&f=sn"
    for t in lst_tik:
        url2 = url2+str(t)+'+'
    url2 = url2[:-1]
    for c in lst_codes:
        url3 = url3+c

    url = url1 + url2 + url3
    return url

def get_funda_df(url, lst_val):
    r = requests.get(url)
    text = r.text
    rows = text.split('\n')
    lstrep = get_replist(rows)

    for i in range(len(rows)-1):
        rows[i] = rep_all(lstrep, rows[i])

    try:
        data = [x.replace('"','').split(',') for x in rows if x!='']
    except:
        data = ''

    df = pd.DataFrame(data)
    y=['Ticker', 'Name']
    for val in lst_val:
        y.append(val)
    df.columns = y
    return df

def get_funda(lst_tik, lst_codes, lst_val): #funda url + funda df
    urlf_ = get_funda_url(lst_tik, lst_codes)
    return get_funda_df(urlf_, lst_val)
    
def get_pricing_url(tkr, dtfrom, dtto=''): #'http://ichart.finance.yahoo.com/table.csv?s=WNGRF&a=1&b=21&c=2001&d=6&e=21&f=2016&g=d&ignore=.csv'
    fr = dt.datetime.strptime(dtfrom, '%Y-%m-%d')
    if dtto!='':
        to = dt.datetime.strptime(dtto, '%Y-%m-%d')
    else:
        to = dt.date.today()

    url1='http://ichart.finance.yahoo.com/table.csv?'
    url2 = "s=" + tkr
    url3='&a=' + str(fr.month) + '&b=' + str(fr.day) + '&c=' + str(fr.year)
    url4='&d=' + str(to.month) + '&e=' + str(to.day) + '&f=' + str(to.year)
    url5='&g=d&ignore=.csv'

    url_ = url1 + url2 + url3 + url4 + url5
    
    return url_
    
def get_pricing_df(url):
    r = requests.get(url)
    text = r.text
    rows = text.split('\n')

    try:
        data = [x.split(',') for x in rows if x!='']
    except:
        data = ''

    df = pd.DataFrame(data[1:])
    df.columns = data[0]
    for c in df.columns[1:]:
        df.loc[:,c] = df.loc[:,c].apply(lambda x: float(x))
    return df
    
def get_pricing_full(tkr, dtfrom, dtto=''):
    urlp_ = get_pricing_url(tkr, dtfrom, dtto='')
    return get_pricing_df(urlp_).sort('Date')
    
def get_pricing(tkr, dtfrom, dtto=''):
    urlp_ = get_pricing_url(tkr, dtfrom, dtto='')
    x = get_pricing_df(urlp_).sort('Date')
    x = x[['Date','Close', 'Adj Close']]
    x.index = x.Date
    x.index.name = None
    del x['Date']
    x.columns = ['cl', 'adjcl']
    x = x.fillna(method='pad')
    return x


def get_timeseries_pricing(symbols, startdate, enddate, field='Adj Close'):

    """Returns pricing/volume table field timeseries
    
    Keyword arguments:
    symbols -- Symbol or list of Symbols (string / [string])
    startdate -- timeseries start date (datetime.date(year, month, day))
    enddate -- timeseries end date (datetime.date(year, month, day))
    field -- 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'
    """
    if type(symbols) == str:
        symbols = [symbols]
    data = []
    for symbol in symbols:
        try:
            df = DataReader(symbol, 'yahoo', startdate, enddate)[[field]]
            df.columns = [symbol]
        except:
            df = pd.DataFrame(np.nan, index=pd.bdate_range(startdate, enddate), columns=[symbol])
        data.append(df)
    return pd.concat(data, axis=1, join='outer')
    
def get_adtv(symbols, startdate, enddate):

    """Returns pricing/volume table field timeseries
    fields used 'Close', 'Volume'
    """
    if type(symbols) == str:
        symbols = [symbols]
    data = []
    for symbol in symbols:
        try:
            df = DataReader(symbol, 'yahoo', startdate, enddate)[['Close', 'Volume']]
            df['adtv'] = df['Close'] * df['Volume']
            df.rename(columns = {'adtv':symbol}, inplace = True)
            df = df[symbol]
        except:
            df = pd.DataFrame(np.nan, index=pd.bdate_range(startdate, enddate), columns=[symbol])
        data.append(df)
    df = pd.concat(data, axis=1)
    df = pd.DataFrame(df.apply(np.mean, axis=0), columns=['adtv'])
    df['Ticker'] = df.index
    return df.reset_index(drop=True)
    
def dt2yahoo(dt):
    try:
        return str(dt.year) + '/' + str(dt.month) + '/' + str(dt.day)
    except:
        return str(dt_).replace('-','/')

def add_months(date, months):
    import calendar
    month = int(date.month - 1 + months)
    year = int(date.year + month / 12)
    month = int(month % 12 + 1)
    day = min(date.day, calendar.monthrange(year, month)[1])
    return dt.date(year, month, day)

def monthend(dts):
    ismonthend=(dts.day[0:len(dts)-1]>dts.day[1:len(dts)])
    dts1=dts[:len(dts)]
    monthend=dts1[ismonthend]
    return pd.DataFrame(monthend)