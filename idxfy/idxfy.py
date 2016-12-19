import pandas as pd
import datetime as dt
import requests
import sys
import numpy as np

from pandas_datareader.data import DataReader
#%pylab inline

import yafin

class calendar:
    def __init__(self, reviewFreq, startdate='2014/12/31'):
        self.reviewFreq = reviewFreq
        self.reviewCutDates = self.__getReviewCutDates(startdate)
        
    def __getReviewCutDates(self, startdate):
        startdate = dt.datetime.strptime(startdate, '%Y/%m/%d')
        start_ = dt.date(startdate.year, startdate.month, startdate.day)
        end_ = dt.date.today()
        lstdts = np.array([dt.date(y,m,1) for y in range(start_.year, end_.year+1) for m in self.reviewFreq])
        x = np.array([(lstdts[i]>=start_) and (lstdts[i]<=end_) for i in range(len(lstdts))])
        return [lstdts[x][i] - dt.timedelta(lstdts[x][i].day) for i, v in enumerate(lstdts[x])]

class universe:
    
    def __init__(self, df, unitype, country='', industry='', exchange=''):
        self.type = unitype
        self.table = self.__applyFilter(df, country=country, industry=industry, exchange=exchange)  

    def __findField(self, df, field, fieldval): 
        """receives master table as dfs"""
        df = df[-df[field].isnull()].sort_values(by=field).reset_index(drop=True)
        stofo = [(fieldval.lower() in df.loc[i,field].lower()) for i in range(len(df))]
        return df[stofo].reset_index(drop=True)[[0,1,2,3,4]]
    
    def __applyFilter(self, df, country='', industry='', exchange=''):
        if len(country)>0:
            df = self.__findField(df, 'Country', country)
        if len(industry)>0:
            df = self.__findField(df, 'Category Name', industry)
        if len(industry)>0:
            df = self.__findField(df, 'Exchange', exchange)
        return df

    def getSecList(self, byName=True):
        if byName==True:
            return list(self.table['Name'])
        else:
            return list(self.table['Ticker'])
        
    def findSec(self, sto_ = ''): 
        """receives master table as df"""
        df = self.table
        return self.__findField(df, 'Name', sto_)   

    
class portfolio(universe):
    
    def __init__(self, uni, cutDate, country='', industry='', exchange='', maxMcap = np.nan, minMcap = np.nan):
        self.type = uni.type
        self.cutDate = cutDate
        self.table = self.applyFilter(uni.table, country=country, industry=industry, exchange=exchange)
        #self.__getMcap()
        #self.__getAdtv(3)
        
    def getMcap(self):
        if 'ffmcap' not in self.table.columns:
            seclist = self.getSecList(byName=False)
            self.table = pd.merge(self.table, yafin.get_funda(seclist, ['f6'], ['Float Shares'])[[0,2]], on='Ticker', how='left') #get shares
            dt_ = yafin.dt2yahoo(self.cutDate)
            dftemp = yafin.get_timeseries_pricing(seclist, dt_, dt_, field='Adj Close') #get close on the cut date
            dftemp.index.name = None
            dftemp = dftemp.T
            dftemp.columns=['Adj Price']
            dftemp['Ticker'] = dftemp.index
            self.table = pd.merge(self.table, dftemp, on='Ticker', how='left')
            self.table.loc[(self.table[self.table['Float Shares']=='N/A']).index,'Float Shares'] = np.nan
            self.table['Float Shares'] = self.table['Float Shares'].map(lambda x: float(x))
            self.table['ffmcap'] = self.table['Float Shares'] * self.table['Adj Price']
        return self.table

    def getAdtv(self, months):
        if 'adtv' + str(months) + 'm' not in self.table.columns: 
            dftemp = yafin.get_adtv(self.getSecList(byName=False), yafin.dt2yahoo(yafin.add_months(self.cutDate, -months)), yafin.dt2yahoo(self.cutDate))
            dftemp.rename(columns = {'adtv':'adtv' + str(months) + 'm'}, inplace = True)
            self.table = pd.merge(self.table, dftemp, on='Ticker', how='left')
        return self.table

    def getItem(self, item):
        dfitem = yafin.find_tag(item).reset_index(drop=True)
        if len(dfitem)==1:
            code = dfitem.loc[0,'tag']
            val = dfitem.loc[0,'value']
            if val not in self.table.columns: 
                seclist = self.getSecList(byName=False)
                self.table = pd.merge(self.table, yafin.get_funda(seclist, [code], [val])[[0,2]], on='Ticker', how='left')
            return self.table
        else:
            print('specify a unique item')
        
    def applyFilter(self, df, country='', industry='', exchange=''):
        if len(country)>0:
            df = self.__findField(df, 'Country', country)
        if len(industry)>0:
            df = self.__findField(df, 'Category Name', industry)
        if len(industry)>0:
            df = self.__findField(df, 'Exchange', exchange)
        return df
    
    def update(self, df):
        self.table = df

    def rank(self, by, ascending, prc = False, rankName = 'rk'):
        df = self.table
        idx = df.sort_values(by=by, ascending=ascending).index
        if prc == False:
            rk = np.arange(1,len(df)+1,1)
            df.loc[idx, rankName] = rk
            return df.sort_values('rk', ascending=True)
        else:
            rk = np.arange(len(df)-1,-1,-1)
            df.loc[idx, rankName] = rk
            df.loc[idx, rankName] = df.loc[idx, rankName] / max(df.loc[idx, rankName])
            return df.sort_values('rk', ascending=False)
    
    def selectTop(self, field, ct, ascending=False):
        idx = (self.table[(self.table[field]!=np.nan) & -(self.table[field].isnull()) & (self.table[field]!='N/A')]).index
        self.selection = self.table.loc[idx,:].sort_values(field, ascending=ascending).reset_index(drop=True).head(ct)
        return self.selection       